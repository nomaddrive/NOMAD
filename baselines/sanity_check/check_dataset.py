"""Extract expert states and actions from Waymo Open Dataset."""
from gpudrive.datatypes.trajectory import LogTrajectory
import torch
import os
import numpy as np
import mediapy
import logging
import argparse
import pufferlib
import yaml
import random
import json
import csv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from box import Box
import matplotlib.pyplot as plt

from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.networks.actor_critic import NeuralNet
from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.trajectory import LogTrajectory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def map_to_closest_discrete_value(grid, cont_actions):
    """
    Find the nearest value in the action grid for a given expert action.
    """
    # Calculate the absolute differences and find the indices of the minimum values
    abs_diff = torch.abs(grid.unsqueeze(0) - cont_actions.unsqueeze(-1))
    indx = torch.argmin(abs_diff, dim=-1)

    # Gather the closest values based on the indices
    closest_values = grid[indx]

    return closest_values, indx


def get_log_trajectory(env) -> dict:
    """Load log trajectories and validity, restored to global coordinates.

    Returns dict with keys: pos_x, pos_y, yaw, valid; each (W, A, T)
    """
    means_xy = env.sim.world_means_tensor().to_torch().to(env.device)[:, :2]
    mean_x, mean_y = means_xy[:, 0], means_xy[:, 1]
    logs = LogTrajectory.from_tensor(
        env.sim.expert_trajectory_tensor(), env.num_worlds, env.max_agent_count, backend=env.backend
    )
    logs.restore_mean(mean_x=mean_x, mean_y=mean_y)
    pos_x = logs.pos_xy[..., 0].squeeze(-1).to(env.device)
    pos_y = logs.pos_xy[..., 1].squeeze(-1).to(env.device)
    yaw = logs.yaw.squeeze(-1).to(env.device)
    valid = logs.valids.squeeze(-1).to(torch.bool).to(env.device)
    return {"pos_x": pos_x, "pos_y": pos_y, "yaw": yaw, "valid": valid}


def get_current_sim_trajectory(env):
    """Obtain raw agent states (global XY, Z, heading, id)."""
    ego_state = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        backend=env.backend,
        device=env.device,
    )
    # Restore absolute XY using per-world means on the correct device
    mean_xy = env.sim.world_means_tensor().to_torch().to(env.device)[:, :2]
    mean_x = mean_xy[:, 0].unsqueeze(1)
    mean_y = mean_xy[:, 1].unsqueeze(1)
    ego_state.restore_mean(mean_x=mean_x, mean_y=mean_y)
    return (
        ego_state.pos_x,
        ego_state.pos_y,
        ego_state.pos_z,  # already absolute (no z-mean available)
        ego_state.rotation_angle,
        ego_state.id,
    )


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)
    

def evaluate(
    config,
    env,
    data_loader,
    results_dir,
    expert_policy_mode: str | None = None,
):
    """Evaluate a learned policy or an oracle expert.

    expert_policy_mode:
        None -> use provided policy.
        "multi_discrete" -> use discretized expert (indices into dx, dy, dyaw bins).
        "continuous" -> use raw continuous expert deltas (dx, dy, dyaw) directly.
    """
    logging.info("Evaluating the policy...")

    logging.info(f"Using expert policy for evaluation (mode={expert_policy_mode}).")
    assert config.environment.dynamics_model == "delta_local", "Expert evaluation currently only supported for delta_local dynamics."
    if expert_policy_mode not in {"multi_discrete", "continuous"}:
        raise ValueError(f"Invalid expert_policy_mode={expert_policy_mode}")

    worlds = {}

    for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        env.swap_data_batch(batch)

        controlled_agent_mask = env.cont_agent_mask.clone()
        num_worlds = env.num_worlds
        num_agents = env.max_cont_agents

        if expert_policy_mode is not None:
            expert_actions, _, _, _ = env.get_expert_actions()  # shape: (W, A, T, 3)
            if expert_policy_mode == "multi_discrete":
                # Convert continuous expert deltas to per-dimension discrete indices
                _, idx_dx = map_to_closest_discrete_value(grid=env.dx,   cont_actions=expert_actions[:, :, :, 0])
                _, idx_dy = map_to_closest_discrete_value(grid=env.dy,   cont_actions=expert_actions[:, :, :, 1])
                _, idx_dyaw = map_to_closest_discrete_value(grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2])
                expert_actions = (
                    torch.stack([idx_dx, idx_dy, idx_dyaw], dim=-1)
                    .to(torch.int64)[controlled_agent_mask]
                )  # (num_controlled_agents, T, 3) indices
            elif expert_policy_mode == "continuous":
                # Use raw continuous deltas (clip later if needed)
                expert_actions = expert_actions.to(torch.float32)[controlled_agent_mask]
            else:
                raise ValueError(f"Unsupported expert_policy_mode={expert_policy_mode}")

        space = env.single_action_space
        if hasattr(space, 'nvec'):
            inferred_action_dim = len(space.nvec)
        elif getattr(space, 'shape', None) is not None:
            inferred_action_dim = space.shape[0]
        else:
            inferred_action_dim = 1

        action_tensor = torch.zeros((num_worlds, num_agents, inferred_action_dim), dtype=torch.long, device=env.device)

        obs = env.reset(controlled_agent_mask)
        infos = env.get_infos()
        dones = env.get_dones().bool()  # (num_worlds, max_agents_in_scene) binary tensor

        for time_step in tqdm(range(env.episode_len)):


            for idx, scenario_id in enumerate(env.get_scenario_ids().values()):

                if scenario_id not in worlds:
                    worlds[scenario_id] = {
                        "done_timestep": 91,
                        "off_road": False,
                        "collied": False,
                    }
                controlled_done_mask = dones[idx][controlled_agent_mask[idx]]
                controlled_offroad_mask = infos.off_road[idx][controlled_agent_mask[idx]]
                controlled_collied_mask = infos.collided[idx][controlled_agent_mask[idx]]
                if controlled_done_mask.all() and worlds[scenario_id]["done_timestep"] == 91:
                    worlds[scenario_id]["done_timestep"] = time_step
                if torch.sum(controlled_offroad_mask) > 0. :
                    worlds[scenario_id]["off_road"] = True
                if torch.sum(controlled_collied_mask) > 0. :
                    worlds[scenario_id]["collied"] = True

            action = expert_actions[:, time_step]
            if expert_policy_mode == "continuous":
                # Ensure action dimensions align with env expectations (clip if bounds exist)
                # env.single_action_space for continuous likely a Box
                if hasattr(env.single_action_space, 'low') and hasattr(env.single_action_space, 'high'):
                    low = torch.as_tensor(env.single_action_space.low, device=action.device, dtype=action.dtype)
                    high = torch.as_tensor(env.single_action_space.high, device=action.device, dtype=action.dtype)
                    action = torch.clamp(action, low, high)

            action_tensor[controlled_agent_mask] = action
            env.step_dynamics(action_tensor)
            obs = env.get_obs(controlled_agent_mask)
            infos = env.get_infos()
            dones = env.get_dones().bool()
        
    # After all batches are processed, compute overall statistics

    # Plot histogram of done timesteps
    done_timesteps = [world["done_timestep"] for world in worlds.values() if "done_timestep" in world]
    collided = [world["collied"] for world in worlds.values() if "collied" in world]
    offroad = [world["off_road"] for world in worlds.values() if "off_road" in world]
    print(f"sum of collided worlds: {sum(collided)}/{len(collided)}")
    print(f"sum of off road worlds: {sum(offroad)}/{len(offroad)}")
    plt.figure(figsize=(10, 6))
    plt.hist(done_timesteps, bins=env.episode_len, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribution of Done Timesteps ({expert_policy_mode})")
    plt.xlabel("Timestep")
    plt.ylabel("Number of Worlds")
    plt.grid(axis='y', alpha=0.75)
    plot_path = results_dir / f"done_timestep_hist_{expert_policy_mode}.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved done timestep histogram to {plot_path}")


if __name__ == "__main__":

    parse = argparse.ArgumentParser(
        description="Generate expert actions and observations from Waymo Open Dataset."
    )
    parse.add_argument("--config", "-c", default="baselines/sanity_check/config/check_dataset.yaml", type=str, help="Path to the configuration file.")
    parse.add_argument("--eval-continuous-expert", action="store_true", help="Also evaluate continuous expert (delta_local + continuous action space).")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    seed_everything(
        seed=config.train.seed,
        torch_deterministic=config.train.torch_deterministic,
    )

    results_dir = Path("dataset_sanity_check") / str(args.eval_continuous_expert) / config.environment.collision_behavior
    results_dir = results_dir / config.data_loader.root.split('/')[-1]  # Use last part of data root as subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []

    # Make dataloader
    data_loader = SceneDataLoader(
        **config.data_loader
    )

    env_config = EnvConfig(
        **config.environment
    )

    # Optional: continuous expert evaluation
    if args.eval_continuous_expert:
        print("Evaluating continuous expert policy (continuous + delta_local)")
        # Build a continuous-action environment sharing same config except action_type
        cont_env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=data_loader,
            max_cont_agents=config.environment.max_controlled_agents,
            device=config.train.device,
            action_type="continuous",
        )
        evaluate(
            config,
            cont_env,
            data_loader,
            results_dir,
            expert_policy_mode="continuous",
        )
        cont_env.close()
        del cont_env

    # Make environment
    # You should NOT use PufferGPUDrive, which will reset the environment after each episode.
    # Instead, you should use GPUDriveTorchEnv directly and implement your own resampling logic
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=config.environment.max_controlled_agents,
        device=config.train.device,
        action_type=config.train.action_type,
    )

    print("Evaluating discretized expert policy (multi_discrete + delta_local)")
    evaluate(
        config,
        env,
        data_loader,
        results_dir,
        expert_policy_mode="multi_discrete",
    )

    env.close()
    del env
