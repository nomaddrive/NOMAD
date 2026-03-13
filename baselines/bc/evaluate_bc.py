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


def summary_row_from(name, summary_dict):
    fracs = summary_dict['fractions']
    return {
        'checkpoint': name,
        'num_envs': summary_dict['num_envs'],
        # Fraction means
        'goal_frac_mean': fracs['goal']['mean'],
        'collided_frac_mean': fracs['collided']['mean'],
        'offroad_frac_mean': fracs['offroad']['mean'],
        # Raw counts means
        'goal_count_mean': summary_dict['goal_reached']['mean'],
        'collided_count_mean': summary_dict['collided']['mean'],
        'offroad_count_mean': summary_dict['offroad']['mean'],
        'controlled_mean': summary_dict['controlled']['mean'],
        # ADE
        'ade_mean': summary_dict.get('ade', {}).get('mean', 0.0),
        # Returns
        'agent_return_mean': summary_dict.get('agent_returns', {}).get('mean', 0.0),
    }


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


class Helper:
    def __init__(self, num_worlds, max_cont_agents_per_env, num_agents, action_dim, device, action_dtype=torch.int64):
        """Runtime helper that stores per-step tensors used during evaluation.

        action_dtype: torch.int64 for (multi) discrete actions, torch.float32 for continuous.
        """
        self.num_worlds = num_worlds
        self.max_cont_agents_per_env = max_cont_agents_per_env
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = device
        self.action_dtype = action_dtype
        self.reset()

    def reset(self):
        self.actions = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env, self.action_dim),
            dtype=self.action_dtype, device=self.device
        )
        # self.rewards = torch.zeros(self.num_agents, dtype=torch.float32)
        # self.terminals = torch.zeros(self.num_agents, dtype=torch.bool)
        # self.truncations = torch.zeros(self.num_agents, dtype=torch.bool)
        # self.episode_returns = torch.zeros(
        #     self.num_agents, dtype=torch.float32
        # )
        # self.agent_episode_returns = torch.zeros(
        #     (self.num_worlds, self.max_cont_agents_per_env),
        #     dtype=torch.float32,
        # )
        # self.episode_lengths = torch.zeros(
        #     (self.num_worlds, self.max_cont_agents_per_env),
        #     dtype=torch.float32,
        # )
        # self.live_agent_mask = torch.ones(
        #     (self.num_worlds, self.max_cont_agents_per_env), dtype=bool
        # )
        self.collided_in_episode = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32, device=self.device
        )
        self.offroad_in_episode = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32, device=self.device
        )
        self.goal_reached_in_episode = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32, device=self.device
        )
        self.returns = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env),
            dtype=torch.float32, device=self.device
        )


def safe_fraction(numerators: np.ndarray, denominators: np.ndarray):
    mask = denominators > 0
    out = np.zeros_like(numerators, dtype=np.float64)
    out[mask] = numerators[mask] / denominators[mask]
    return out


def summarize_array(arr: np.ndarray, name: str):
    if arr.size == 0:
        return {
            'total': 0,
            'mean': 0.0,
            'var': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'p25': 0.0,
            'p75': 0.0,
        }
    # Be robust to NaN/inf values (e.g., ADE when no valid timesteps)
    arr = arr.astype(np.float64, copy=False)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.isnan(arr).all():
        return {
            'total': 0,
            'mean': 0.0,
            'var': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'p25': 0.0,
            'p75': 0.0,
        }
    return {
        'total': float(np.nansum(arr)),
        'mean': float(np.nanmean(arr)),
        'var': float(np.nanvar(arr, ddof=0)),  # population variance
        'std': float(np.nanstd(arr, ddof=0)),
        'min': float(np.nanmin(arr)),
        'max': float(np.nanmax(arr)),
        'median': float(np.nanmedian(arr)),
        'p25': float(np.nanpercentile(arr, 25)),
        'p75': float(np.nanpercentile(arr, 75)),
    }


def finalize_stats(stats: dict):
    """Build a rich summary (counts + fractions + dispersion) per metric."""
    # Convert lists to numpy arrays
    controlled = np.asarray(stats['num_controlled_agents'], dtype=np.int32)
    collided = np.asarray(stats['num_collided_agents'], dtype=np.int32)
    offroad = np.asarray(stats['num_offroad_agents'], dtype=np.int32)
    goal = np.asarray(stats['num_goal_reached_agents'], dtype=np.int32)
    truncated = np.asarray(stats['num_truncated_agents'], dtype=np.int32) if stats.get('num_truncated_agents') else np.zeros(0, dtype=np.int32)
    agent_returns = np.asarray(stats.get('agent_returns', []), dtype=np.float64)

    fractions = {}
    fractions['goal_fraction_per_env'] = safe_fraction(goal, controlled)
    fractions['collided_fraction_per_env'] = safe_fraction(collided, controlled)
    fractions['offroad_fraction_per_env'] = safe_fraction(offroad, controlled)
    if truncated.size > 0:
        fractions['truncated_fraction_per_env'] = safe_fraction(truncated, controlled)

    summary = {
        'num_envs': int(controlled.size),
        'controlled': summarize_array(controlled, 'controlled'),
        'goal_reached': summarize_array(goal, 'goal'),
        'collided': summarize_array(collided, 'collided'),
        'offroad': summarize_array(offroad, 'offroad'),
    }
    if truncated.size > 0:
        summary['truncated'] = summarize_array(truncated, 'truncated')
    if agent_returns.size > 0:
        summary['agent_returns'] = summarize_array(agent_returns, 'agent_returns')

    # Optional metrics
    if 'ade' in stats and len(stats['ade']) > 0:
        ade_arr = np.asarray(stats['ade'], dtype=np.float64)
        summary['ade'] = summarize_array(ade_arr, 'ade')

    # Add fraction summaries
    summary['fractions'] = {
        'goal': summarize_array(fractions['goal_fraction_per_env'], 'goal_fraction'),
        'collided': summarize_array(fractions['collided_fraction_per_env'], 'collided_fraction'),
        'offroad': summarize_array(fractions['offroad_fraction_per_env'], 'offroad_fraction'),
    }
    if 'truncated_fraction_per_env' in fractions:
        summary['fractions']['truncated'] = summarize_array(fractions['truncated_fraction_per_env'], 'truncated_fraction')

    # Store fraction arrays in original stats for potential downstream analysis
    stats['goal_fraction_per_env'] = fractions['goal_fraction_per_env'].tolist()
    stats['collided_fraction_per_env'] = fractions['collided_fraction_per_env'].tolist()
    stats['offroad_fraction_per_env'] = fractions['offroad_fraction_per_env'].tolist()
    if 'truncated_fraction_per_env' in fractions:
        stats['truncated_fraction_per_env'] = fractions['truncated_fraction_per_env'].tolist()

    return summary


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


def get_model_parameters(policy):
    """Helper function to count the number of trainable parameters."""
    params = filter(lambda p: p.requires_grad, policy.parameters())
    return sum([np.prod(p.size()) for p in params])


def make_agent(env, config):
    """Create a policy based on the environment."""

    if config.train.continue_training:
        print("Loading checkpoint...")
        # Load checkpoint
        saved_cpt = torch.load(
            f=config.train.model_cpt,
            map_location=config.train.device,
            weights_only=False,
        )

        single_action_space = env.single_action_space
        if hasattr(single_action_space, "nvec"):
            action_dims = list(map(int, single_action_space.nvec))
            policy = NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dims=action_dims,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )
        else:
            policy = NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dim=single_action_space.n,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )
        # Detect multi-discrete vs single-discrete in checkpoint
        # ckpt_action_dims = saved_cpt.get("action_dims", None)
        # if ckpt_action_dims is not None and ckpt_action_dims:
        #     policy = NeuralNet(
        #         input_dim=saved_cpt["model_arch"]["input_dim"],
        #         action_dims=ckpt_action_dims,
        #         hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        #         config=config.environment,
        #     )
        # else:
        #     policy = NeuralNet(
        #         input_dim=saved_cpt["model_arch"]["input_dim"],
        #         action_dim=saved_cpt["action_dim"],
        #         hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        #         config=config.environment,
        #     )

        # Load the model parameters
        policy.load_state_dict(saved_cpt["parameters"])

        return policy

    else:
        # Start from scratch
        # Detect multi-discrete action space (Gym MultiDiscrete has attribute nvec)
        single_action_space = env.single_action_space
        if hasattr(single_action_space, "nvec"):
            action_dims = list(map(int, single_action_space.nvec))
            return NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dims=action_dims,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )
        else:
            return NeuralNet(
                input_dim=config.train.network.input_dim,
                action_dim=single_action_space.n,
                hidden_dim=config.train.network.hidden_dim,
                dropout=config.train.network.dropout,
                config=config.environment,
            )
    

def evaluate(
    config,
    env,
    data_loader,
    policy,
    expert_policy_mode: str | None = None,
    make_video: bool = False,
    eval_num_batches: int = 5,
    video_save_path: str = "visualization/visualization_bc_eval",
):
    """Evaluate a learned policy or an oracle expert.

    expert_policy_mode:
        None -> use provided policy.
        "multi_discrete" -> use discretized expert (indices into dx, dy, dyaw bins).
        "continuous" -> use raw continuous expert deltas (dx, dy, dyaw) directly.
    """
    logging.info("Evaluating the policy...")

    if expert_policy_mode is not None:
        logging.info(f"Using expert policy for evaluation (mode={expert_policy_mode}).")
        assert config.environment.dynamics_model == "delta_local", "Expert evaluation currently only supported for delta_local dynamics."
        assert policy is None, "Policy must be None when using expert expert_policy_mode."
        if expert_policy_mode not in {"multi_discrete", "continuous"}:
            raise ValueError(f"Invalid expert_policy_mode={expert_policy_mode}")
    else:
        policy.eval()

    stats = dict(
        num_controlled_agents=[],
        num_collided_agents=[],
        num_offroad_agents=[],
        num_goal_reached_agents=[],
        num_truncated_agents=[],
        ade=[],  # per-environment ADE
        agent_returns=[],  # per-agent cumulative returns
    )
    if make_video:
        video_save_path = Path(video_save_path)
        os.makedirs(video_save_path, exist_ok=True)
        done_envs = []
        frames = {f"env_{i}": [] for i in range(env.num_worlds)}

    for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        # try:
        #     env.swap_data_batch(batch)
        #     logging.info(f"Swapped in batch {i} successfully.")
        # except ValueError as e:
        #     logging.warning(f"Stop swapping due to ValueError in swap_data_batch: {e}. Done")
        #     break
        if i > 0:
            break

        log_trajectory = get_log_trajectory(env)
        sim_pos_x_list, sim_pos_y_list, sim_yaw_list = [], [], []
        sim_dones_list = []

        # Reset
        controlled_agent_mask = env.cont_agent_mask.clone()
        dead_agent_mask = ~env.cont_agent_mask.clone()
        obs = env.reset(controlled_agent_mask)
        dones = env.get_dones()  # (num_worlds, max_agents_in_scene) binary tensor
        pos_x, pos_y, pos_z, heading, id = get_current_sim_trajectory(env)

        sim_pos_x_list.append(pos_x)
        sim_pos_y_list.append(pos_y)
        sim_yaw_list.append(heading)
        sim_dones_list.append(dones)

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

        # Determine action dimension robustly
        if expert_policy_mode is not None:
            # expert_actions already shaped (num_controlled_agents, T, D)
            inferred_action_dim = expert_actions.shape[-1]
        else:
            space = env.single_action_space
            if hasattr(space, 'nvec'):
                inferred_action_dim = len(space.nvec)
            elif getattr(space, 'shape', None) is not None:
                inferred_action_dim = space.shape[0]
            else:
                inferred_action_dim = 1

        helper = Helper(
            num_worlds=env.num_worlds,
            max_cont_agents_per_env=env.max_cont_agents,
            num_agents=controlled_agent_mask.sum().item(),
            action_dim=inferred_action_dim,
            device=env.device,
            action_dtype=(torch.float32 if expert_policy_mode == "continuous" else torch.int64),
        )

        for time_step in tqdm(range(env.episode_len)):

            if expert_policy_mode is not None:
                action = expert_actions[:, time_step]
                if expert_policy_mode == "continuous":
                    # Ensure action dimensions align with env expectations (clip if bounds exist)
                    # env.single_action_space for continuous likely a Box
                    if hasattr(env.single_action_space, 'low') and hasattr(env.single_action_space, 'high'):
                        low = torch.as_tensor(env.single_action_space.low, device=action.device, dtype=action.dtype)
                        high = torch.as_tensor(env.single_action_space.high, device=action.device, dtype=action.dtype)
                        action = torch.clamp(action, low, high)
            else:
                with torch.no_grad():
                    action, _, _, _ = policy(obs, action=None, deterministic=True)
                
            helper.actions[controlled_agent_mask] = action
            env.step_dynamics(helper.actions)

            next_obs = env.get_obs(controlled_agent_mask)
            # Get rewards, terminal (dones) and info
            reward = env.get_rewards(
                collision_weight=config.environment.collision_weight,
                off_road_weight=config.environment.off_road_weight,
                goal_achieved_weight=config.environment.goal_achieved_weight,
            )
            dones = env.get_dones() # (num_worlds, max_agents_in_scene) binary tensor
            """
            Info:
                off_road: (num_worlds, max_agents_in_scene) binary tensor
                collided: (num_worlds, max_agents_in_scene) binary tensor
                goal_achieved: (num_worlds, max_agents_in_scene) binary tensor
            """
            info = env.get_infos()
            # Accumulate (logical OR) whether an agent has EVER experienced each event this episode.
            # Using torch.maximum keeps dtype float32 while behaving like boolean OR for 0/1 tensors.
            helper.offroad_in_episode = torch.maximum(
                helper.offroad_in_episode, info.off_road.to(helper.offroad_in_episode.dtype)
            )
            helper.collided_in_episode = torch.maximum(
                helper.collided_in_episode, info.collided.to(helper.collided_in_episode.dtype)
            )
            helper.goal_reached_in_episode = torch.maximum(
                helper.goal_reached_in_episode, info.goal_achieved.to(helper.goal_reached_in_episode.dtype)
            )
            helper.returns += reward

            pos_x, pos_y, pos_z, heading, id = get_current_sim_trajectory(env)

            sim_pos_x_list.append(pos_x)
            sim_pos_y_list.append(pos_y)
            sim_yaw_list.append(heading)
            sim_dones_list.append(dones)

            # Update
            obs = next_obs
            dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

            # Render
            if make_video:
                env_indices = [i for i in range(env.num_worlds) if i not in done_envs]
                figs = env.vis.plot_simulator_state(
                    env_indices=env_indices,
                    time_steps=[time_step]*env.num_worlds,
                    zoom_radius=100,
                    #center_agent_indices=[0]*env.num_worlds,
                )
                for i, env_id in enumerate(env_indices):
                    frames[f"env_{env_id}"].append(img_from_fig(figs[i])) 

                # Check if done
                for env_id in range(env.num_worlds):
                    if dones[env_id].all():
                        done_envs.append(env_id)

            
            if (dead_agent_mask == True).all():
                logging.info("All agents are done in this batch. Moving to the next batch. \n")
                # Complement the position and done lists to have same length
                T_current = len(sim_pos_x_list)
                if T_current < env.episode_len + 1:
                    last_pos_x = sim_pos_x_list[-1]
                    last_pos_y = sim_pos_y_list[-1]
                    last_yaw = sim_yaw_list[-1]
                    last_dones = sim_dones_list[-1]
                    for _ in range(env.episode_len + 1 - T_current):
                        sim_pos_x_list.append(last_pos_x)
                        sim_pos_y_list.append(last_pos_y)
                        sim_yaw_list.append(last_yaw)
                        sim_dones_list.append(last_dones)

                # Per-environment counts (shape: num_worlds)
                # Only consider controlled agents when tallying.
                per_env_controlled = controlled_agent_mask.sum(dim=1)
                per_env_collided = ((helper.collided_in_episode > 0) & controlled_agent_mask).sum(dim=1)
                per_env_offroad = ((helper.offroad_in_episode > 0) & controlled_agent_mask).sum(dim=1)
                per_env_goal = ((helper.goal_reached_in_episode > 0) & controlled_agent_mask).sum(dim=1)

                for env_id in range(env.num_worlds):
                    returns_env = helper.returns[env_id][controlled_agent_mask[env_id]].sum()
                    stats['agent_returns'].append(returns_env.item())

                # Store (extend) so we retain one entry per environment for this batch
                stats['num_controlled_agents'].extend(per_env_controlled.tolist())
                stats['num_collided_agents'].extend(per_env_collided.tolist())
                stats['num_offroad_agents'].extend(per_env_offroad.tolist())
                stats['num_goal_reached_agents'].extend(per_env_goal.tolist())
                # Truncations: an agent that finished without goal/collision & not off-road (optional)
                # Define truncation here if needed; placeholder zeros for now to keep list lengths aligned.
                if 'num_truncated_agents' in stats:
                    per_env_truncated = (
                        per_env_controlled - per_env_collided - per_env_offroad - per_env_goal
                    ).clamp(min=0)
                    stats['num_truncated_agents'].extend(per_env_truncated.tolist())
                break
        if make_video:
            for i in range(env.num_worlds):
                filename = env.get_env_filenames()[i][:-5]
                mediapy.write_video(path=video_save_path / f"{filename}.mp4", images=frames[f'env_{i}'], fps=10)
            break

        # ADE per world against logs for evaluated agents
        T_sim = len(sim_pos_x_list)
        pos_x_sim = torch.stack(sim_pos_x_list, dim=0)  # (T,W,A)
        pos_y_sim = torch.stack(sim_pos_y_list, dim=0)
        done_sim = torch.stack(sim_dones_list, dim=0)  # (T,W,A) bool
        # We need to deal with agents' position.
        # After agents done, their positions will be random numbers in pos_x_sim and pos_y_sim.
        # We want to modify these to be the last valid position (done position) for ADE computation.
        # For each (world, agent), find the first timestep where done becomes True
        # and hold the position constant from that timestep forward.
        T, W, A = pos_x_sim.shape
        for w in range(W):
            for a in range(A):
                dones_wa = done_sim[:, w, a]
                if torch.any(dones_wa):
                    first_done_idx = torch.nonzero(dones_wa, as_tuple=False)[0].item()
                    # Use Python scalars to avoid overlapping memory issues in assignment
                    x_last = pos_x_sim[first_done_idx, w, a].item()
                    y_last = pos_y_sim[first_done_idx, w, a].item()
                    pos_x_sim[first_done_idx:, w, a] = x_last
                    pos_y_sim[first_done_idx:, w, a] = y_last

        # Align to log length
        T_log = log_trajectory["pos_x"].shape[-1]
        T = min(T_sim, T_log)
        pos_x_sim = pos_x_sim[:T]
        pos_y_sim = pos_y_sim[:T]
        # Bring time to last axis for index convenience
        pos_x_sim = pos_x_sim.permute(1, 2, 0)  # (W,A,T)
        pos_y_sim = pos_y_sim.permute(1, 2, 0)
        valid_log = log_trajectory["valid"][
            :, :, :T
        ]
        ade_per_env = []
        for w in range(env.num_worlds):
            # Compute ADE for controlled agents only
            maskA = controlled_agent_mask[w]
            if not maskA.any():
                ade_per_env.append(float('nan'))
                continue
            dx = pos_x_sim[w, maskA, :] - log_trajectory["pos_x"][w, maskA, :T]
            dy = pos_y_sim[w, maskA, :] - log_trajectory["pos_y"][w, maskA, :T]
            d = torch.sqrt(dx * dx + dy * dy)
            vmask = valid_log[w, maskA, :]
            # avoid zero division; compute mean over valid timesteps per agent then average agents
            ade_agents = []
            for ai in range(d.shape[0]):
                dm = d[ai][vmask[ai]]
                if dm.numel() > 0:
                    ade_agents.append(dm.mean().item())
            ade = float(np.mean(ade_agents)) if ade_agents else float('nan')
            ade_per_env.append(ade)

        # store ADE per environment for this batch
        stats['ade'].extend(ade_per_env)

        if i >= eval_num_batches:  # We only do a few batches for evaluation
            break

    # Compute aggregate statistics (mean, variance, etc.) across environments
    summary = finalize_stats(stats)

    return stats, summary


if __name__ == "__main__":

    parse = argparse.ArgumentParser(
        description="Generate expert actions and observations from Waymo Open Dataset."
    )
    parse.add_argument("--config", "-c", default="baselines/bc/config/bc_evaluate.yaml", type=str, help="Path to the configuration file.")
    parse.add_argument("--render", "-r", action="store_true", help="Whether to render the environment.")
    parse.add_argument("--eval-continuous-expert", action="store_true", help="Also evaluate continuous expert (delta_local + continuous action space).")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    seed_everything(
        seed=config.train.seed,
        torch_deterministic=config.train.torch_deterministic,
    )

    if args.eval_continuous_expert:
        results_dir = Path("eval_results_with_continous_expert") / config.environment.collision_behavior / config.train.model_cpt_dir.split('/')[-1]
    else:
        results_dir = Path("eval_results_more_detailed") / config.environment.collision_behavior / config.train.model_cpt_dir.split('/')[-1]
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
        cont_stats, cont_summary = evaluate(
            config,
            cont_env,
            data_loader,
            policy=None,
            expert_policy_mode="continuous",
            make_video=args.render,
            video_save_path=results_dir / "continuous_expert_videos",
        )
        print(f"Results for continuous expert policy: {cont_stats}")
        print(f"Summary: {cont_summary}")
        aggregate_rows.append(summary_row_from("expert_policy_continuous", cont_summary))
        if not args.render:
            with open(results_dir / "expert_policy_continuous.stats.json", 'w') as f:
                json.dump({'checkpoint': 'expert_policy_continuous', 'stats': cont_stats, 'summary': cont_summary}, f, indent=2)
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
    stats, summary = evaluate(
        config,
        env,
        data_loader,
        policy=None,
        expert_policy_mode="multi_discrete",
        make_video=args.render,
        video_save_path=results_dir / "discrete_expert_videos",
    )
    print(f"Results for discretized expert policy: {stats}")
    print(f"Summary: {summary}")

    aggregate_rows.append(summary_row_from("expert_policy_multi_discrete", summary))

    # Save discretized expert policy results
    policy_name = "expert_policy_multi_discrete"
    if not args.render:
        with open(results_dir / f"{policy_name}.stats.json", 'w') as f:
            json.dump({'checkpoint': policy_name, 'stats': stats, 'summary': summary}, f, indent=2)

    print(f"Evaluating random model")
    config.train.continue_training = False
    # Create policy
    policy = make_agent(env=env, config=config).to(
        config.train.device
    )
    config.train.network.num_parameters = get_model_parameters(policy)

    stats, summary = evaluate(config, env, data_loader, policy, make_video=args.render, video_save_path=results_dir / "random_policy_videos")
    print(f"Results for random policy (raw): {stats}")
    print(f"Summary: {summary}")

    # Save random policy results
    random_name = "random_policy"
    with open(results_dir / f"{random_name}.stats.json", 'w') as f:
        json.dump({'checkpoint': random_name, 'stats': stats, 'summary': summary}, f, indent=2)

    aggregate_rows.append(summary_row_from(random_name, summary))

    # Get all *.pt files in the model_cpt_dir
    config.train.continue_training = True
    model_cpt_dir = config.train.model_cpt_dir
    model_files = [f for f in os.listdir(model_cpt_dir) if f.endswith(".pt")][:-20]
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint files found in {model_cpt_dir}")

    for file in model_files:
        print(f"Evaluating model checkpoint: {file}")
        config.train.model_cpt = os.path.join(model_cpt_dir, file)

        # Create policy
        policy = make_agent(env=env, config=config).to(
            config.train.device
        )
        
        config.train.network.num_parameters = get_model_parameters(policy)

        stats, summary = evaluate(config, env, data_loader, policy, make_video=args.render, video_save_path=results_dir / "model_videos" / Path(file).stem)
        print(f"Results for {file} (raw): {stats}")
        print(f"Summary: {summary}")

        # Save per-checkpoint JSON
        ckpt_name = Path(file).stem
        if not args.render:
            with open(results_dir / f"{ckpt_name}.stats.json", 'w') as f:
                json.dump({'checkpoint': ckpt_name, 'stats': stats, 'summary': summary}, f, indent=2)

        aggregate_rows.append(summary_row_from(ckpt_name, summary))

    # Save aggregate JSON
    if not args.render:
        with open(results_dir / "aggregate_results.json", 'w') as f:
            json.dump({'results': aggregate_rows}, f, indent=2)

    # Save aggregate CSV
    if aggregate_rows and not args.render:
        csv_path = results_dir / "aggregate_results.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print(f"Aggregate CSV written to {csv_path}")

    env.close()
    del env
