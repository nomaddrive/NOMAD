"""Cross-play evaluation script for PufferDrive."""
import torch
import os
import numpy as np
import logging
import argparse
import pufferlib
import yaml
import random
import json
import csv
from pathlib import Path
from box import Box
from typing import Dict

import madrona_gpudrive
from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.networks.actor_critic import NeuralNet
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.trajectory import LogTrajectory

from benchmark.evaluator import WOSACEvaluator
from wosac_evaluate import get_agent_size, get_road_edge_polylines, seed_everything, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def make_agent_from_ckpt(env, config, ckpt_path):
    """Create a policy and load from a specific checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}...")
    saved_cpt = torch.load(
        f=ckpt_path,
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
    
    # Load parameters
    policy.load_state_dict(saved_cpt["parameters"])
    return policy

class XPlayWOSACEvaluator(WOSACEvaluator):
    """Extended WOSACEvaluator for Cross-Play."""

    def collect_xplay_trajectories(self, env, policy1, policy2):
        """Roll out two policies in mixed fashion."""
        
        controlled_agent_mask = env.cont_agent_mask.clone()
        num_worlds = env.num_worlds
        num_agents = madrona_gpudrive.kMaxAgentCount

        trajectories = {
            "x": np.zeros((num_worlds, num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "y": np.zeros((num_worlds, num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "z": np.zeros((num_worlds, num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "heading": np.zeros((num_worlds, num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "id": np.zeros((num_worlds, num_agents, self.num_rollouts, self.sim_steps), dtype=np.int32),
            "rewards" : np.zeros((num_worlds, num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "goal_achieved_mask": np.zeros((num_worlds, num_agents, self.num_rollouts, self.sim_steps), dtype=bool),
            "cont_agent_mask": np.tile(controlled_agent_mask.cpu().numpy()[:,:,None,None], (1, 1, self.num_rollouts, self.sim_steps)),
            # Optional: track which policy controlled which agent
            "policy_id": np.zeros((num_worlds, num_agents, self.num_rollouts), dtype=np.int8) # 1 or 2
        }

        # Restore absolute XY
        mean_xy = env.sim.world_means_tensor().to_torch().to(env.device)[:, :2]
        mean_x = mean_xy[:, 0].unsqueeze(1)
        mean_y = mean_xy[:, 1].unsqueeze(1)

        # Infer action dimension
        space = env.single_action_space
        if hasattr(space, 'nvec'):
            inferred_action_dim = len(space.nvec)
        elif getattr(space, 'shape', None) is not None:
            inferred_action_dim = space.shape[0]
        else:
            inferred_action_dim = 1
            
        action_tensor = torch.zeros((num_worlds, num_agents, inferred_action_dim), dtype=torch.int64, device=env.device)

        for rollout_idx in range(self.num_rollouts):
            print(f"\rCollecting rollout {rollout_idx + 1}/{self.num_rollouts}...", end="", flush=True)
            obs = env.reset(controlled_agent_mask)
            
            # --- Assign Policies ---
            # obs is shape (num_valid_agents, obs_dim)
            num_valid_agents = obs.shape[0]
            
            # Randomly split indices for this rollout
            perm = torch.randperm(num_valid_agents, device=env.device)
            split_idx = num_valid_agents // 2
            
            # Indices into the FLATTENED obs/action arrays
            indices1 = perm[:split_idx]
            indices2 = perm[split_idx:]
            
            # Record policy IDs for this rollout
            # We need to map flattened indices back to (W, A) to store in trajectories["policy_id"]
            # controlled_agent_mask is (W, A, 1)
            # We can create a flat tensor of IDs and scatter it back
            flat_pids = torch.zeros(num_valid_agents, dtype=torch.int8, device=env.device)
            flat_pids[indices1] = 1
            flat_pids[indices2] = 2
            
            # Scatter back to (W, A)
            # Make sure mask is boolean and correct shape for indexing
            mask_bool = controlled_agent_mask.squeeze(-1).bool() # (W, A)
            
            # Use specific types to avoid issues
            full_pids = torch.zeros((num_worlds, num_agents), dtype=torch.int8, device=env.device)
            full_pids[mask_bool] = flat_pids
            
            trajectories["policy_id"][:, :, rollout_idx] = full_pids.cpu().numpy()

            reward = env.get_rewards(
                 collision_weight=self.config.environment.collision_weight,
                 off_road_weight=self.config.environment.off_road_weight,
                 goal_achieved_weight=self.config.environment.goal_achieved_weight,
            )
            
            infos = env.get_infos()
            
            for time_idx in range(self.sim_steps):
                # Record state
                agent_state = GlobalEgoState.from_tensor(
                    env.sim.absolute_self_observation_tensor(),
                    backend=env.backend,
                    device=env.device,
                )
                agent_state.restore_mean(mean_x=mean_x, mean_y=mean_y)

                trajectories["x"][..., rollout_idx, time_idx] = agent_state.pos_x.cpu().numpy()
                trajectories["y"][..., rollout_idx, time_idx] = agent_state.pos_y.cpu().numpy()
                trajectories["z"][..., rollout_idx, time_idx] = agent_state.pos_z.cpu().numpy()
                trajectories["heading"][..., rollout_idx, time_idx] = agent_state.rotation_angle.cpu().numpy()
                trajectories["id"][..., rollout_idx, time_idx] = agent_state.id.cpu().numpy()
                trajectories["rewards"][..., rollout_idx, time_idx] = reward.cpu().numpy()
                trajectories["goal_achieved_mask"][..., rollout_idx, time_idx] = infos.goal_achieved.cpu().numpy()

                # Get Actions
                # obs is already (N_valid, D)
                # We split obs into obs1 and obs2
                
                # Container for combined actions
                combined_actions = torch.zeros((num_valid_agents, inferred_action_dim), dtype=action_tensor.dtype, device=env.device)
                
                # Policy 1
                if len(indices1) > 0:
                    obs1 = obs[indices1]
                    actions1, _, _, _ = policy1(obs1, action=None, deterministic=True)
                    combined_actions[indices1] = actions1.to(dtype=combined_actions.dtype)
                
                # Policy 2
                if len(indices2) > 0:
                    obs2 = obs[indices2]
                    actions2, _, _, _ = policy2(obs2, action=None, deterministic=True)
                    combined_actions[indices2] = actions2.to(dtype=combined_actions.dtype)
                
                # Map back to action_tensor (W, A, D)
                # action_tensor needs to be zeroed or we just update masked parts
                action_tensor.zero_()
                action_tensor[mask_bool] = combined_actions
                
                # Step
                env.step_dynamics(action_tensor)
                obs = env.get_obs(controlled_agent_mask)
                infos = env.get_infos()
                reward = env.get_rewards(
                    collision_weight=self.config.environment.collision_weight,
                    off_road_weight=self.config.environment.off_road_weight,
                    goal_achieved_weight=self.config.environment.goal_achieved_weight,
                )

        for key in trajectories:
            if key == "policy_id":
                 trajectories[key] = trajectories[key].reshape(num_worlds * num_agents, self.num_rollouts)
            else:
                 trajectories[key] = trajectories[key].reshape(num_worlds * num_agents, self.num_rollouts, self.sim_steps)

        return trajectories


@torch.no_grad()
def evaluate_xplay(
    config,
    env,
    policy1,
    policy2,
    ckpt1_name,
    ckpt2_name,
):
    logging.info(f"Evaluating Cross-Play: {ckpt1_name} vs {ckpt2_name}")
    
    policy1.eval()
    policy2.eval()

    evaluator = XPlayWOSACEvaluator(config=config)

    # Collect GT
    gt_trajectories = evaluator.collect_ground_truth_trajectories(env)

    # Collect XPlay
    simulated_trajectories = evaluator.collect_xplay_trajectories(env, policy1, policy2)
    
    print(f"\\nCollected trajectories on {len(np.unique(gt_trajectories['scenario_id']))} scenarios.")

    agent_state = get_agent_size(env)
    road_edge_polylines = get_road_edge_polylines(env)

    # Compute metrics
    results = evaluator.compute_metrics(
        gt_trajectories,
        simulated_trajectories,
        agent_state,
        road_edge_polylines,
        config["eval"]["wosac_aggregate_results"],
    )

    return results


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Cross-play evaluation."
    )
    parse.add_argument("--config", "-c", default="baselines/eval/config/wosac_evaluate.yaml", type=str)
    parse.add_argument("--model_cpt_1", required=True, type=str, help="First model checkpoint")
    parse.add_argument("--model_cpt_2", required=True, type=str, help="Second model checkpoint")
    
    args = parse.parse_args()
    config = load_config(args.config)
    
    seed_everything(
        seed=config.train.seed,
        torch_deterministic=config.train.torch_deterministic,
    )

    # Determine Device
    device = config.train.device

    # Setup Results Dir
    ckpt1_stem = Path(args.model_cpt_1).stem
    ckpt2_stem = Path(args.model_cpt_2).stem
    
    results_dir = Path("eval_wosac_xplay") / f"{ckpt1_stem}_vs_{ckpt2_stem}"
    results_dir = results_dir / config.data_loader.root.split('/')[-1]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Dataloader & Env
    data_loader = SceneDataLoader(**config.data_loader)
    env_config = EnvConfig(**config.environment)
    
    # Force multi_discrete for now as per wosac_evaluate defaults usually
    config.train.action_type = "multi_discrete"

    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=config.environment.max_controlled_agents,
        device=device,
        action_type=config.train.action_type,
    )

    # Load Policies
    policy1 = make_agent_from_ckpt(env, config, args.model_cpt_1).to(device)
    policy2 = make_agent_from_ckpt(env, config, args.model_cpt_2).to(device)

    # Evaluate
    results = evaluate_xplay(
        config,
        env,
        policy1,
        policy2,
        ckpt1_stem,
        ckpt2_stem
    )
    
    # Save Results
    json_path = results_dir / "xplay_results.json"
    
    final_output = {
        "model_1": args.model_cpt_1,
        "model_2": args.model_cpt_2,
        "results": results
    }
    
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Saved results to {json_path}")
    
    env.close()
