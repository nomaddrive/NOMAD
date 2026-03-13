"""Extract expert states and actions from Waymo Open Dataset."""
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
    def __init__(self, num_worlds, max_cont_agents_per_env, num_agents, action_dim, device):
        self.num_worlds = num_worlds
        self.max_cont_agents_per_env = max_cont_agents_per_env
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        self.actions = torch.zeros(
            (self.num_worlds, self.max_cont_agents_per_env, self.action_dim),
            dtype=torch.int64, device=self.device
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
    return {
        'total': float(arr.sum()),
        'mean': float(arr.mean()),
        'var': float(arr.var(ddof=0)),  # population variance
        'std': float(arr.std(ddof=0)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
    }


def finalize_stats(stats: dict):
    """Build a rich summary (counts + fractions + dispersion) per metric."""
    # Convert lists to numpy arrays
    controlled = np.asarray(stats['num_controlled_agents'], dtype=np.int32)
    collided = np.asarray(stats['num_collided_agents'], dtype=np.int32)
    offroad = np.asarray(stats['num_offroad_agents'], dtype=np.int32)
    goal = np.asarray(stats['num_goal_reached_agents'], dtype=np.int32)
    truncated = np.asarray(stats['num_truncated_agents'], dtype=np.int32) if stats.get('num_truncated_agents') else np.zeros(0, dtype=np.int32)

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
        # policy.load_state_dict(saved_cpt["parameters"])
        policy.load_state_dict(saved_cpt)

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
    expert_policy=False,
    make_video=False,
):
    """Evaluate the policy on the environment."""
    logging.info("Evaluating the policy...")

    if expert_policy:
        logging.info("Using expert policy for evaluation.")
        assert config.environment.dynamics_model == "delta_local"
        assert config.train.action_type == "multi_discrete"
        assert policy is None
    else:
        policy.eval()

    stats = dict(
        num_controlled_agents=[],
        num_collided_agents=[],
        num_offroad_agents=[],
        num_goal_reached_agents=[],
        num_truncated_agents=[],
    )
    if make_video:
        VIDEO_SAVE_PATH = Path(f"visualization/visualization_bc_eval/{config.data_loader.root.split('/')[-1]}/")
        os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
        done_envs = []
        frames = {f"env_{i}": [] for i in range(env.num_worlds)}

    for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        try:
            env.swap_data_batch(batch)
            logging.info(f"Swapped in batch {i} successfully.")
        except ValueError as e:
            logging.warning(f"Skipping a batch due to ValueError in swap_data_batch: {e}. Done")
            break

        # Reset
        controlled_agent_mask = env.cont_agent_mask.clone()
        dead_agent_mask = ~env.cont_agent_mask.clone()
        obs = env.reset(controlled_agent_mask)
        if expert_policy:
            expert_actions, _, _, _ = env.get_expert_actions()
            # Multi-discrete (separate dx, dy, dyaw bins) for delta_local dynamics
            # Get per-dimension discrete indices (we keep indices, not joint encoding)
            _, idx_dx = map_to_closest_discrete_value(grid=env.dx,   cont_actions=expert_actions[:, :, :, 0])
            _, idx_dy = map_to_closest_discrete_value(grid=env.dy,   cont_actions=expert_actions[:, :, :, 1])
            _, idx_dyaw = map_to_closest_discrete_value(grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2])

            # Shape: (worlds, agents, time, 3)
            expert_actions = torch.stack([idx_dx, idx_dy, idx_dyaw], dim=-1).to(torch.int64)[controlled_agent_mask]

        helper = Helper(
            num_worlds=env.num_worlds,
            max_cont_agents_per_env=env.max_cont_agents,
            num_agents=controlled_agent_mask.sum().item(),
            action_dim=env.single_action_space.shape[0] if hasattr(env.single_action_space, "shape") else 1,
            device=env.device
        )

        for time_step in tqdm(range(env.episode_len)):

            if expert_policy:
                action = expert_actions[:, time_step]
            else:
                with torch.no_grad():
                    action, _, _, _ = policy(obs, action=None, deterministic=True)
                
            helper.actions[controlled_agent_mask] = action
            env.step_dynamics(helper.actions)

            next_obs = env.get_obs(controlled_agent_mask)
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
                logging.info("All agents are done in this batch. Moving to the next batch.")
                # Per-environment counts (shape: num_worlds)
                # Only consider controlled agents when tallying.
                per_env_controlled = controlled_agent_mask.sum(dim=1)
                per_env_collided = ((helper.collided_in_episode > 0) & controlled_agent_mask).sum(dim=1)
                per_env_offroad = ((helper.offroad_in_episode > 0) & controlled_agent_mask).sum(dim=1)
                per_env_goal = ((helper.goal_reached_in_episode > 0) & controlled_agent_mask).sum(dim=1)

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
                mediapy.write_video(path=VIDEO_SAVE_PATH / f"{filename}.mp4", images=frames[f'env_{i}'], fps=10)
            break

        if i > 5:  # We only do a few batches for evaluation
            break



if __name__ == "__main__":

    parse = argparse.ArgumentParser(
        description="Generate expert actions and observations from Waymo Open Dataset."
    )
    parse.add_argument("--config", "-c", default="baselines/bc/config/bc_visualize.yaml", type=str, help="Path to the configuration file.")
    parse.add_argument("--render", "-r", action="store_true", help="Whether to render the environment.")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    seed_everything(
        seed=config.train.seed,
        torch_deterministic=config.train.torch_deterministic,
    )

    # Make dataloader
    data_loader = SceneDataLoader(
        **config.data_loader
    )

    env_config = EnvConfig(
        **config.environment
    )

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

    # Get all *.pt files in the model_cpt_dir
    config.train.continue_training = True
    model_cpt_dir = config.train.model_cpt_dir
    model_files = [f for f in os.listdir(model_cpt_dir) if f.endswith(".pt")]
    print("Found model checkpoint files:")
    print(model_files)
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint files found in {model_cpt_dir}")

    for file in model_files:
        if file != "model_bc_multi_discrete_debug_000199.pt":
            continue
        print(f"Evaluating model checkpoint: {file}")
        config.train.model_cpt = os.path.join(model_cpt_dir, file)

        # Create policy
        policy = make_agent(env=env, config=config).to(
            config.train.device
        )
        
        config.train.network.num_parameters = get_model_parameters(policy)

        evaluate(config, env, data_loader, policy, make_video=args.render)

    env.close()
    del env
