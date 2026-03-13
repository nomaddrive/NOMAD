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
    

@torch.no_grad()
def evaluate(
    config,
    env,
    data_loader,
    policy,
    video_save_path: str = "visualization/visualization_bc_eval",
):
    """Evaluate a learned policy or an oracle expert.

    expert_policy_mode:
        None -> use provided policy.
        "multi_discrete" -> use discretized expert (indices into dx, dy, dyaw bins).
        "continuous" -> use raw continuous expert deltas (dx, dy, dyaw) directly.
    """
    logging.info("Evaluating the policy...")

    policy.eval()

    video_save_path = Path(video_save_path)
    os.makedirs(video_save_path, exist_ok=True)
    done_envs = []
    frames = {f"env_{i}": [] for i in range(env.num_worlds)}


    # Reset
    controlled_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    obs = env.reset(controlled_agent_mask)
    dones = env.get_dones()  # (num_worlds, max_agents_in_scene) binary tensor

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
        action_dtype=torch.int64,
    )

    for time_step in tqdm(range(env.episode_len)):

        action, _, _, _ = policy(obs, action=None, deterministic=True)
            
        helper.actions[controlled_agent_mask] = action
        env.step_dynamics(helper.actions)
        next_obs = env.get_obs(controlled_agent_mask)
        dones = env.get_dones() # (num_worlds, max_agents_in_scene) binary tensor

        # Update
        obs = next_obs
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        # Render
        env_indices = [i for i in range(env.num_worlds) if i not in done_envs]
        figs = env.vis.plot_simulator_state(
            env_indices=env_indices,
            time_steps=[time_step]*env.num_worlds,
            zoom_radius=200,
            #center_agent_indices=[0]*env.num_worlds,
        )
        for i, env_id in enumerate(env_indices):
            frames[f"env_{env_id}"].append(img_from_fig(figs[i])) 

        # Check if done
        for env_id in range(env.num_worlds):
            if dones[env_id].all():
                done_envs.append(env_id)

        
        if (dead_agent_mask == True).all():
            break

    for i in range(env.num_worlds):
        filename = Path(env.get_env_filenames()[i])
        mediapy.write_video(path=video_save_path / f"{filename}.mp4", images=frames[f'env_{i}'], fps=10)


if __name__ == "__main__":

    parse = argparse.ArgumentParser(
        description="Generate expert actions and observations from Waymo Open Dataset."
    )
    parse.add_argument("--config", "-c", default="baselines/render/config/render.yaml", type=str, help="Path to the configuration file.")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    seed_everything(
        seed=config.train.seed,
        torch_deterministic=config.train.torch_deterministic,
    )

    results_dir = Path("visualization_icml") / config.train.model_cpt.split('/')[-1][:-3]
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

    config.train.continue_training = True
    policy = make_agent(env=env, config=config).to(config.train.device)
    evaluate(config, env, data_loader, policy, video_save_path=results_dir / "model_videos")
