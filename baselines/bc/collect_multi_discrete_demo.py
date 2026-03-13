"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import mediapy
import logging
import argparse
import pufferlib
import pufferlib.pytorch
import yaml
import random
import os
from pathlib import Path
from tqdm import tqdm

from box import Box

from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Experience:
    """Flat tensor storage (buffer) and array views for faster indexing."""

    def __init__(
        self,
        data_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        device="cpu",
    ):
        if not isinstance(obs_dtype, torch.dtype):
            obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]

        self.obs = torch.zeros(data_size, *obs_shape, dtype=obs_dtype, device=device)
        self.actions = torch.zeros(data_size, *atn_shape, dtype=int, device=device)

        self.data_size = data_size
        self.ptr = 0

    @property
    def full(self):
        return self.ptr >= self.data_size

    def store(self, obs, action, mask):
        # Mask learner and Ensure indices do not exceed data size
        if len(torch.where(mask)[0].cpu().numpy()) + self.ptr > self.data_size:
            raise IndexError("Experience buffer overflow. Reduce data_size or increase frequency of saving.")
        ptr = self.ptr
        alive_obs = obs[mask]
        alive_action = action[mask]
        indices = torch.where(mask)[0].cpu().numpy()
        end = ptr + len(indices)
        self.obs[ptr:end] = alive_obs.cpu()
        self.actions[ptr:end] = alive_action.cpu()
        self.ptr = end

    def save(self, path='./'):
        """Save the experience to a file."""
        np.savez(
            path,
            obs=self.obs[: self.ptr].cpu().numpy(),
            actions=self.actions[: self.ptr].cpu().numpy(),
        )
        logging.info(f"Experience saved with {self.ptr} entries.")
        self.ptr = 0

    @classmethod
    def load(cls, path='./', device="cpu"):
        """Load the experience from a file."""
        # DONT FORGET TO SHUFFLE before BC!
        data = np.load(path)
        obs = torch.tensor(data['obs'], device=device)
        actions = torch.tensor(data['actions'], device=device)
        instance = cls(
            data_size=obs.size(0),
            obs_shape=obs.shape[1:],
            obs_dtype=obs.dtype,
            atn_shape=actions.shape[1:],
        )
        instance.obs = obs
        instance.actions = actions
        instance.ptr = len(obs)
        logging.info(f"Experience loaded with {instance.ptr} entries.")
        return instance


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


def generate_state_action_pairs(
    env,
    data_loader,
    save_path,
    data_size,
    device,
    action_space_type="multi_discrete",
    use_action_indices=True,
    make_video=False
):
    """Generate pairs of states and actions from the Waymo Open Dataset."""
    logging.info("Generating state-action pairs...")
    logging.info(f"Action space type: {action_space_type}")
    logging.info(f"Using action indices: {use_action_indices}")
    logging.info(f"Data size: {data_size}")

    experience = Experience(
        data_size=data_size,
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        atn_shape=env.action_space.shape,
    )

    for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        try:
            env.swap_data_batch(batch)
            logging.info(f"Swapped in batch {i} successfully.")
        except ValueError as e:
            logging.warning(f"Skipping a batch due to ValueError in swap_data_batch: {e}. Done")
            break

        # Reset
        obs = env.reset()

        # Get expert actions for full trajectory in all worlds
        logging.info("Getting expert actions for the full trajectory Started.\n")
        expert_actions, _, _, _ = env.get_expert_actions()
        assert action_space_type == "multi_discrete" and env.config.dynamics_model == "delta_local"
        # Multi-discrete (separate dx, dy, dyaw bins) for delta_local dynamics
        # Get per-dimension discrete indices (we keep indices, not joint encoding)
        _, idx_dx = map_to_closest_discrete_value(grid=env.dx,   cont_actions=expert_actions[:, :, :, 0])
        _, idx_dy = map_to_closest_discrete_value(grid=env.dy,   cont_actions=expert_actions[:, :, :, 1])
        _, idx_dyaw = map_to_closest_discrete_value(grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2])

        # Shape: (worlds, agents, time, 3)
        expert_actions = torch.stack([idx_dx, idx_dy, idx_dyaw], dim=-1).to(torch.int64)

        logging.info("Getting expert actions for the full trajectory Ended.\n")

        # Initialize dead agent mask

        dead_agent_mask = ~env.cont_agent_mask.clone()
        if make_video:
            VIDEO_SAVE_PATH = Path("visualization/visualization_nuplan_multi_discrete")
            os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
            done_envs = []
            frames = {f"env_{i}": [] for i in range(env.num_worlds)}
        for time_step in tqdm(range(env.episode_len)):

            experience.store(
                obs=obs,
                action=expert_actions[:, :, time_step],
                mask=~dead_agent_mask,
            )

            # Step the environment with inferred expert actions
            env.step_dynamics(expert_actions[:, :, time_step])

            next_obs = env.get_obs()

            dones = env.get_dones()

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
                logging.info(f"All agents are done for this batch at the {time_step}th step. Moving to the next batch.\n")
                break

        if make_video:
            for i in range(env.num_worlds):
                filename = env.get_env_filenames()[i][:-5]
                mediapy.write_video(path=VIDEO_SAVE_PATH / f"{filename}.mp4", images=frames[f'env_{i}'], fps=10)
            break

    # Save the experience to a file
    experience.save(path=save_path)


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return config


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


if __name__ == "__main__":

    parse = argparse.ArgumentParser(
        description="Generate expert actions and observations from Waymo Open Dataset."
    )
    parse.add_argument("--config", "-c", default="baselines/bc/config/bc_data_generation.yaml", type=str, help="Path to the configuration file.")
    parse.add_argument("--render", "-r", action="store_true", help="Whether to render the environment.")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    seed_everything(
        seed=config.seed,
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

    generate_state_action_pairs(
        env=env,
        data_loader=data_loader,
        save_path=config.experience.experience_path,  # Path to save the experience
        data_size=config.experience.dataset_size,
        device=config.train.device,
        action_space_type=config.train.action_type,  # Discretize the expert actions
        use_action_indices=True,  # Map action values to joint action index
        make_video=args.render
    )
    env.close()
    del env
    del env_config
