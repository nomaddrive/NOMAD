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
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.roadgraph import (
    LocalRoadGraphPoints,
    GlobalRoadGraphPoints,
)
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
    LidarObs,
    BevObs,
)
from gpudrive.datatypes.metadata import Metadata


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_goal_dataset(
    env,
    data_loader,
    save_path,
    dataset_name,
    device,
    action_space_type="multi_discrete",
    use_action_indices=True,
    make_video=False
):
    """Generate pairs of states and actions from the Waymo Open Dataset."""
    logging.info("Generating state-action pairs...")
    logging.info(f"Action space type: {action_space_type}")

    agent_state_list = []
    controllable_agent_mask_list = []
    goal_position_list = []
    roadgraph_list = []

    for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        try:
            env.swap_data_batch(batch)
            logging.info(f"Swapped in batch {i} successfully.")
        except ValueError as e:
            logging.warning(f"Skipping a batch due to ValueError in swap_data_batch: {e}. Done")
            break

        # Reset
        obs = env.reset()


        """
        (num_worlds, max_agents, 14)
        self.pos_x = abs_self_obs_tensor[:, :, 0]
        self.pos_y = abs_self_obs_tensor[:, :, 1]
        self.pos_z = abs_self_obs_tensor[:, :, 2]
        self.rotation_as_quaternion = abs_self_obs_tensor[:, :, 3:7]
        self.rotation_angle = abs_self_obs_tensor[:, :, 7]
        self.goal_x = abs_self_obs_tensor[:, :, 8]
        self.goal_y = abs_self_obs_tensor[:, :, 9]
        self.vehicle_length = abs_self_obs_tensor[:, :, 10] * AGENT_SCALE
        self.vehicle_width = abs_self_obs_tensor[:, :, 11] * AGENT_SCALE
        self.vehicle_height = abs_self_obs_tensor[:, :, 12]
        self.id = abs_self_obs_tensor[:, :, 13]
        """
        global_agent_states = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend=env.backend,
            device=device,
        )
        agent_states = torch.cat(   # (num_worlds, max_agents, 3)
            [
                global_agent_states.pos_x.unsqueeze(-1),
                global_agent_states.pos_y.unsqueeze(-1),
                global_agent_states.rotation_angle.unsqueeze(-1),
            ],
            dim=-1,
        )
        agent_state_list.append(agent_states.cpu())

        controllable_agent_mask = env.sim.controllable_agent_mask()  # (num_worlds, max_agents)
        controllable_agent_mask_list.append(controllable_agent_mask.cpu())

        goal_positions = torch.cat(   # (num_worlds, max_agents, 2)
            [
                global_agent_states.goal_x.unsqueeze(-1),
                global_agent_states.goal_y.unsqueeze(-1),
            ],
            dim=-1,
        )
        goal_position_list.append(goal_positions.cpu())

        """
        (num_worlds, num_road_points, 9)
        self.x = roadgraph_tensor[:, :, 0]
        self.y = roadgraph_tensor[:, :, 1]
        self.xy = torch.stack((self.x, self.y), dim=-1)
        self.segment_length = roadgraph_tensor[:, :, 2]
        self.segment_width = roadgraph_tensor[:, :, 3]
        self.segment_height = roadgraph_tensor[:, :, 4]
        self.orientation = roadgraph_tensor[:, :, 5]
        self.type = roadgraph_tensor[:, :, 6] # Original GPUDrive road types, used for plotting
        self.id = roadgraph_tensor[:, :, 7]
        self.vbd_type = roadgraph_tensor[:, :, 8] # VBD map types aligned with Waymax
        self.num_points = roadgraph_tensor.shape[1]
        """
        global_roadgraph = GlobalRoadGraphPoints.from_tensor(
            roadgraph_tensor=env.sim.map_observation_tensor(),
            backend=env.backend,
            device=device,
        )
        global_roadgraph.one_hot_encode_road_point_types()
        global_roadgraph.normalize()
        roadgraph = torch.cat(   # (num_worlds, num_road_points, 27)
            [
                global_roadgraph.x.unsqueeze(-1),
                global_roadgraph.y.unsqueeze(-1),
                global_roadgraph.segment_length.unsqueeze(-1),
                global_roadgraph.segment_width.unsqueeze(-1),
                global_roadgraph.segment_height.unsqueeze(-1),
                global_roadgraph.orientation.unsqueeze(-1),
                global_roadgraph.type,
            ],
            dim=-1,
        ).flatten(start_dim=2)
        roadgraph_list.append(roadgraph.cpu())

    agent_states_all = torch.cat(agent_state_list, dim=0)
    controllable_agent_mask_all = torch.cat(controllable_agent_mask_list, dim=0)
    goal_positions_all = torch.cat(goal_position_list, dim=0)
    roadgraph_all = torch.cat(roadgraph_list, dim=0)

    logging.info(f"Number of collected worlds: {agent_states_all.shape[0]}")
    logging.info(f"Saving dataset to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "agent_states": agent_states_all,       # (num_worlds, max_agents, 3)
        "controllable_agent_mask": controllable_agent_mask_all,  # (num_worlds, max_agents)
        "goal_positions": goal_positions_all,   # (num_worlds, max_agents, 2)
        "roadgraph": roadgraph_all,           # (num_worlds, num_road_points, 27)
    }, os.path.join(save_path, dataset_name))


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
    parse.add_argument("--config", "-c", default="baselines/traffic_gen/config/collect_dataset.yaml", type=str, help="Path to the configuration file.")
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

    generate_goal_dataset(
        env=env,
        data_loader=data_loader,
        save_path=config.goal_dataset.save_path,  # Path to save the experience
        dataset_name=config.goal_dataset.dataset_name,
        device=config.train.device,
        action_space_type=config.train.action_type,  # Discretize the expert actions
        use_action_indices=True,  # Map action values to joint action index
        make_video=args.render
    )
    env.close()
    del env
    del env_config