"""Generate complete traffic scene datasets using modular components.

This script orchestrates the generation process by:
1. Using initial_generation.py to place vehicles.
2. Using goal_generation.py to generate goals for those vehicles.
3. Combining the results into a final dataset.
"""
import torch
import numpy as np
import logging
import argparse
import yaml
import random
import os
import json
import copy
import dataclasses
from pathlib import Path
from tqdm import tqdm
from box import Box

from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints
from gpudrive.datatypes.observation import LocalEgoState, GlobalEgoState

import goal_generation
import initial_generation
import madrona_gpudrive
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Mock Classes for Goal Generation Interface ---

class MockItem:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value

class MockTensor:
    def __init__(self, data_list):
        self.data = data_list
        
    def __getitem__(self, key):
        # key is (env_idx, agent_idx)
        if isinstance(key, tuple):
            _, agent_idx = key
            if isinstance(agent_idx, torch.Tensor):
                agent_idx = agent_idx.item()
            return MockItem(self.data[int(agent_idx)])
        raise IndexError(f"MockTensor expects tuple index, got {key}")

class MockAgentStates:
    def __init__(self, vehicles):
        self.pos_x = MockTensor([v["position_x"] for v in vehicles])
        self.pos_y = MockTensor([v["position_y"] for v in vehicles])
        self.rotation_angle = MockTensor([v["heading"] for v in vehicles])
        self.id = MockTensor(list(range(len(vehicles))))

class MockLocalStates:
    def __init__(self, vehicles):
        self.speed = MockTensor([v["speed"] for v in vehicles])

# --- Helper Functions ---

def extract_scene_state(env: GPUDriveTorchEnv, device: str = "cpu"):
    """Extract scene state from simulator."""
    # We only need road graph and world means for the modular generation
    # The agent states from the sim are ignored as we generate new ones
    road_graph = GlobalRoadGraphPoints.from_tensor(
        env.sim.map_observation_tensor(), backend="torch", device=device
    )
    world_means = env.sim.world_means_tensor().to_torch().to(device)
    
    # We also need max speed from the original scenario to inform generation
    local_states = LocalEgoState.from_tensor(
        env.sim.self_observation_tensor(), backend="torch", device=device
    )
    return road_graph, world_means, local_states

def update_scenario_with_vehicles(
    original_json: dict,
    vehicles: list[initial_generation.VehicleState],
    world_means: torch.Tensor,
    env_idx: int,
) -> dict:
    """Create new scenario JSON with generated vehicles."""
    if not vehicles:
        return None
    
    updated = copy.deepcopy(original_json)
    
    # Get template
    template_vehicle = None
    for obj in original_json.get("objects", []):
        if obj.get("type") == "vehicle":
            template_vehicle = obj
            break
    
    if template_vehicle is None:
        template_vehicle = {"length": 4.5, "width": 2.0, "height": 1.5}
    
    new_objects = []
    for idx, vehicle in enumerate(vehicles):
        vehicle_obj = initial_generation.create_vehicle_json_object(
            vehicle=vehicle,
            vehicle_id=idx,
            template_obj=template_vehicle,
            world_means=world_means,
            env_idx=env_idx,
        )
        if idx == 0:
            vehicle_obj["is_sdc"] = True
        new_objects.append(vehicle_obj)
    
    updated["objects"] = new_objects
    if "metadata" in updated:
        updated["metadata"]["sdc_track_index"] = 0
    
    return updated

def visualize_generated_scene(
    vehicles: list[initial_generation.VehicleState],
    road_graph,
    env_idx: int,
    world_means: torch.Tensor,
    output_path: Path,
    scene_name: str,
):
    """Create visualization of generated scene."""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot road network
    lane_type = int(madrona_gpudrive.EntityType.RoadLane)
    lane_mask = (road_graph.type[env_idx] == lane_type).cpu()
    
    if lane_mask.any():
        lane_x = road_graph.x[env_idx, lane_mask].cpu().numpy()
        lane_y = road_graph.y[env_idx, lane_mask].cpu().numpy()
        ax.scatter(lane_x, lane_y, c='lightgray', s=1, alpha=0.3, label='Road network')
    
    # Plot vehicles and goals
    for i, v in enumerate(vehicles):
        # Vehicle positions are already in world coordinates (no denormalization needed)
        pos_x = v.position_x
        pos_y = v.position_y
        goal_x = v.goal_x
        goal_y = v.goal_y
        
        is_static = v.is_static
        
        # Vehicle color
        if is_static:
            color = 'red'
            marker = 's' # Square for static
            label = 'Static Vehicle' if i == 0 else None # Simple label logic
        else:
            color = f'C{i % 10}'
            marker = 'o'
            label = f'Vehicle {i}' if i < 10 else None
        
        # Vehicle position
        ax.scatter([pos_x], [pos_y], s=50 if is_static else 30, color=color, marker=marker, 
                  edgecolors='black', linewidths=0.5, zorder=5 if is_static else 4,
                  label=label)
        
        # Goal position (star)
        # For static vehicles, goal is same as start, so maybe skip or plot differently
        if not is_static:
            ax.plot(goal_x, goal_y, '*', color=color, markersize=15, zorder=4)
            # Connection line
            ax.plot([pos_x, goal_x], [pos_y, goal_y], '--', color=color, alpha=0.3, linewidth=1, zorder=3)
        
        # Vehicle orientation arrow (matching heuristic_all.py style)
        arrow_len = 3.0
        dx = arrow_len * np.cos(v.heading)
        dy = arrow_len * np.sin(v.heading)
        ax.arrow(pos_x, pos_y, dx, dy, head_width=1.0, head_length=1.5, 
                fc=color, ec=color, alpha=0.7, zorder=4, length_includes_head=True)
                
    # Create custom legend for static vehicles if any exist
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='Static Vehicle',
                          markerfacecolor='red', markersize=10, markeredgecolor='black')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Generated Scene: {scene_name}\n{len(vehicles)} vehicles', fontsize=14)
    if len(vehicles) <= 10:
        ax.legend(loc='upper right', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved visualization to: {output_path}")

def generate_scene_dataset_modular(
    env,
    data_loader,
    config,
    initial_params: initial_generation.InitialGenerationParams,
    goal_params: goal_generation.GoalGenerationParams,
    visualize: bool = False,
):
    """Main dataset generation function."""
    
    # Create visualization directory if needed
    if visualize:
        viz_dir = Path("visualization") / f"{config.scene_dataset.file_suffix}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Visualization directory: {viz_dir}")
    
    for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        try:
            env.swap_data_batch(batch)
        except ValueError as e:
            logging.warning(f"Skipping batch {i}: {e}")
            break
        
        road_graph, world_means, local_states = extract_scene_state(
            env, config.train.device
        )
        
        num_worlds = road_graph.x.shape[0]
        
        for env_idx in range(num_worlds):
            scene_path = Path(config.data_loader.root) / (env.get_env_filenames()[env_idx] + 'n')
            
            # Determine max speed
            max_speed_scenario = float(local_states.speed[env_idx, :].max().item())
            if max_speed_scenario <= 0:
                max_speed_scenario = 15.0
            
            logging.info(f"Processing: {scene_path.stem}")
            
            # 1. Build Lane Resources (using goal_generation module)
            lane_resources = goal_generation.build_lane_resources(
                road_graph=road_graph,
                env_idx=env_idx,
                params=goal_params, # build_lane_resources uses goal params (eps, snap_tol)
            )
            
            if lane_resources is None:
                logging.warning(f"No lane resources for {scene_path.stem}")
                continue
            
            # 2. Generate Scene Variations (Vehicles + Goals)
            with open(scene_path, "r", encoding="utf-8") as f:
                original_json = json.load(f)
                
            for set_idx in range(goal_params.num_sets):
                # We use the loop count from params
                # Actually, let's just use a loop range based on config
                
                # Re-seed for this variation
                current_seed = initial_params.rng_seed + set_idx
                np.random.seed(current_seed)
                random.seed(current_seed)
                
                # Re-generate vehicles
                vehicles_dict_var = initial_generation.initialize_vehicles_on_network(
                    lane_resources=lane_resources,
                    params=initial_params,
                    max_speed=max_speed_scenario,
                )
                
                if not vehicles_dict_var:
                    continue
                
                # Re-generate goals (just 1 set needed per scene variation)
                # We set num_sets=1 in params temporarily for this call?
                # Or we just take the first set returned.
                
                mock_agent_states_var = MockAgentStates(vehicles_dict_var)
                mock_local_states_var = MockLocalStates(vehicles_dict_var)
                mock_mask_var = torch.zeros((num_worlds, len(vehicles_dict_var)), dtype=torch.bool)
                mock_mask_var[env_idx, :] = True
                
                np_rng_var = np.random.default_rng(current_seed)
                py_rng_var = random.Random(current_seed)
                
                # We only need 1 goal per agent for this scene variation
                # But generate_goal_sets might generate multiple. We'll take the first one.
                # Optimization: Set num_sets to 1 for this call to avoid wasted computation
                original_num_sets = goal_params.num_sets
                goal_params.num_sets = 1
                
                try:
                    per_agent_goals_var, _ = goal_generation.generate_goal_sets(
                        agent_states=mock_agent_states_var,
                        local_states=mock_local_states_var,
                        cont_agent_mask=mock_mask_var,
                        env_idx=env_idx,
                        lane_resources=lane_resources,
                        params=goal_params,
                        np_rng=np_rng_var,
                        py_rng=py_rng_var,
                    )
                except ValueError as e:
                    logging.warning(f"Skipping scene variation {set_idx} due to ValueError: {e}")
                    goal_params.num_sets = original_num_sets
                    continue
                
                # Restore params
                goal_params.num_sets = original_num_sets
                
                # Combine into VehicleState objects
                final_vehicles = []
                for i, v_dict in enumerate(vehicles_dict_var):
                    # Get goal for this agent (ID = i)
                    # per_agent_goals_var[i] is a list of goals. We take the first one.
                    is_static = v_dict.get("is_static", False)
                    
                    if i not in per_agent_goals_var or not per_agent_goals_var[i]:
                        # For intentionally static vehicles, use position as goal
                        # For dynamic vehicles, skip them (goal generation failed)
                        if is_static:
                            goal_x = v_dict["position_x"]
                            goal_y = v_dict["position_y"]
                        else:
                            continue  # Skip dynamic vehicles without goals
                    else:
                        goal = per_agent_goals_var[i][0] # Take 0-th set
                        goal_x = goal[0]
                        goal_y = goal[1]
                    
                    veh_state = initial_generation.VehicleState(
                        position_x=v_dict["position_x"],
                        position_y=v_dict["position_y"],
                        position_z=0.0,
                        heading=v_dict["heading"],
                        speed=v_dict["speed"],
                        lane_id=v_dict["lane_id"],
                        s=v_dict["s"],
                        goal_x=goal_x,
                        goal_y=goal_y,
                        goal_z=0.0,
                        is_static=is_static,
                    )
                    final_vehicles.append(veh_state)
                
                if not final_vehicles:
                    continue
                
                # Visualize this variation if enabled
                if visualize:
                    viz_path = viz_dir / f"{scene_path.stem}_heuristic_scenes_{set_idx:02d}_viz.png"
                    visualize_generated_scene(
                        vehicles=final_vehicles,
                        road_graph=road_graph,
                        env_idx=env_idx,
                        world_means=world_means,
                        output_path=viz_path,
                        scene_name=f"{scene_path.stem} (variation {set_idx})",
                    )
                    
                # Update JSON
                updated_json = update_scenario_with_vehicles(
                    original_json=original_json,
                    vehicles=final_vehicles,
                    world_means=world_means,
                    env_idx=env_idx,
                )
                
                if updated_json is None:
                    continue
                
                # Save
                suffix = config.scene_dataset.file_suffix
                # Shortened naming convention: stem + 's' + index (s for scene)
                new_name = f"{scene_path.stem}s{set_idx:02d}.json"
                original_scenario_id = updated_json["scenario_id"]
                new_scenario_id = f"{original_scenario_id}s{set_idx:02d}"
                
                updated_json["name"] = new_name
                updated_json["scenario_id"] = new_scenario_id
                
                output_dir = Path(config.scene_dataset.save_path) / (
                    Path(config.data_loader.root).name + f"_{suffix}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / new_name
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(updated_json, f, indent=2)
                
                logging.info(f"Saved: {output_path.name}")

def load_config(config_path):
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return config

def create_params_from_config(param_cls, config_section):
    """Helper to create dataclass instance from config section, filtering extraneous keys."""
    valid_fields = {f.name for f in dataclasses.fields(param_cls)}
    # Filter config to only include valid fields
    filtered_args = {k: v for k, v in config_section.items() if k in valid_fields}
    return param_cls(**filtered_args)

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="baselines/goal/config/heuristic_scene_generation.yaml", type=str)
    parser.add_argument("--visualize", action="store_true", help="Generate visualization images for each scene")
    parser.add_argument("--city", type=str, help="City name to use for density parameters (boston, pittsburgh, singapore, vegas)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Determine city and load specific parameters
    city = args.city or config.scene_generation.get("default_city", "boston")
    logging.info(f"Using density parameters for city: {city}")
    
    city_params = config.scene_generation.city_params.get(city)
    if city_params is None:
        logging.warning(f"City '{city}' not found in config. Using defaults.")
        city_params = {}
        
    # Merge city params into scene_generation config
    # We create a copy to avoid modifying the original config structure if needed elsewhere
    scene_gen_config = config.scene_generation.copy()
    scene_gen_config.update(city_params)

    # Extract params for modules
    # We map the config structure to the dataclasses dynamically
    initial_params = create_params_from_config(
        initial_generation.InitialGenerationParams, 
        scene_gen_config
    )
    
    goal_params = create_params_from_config(
        goal_generation.GoalGenerationParams, 
        scene_gen_config
    )
    
    seed_everything(config.seed, config.train.torch_deterministic)
    
    data_loader = SceneDataLoader(**config.data_loader)
    env_config = EnvConfig(**config.environment)
    
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=config.environment.max_controlled_agents,
        device=config.train.device,
        action_type=config.train.action_type,
    )
    
    generate_scene_dataset_modular(
        env=env,
        data_loader=data_loader,
        config=config,
        initial_params=initial_params,
        goal_params=goal_params,
        visualize=args.visualize,
    )
    
    env.close()
