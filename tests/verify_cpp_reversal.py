import torch
import numpy as np
import madrona_gpudrive
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints
import argparse

def get_first_road_edge_vector(env):
    """Extracts the vector (dy) of the first valid RoadEdge found."""
    global_roadgraph = GlobalRoadGraphPoints.from_tensor(
        roadgraph_tensor=env.sim.map_observation_tensor(),
        backend=env.backend,
        device=env.device,
    )
    
    # Mask for RoadEdges
    road_mask = global_roadgraph.type == int(madrona_gpudrive.EntityType.RoadEdge)
    
    # Get orientation
    orientations = global_roadgraph.orientation[road_mask].cpu().numpy()
    
    if len(orientations) == 0:
        return None
        
    # We just need the sin(theta) aka dy component of the vector to check reversal
    # Orientation is angle of the segment.
    # If points are P1->P2, theta is atan2(y2-y1, x2-x1).
    # If points are reversed P2->P1, theta' = atan2(y1-y2, x1-x2) = theta + pi.
    # sin(theta + pi) = -sin(theta).
    # So Y component should flipped.
    
    return np.sin(orientations[0]) # Just take the first one 

from gpudrive.env.config import EnvConfig, SceneConfig
from gpudrive.env.dataset import SceneDataLoader

def run_test():
    # Setup - use a known map path or one from config
    data_dir = "data/nuplan/saeed/singapore_valid_800" 
    
    import os
    if not os.path.exists(data_dir):
         print(f"Data dir {data_dir} not found. Trying Boston...")
         data_dir = "data/nuplan/saeed/boston_valid_800"
         if not os.path.exists(data_dir):
             print(f"Data dir {data_dir} not found. Failing.")
             return
    
    print(f"Using data from: {data_dir}")

    # Helper to load env
    def load_env(mirror_x):
        # Configs
        env_config = EnvConfig(
            mirror_x=mirror_x,
            num_worlds=1,
            # valid defaults to minimize overhead
            ego_state=True, road_map_obs=True, partner_obs=False, 
            bev_obs=False, lidar_obs=False,
            max_controlled_agents=64,
        )

        # Make dataloader
        data_loader = SceneDataLoader(
            root=data_dir, # Path to the dataset
            batch_size=1, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
            dataset_size=1, # Total number of different scenes we want to use
            sample_with_replacement=False, 
            seed=42, 
            shuffle=False,
            file_prefix="",
        )

        # Make environment
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=data_loader,
            device="cuda", 
            action_type="discrete", # "continuous" or "discrete"
            max_cont_agents=64,
        )

        return env

    # CONFIG 1: Normal
    # print("\nInitializing Normal Env (mirror_x=False)...")
    # try:
    #     env_normal = load_env(mirror_x=False)
    #     vec_y_normal = get_first_road_edge_vector(env_normal)
    #     env_normal.close()
    #     print(f"Normal Vector Y-component (sin theta): {vec_y_normal}")
    # except Exception as e:
    #     print(f"Failed to load normal env: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return

    vec_y_normal = 0.929791271686554

    # CONFIG 2: Mirrored
    print("\nInitializing Mirrored Env (mirror_x=True)...")
    try:
        env_mirror = load_env(mirror_x=True)
        vec_y_mirror = get_first_road_edge_vector(env_mirror)
        env_mirror.close()
        print(f"Mirrored Vector Y-component (sin theta): {vec_y_mirror}")
    except Exception as e:
        print(f"Failed to load mirrored env: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n--- RESULTS ---")
    if vec_y_mirror is None:
        print("Could not find road edges to compare.")
        return

    # CHECK
    # If points are NOT reversed (Buggy): Y coordinates are same, Order is same. 
    #   P1=(x1, y1) -> P2=(x2, y2). dy = y2-y1.
    #   Mirror: P1'=(-x1, y1) -> P2'=(-x2, y2). dy' = y2-y1.
    #   Result: vec_y_mirror SHOULD BE EQUAL to vec_y_normal.
    #
    # If points ARE reversed (Fixed):
    #   Mirror: P2'=(-x2, y2) -> P1'=(-x1, y1). dy' = y1-y2 = -(y2-y1).
    #   Result: vec_y_mirror SHOULD BE NEGATIVE of vec_y_normal.
    
    print(f"Normal: {vec_y_normal:.4f}")
    print(f"Mirror: {vec_y_mirror:.4f}")
    
    # Allow for small float diffs, checking sign mainly or approximate value
    if np.isclose(vec_y_mirror, vec_y_normal, atol=1e-3):
        print("\n[FAIL] The Y-components are IDENTICAL. The geometry is NOT reversed.")
        print("The C++ fix is NOT active.")
    elif np.isclose(vec_y_mirror, -vec_y_normal, atol=1e-3):
        print("\n[PASS] The Y-components are OPPOSITE. The geometry IS reversed.")
        print("The C++ fix IS active!")
    else:
        print("\n[UNCLEAR] The values are neither identical nor opposite. Something else changed.")

if __name__ == "__main__":
    run_test()
