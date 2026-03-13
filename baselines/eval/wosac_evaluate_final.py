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

import madrona_gpudrive
from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.networks.actor_critic import NeuralNet
from gpudrive.visualize.utils import img_from_fig
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints, MapElementIds

from benchmark.evaluator import WOSACEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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


def get_agent_size(env) -> dict:
    # Get agent width and length
    agent_abs_observation_tensor = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(),
        backend=env.backend,
        device=env.device,
    )
    agent_state = {
        "length": agent_abs_observation_tensor.vehicle_length.cpu().numpy().reshape(-1),
        "width": agent_abs_observation_tensor.vehicle_width.cpu().numpy().reshape(-1),
    }
    return agent_state


def get_road_edge_polylines(env) -> dict:
    # Get road edge polylines
    global_roadgraph = GlobalRoadGraphPoints.from_tensor(
        roadgraph_tensor=env.sim.map_observation_tensor(),
        backend=env.backend,
        device=env.device,
    )
    # Get mean values for restoring absolute coordinates
    means_xy = env.sim.world_means_tensor().to_torch().to(env.device)[:, :2]
    mean_x, mean_y = means_xy[:, 0], means_xy[:, 1]
    global_roadgraph.restore_mean(mean_x=mean_x, mean_y=mean_y)

    # Create mask for road edges
    road_mask = global_roadgraph.type == int(madrona_gpudrive.EntityType.RoadEdge)
    
    # Get data on CPU numpy
    # global_roadgraph.segment_length is half-length.
    centers_x = global_roadgraph.x[road_mask].cpu().numpy()
    centers_y = global_roadgraph.y[road_mask].cpu().numpy()
    half_lengths = global_roadgraph.segment_length[road_mask].cpu().numpy()
    orientations = global_roadgraph.orientation[road_mask].cpu().numpy()
    road_ids = global_roadgraph.id[road_mask].cpu().numpy()
    
    # Scenario IDs
    raw_scenario_ids = np.array(list(env.get_scenario_ids().values()))
    scenario_ids_expanded = np.repeat(raw_scenario_ids[:, None], global_roadgraph.num_points, axis=-1)
    scenario_ids = scenario_ids_expanded[road_mask.cpu().numpy()]

    # Calculate endpoints
    cos_ori = np.cos(orientations)
    sin_ori = np.sin(orientations)
    
    start_x = centers_x - half_lengths * cos_ori
    start_y = centers_y - half_lengths * sin_ori
    end_x = centers_x + half_lengths * cos_ori
    end_y = centers_y + half_lengths * sin_ori

    if len(start_x) == 0:
         return {
            "x": np.array([], dtype=np.float32),
            "y": np.array([], dtype=np.float32),
            "lengths": np.array([], dtype=np.int32),
            "scenario_id": np.array([], dtype=np.int32),
        }

    # Group by (scenario_id, road_id)
    # Detect changes in either scenario_id or road_id
    scen_diff = scenario_ids[1:] != scenario_ids[:-1]
    id_diff = road_ids[1:] != road_ids[:-1]
    change_mask = scen_diff | id_diff
    
    # Indices where a grouping ENDS
    split_indices = np.nonzero(change_mask)[0] + 1
    poly_starts = np.concatenate(([0], split_indices))
    poly_ends = np.concatenate((split_indices, [len(start_x)]))
    
    out_x_list = []
    out_y_list = []
    out_lengths = []
    out_scenario_ids = []

    for s, e in zip(poly_starts, poly_ends):
        # We take all START points of segments in the group
        # And the END point of the LAST segment in the group
        # This assumes contiguous segments (End[i] == Start[i+1])
        
        p_xs = np.concatenate([start_x[s:e], [end_x[e-1]]])
        p_ys = np.concatenate([start_y[s:e], [end_y[e-1]]])
        
        out_x_list.append(p_xs)
        out_y_list.append(p_ys)
        out_lengths.append(len(p_xs))
        out_scenario_ids.append(scenario_ids[s])

    return {
        "x": np.concatenate(out_x_list).astype(np.float32),
        "y": np.concatenate(out_y_list).astype(np.float32),
        "lengths": np.array(out_lengths, dtype=np.int32),
        "scenario_id": np.array(out_scenario_ids),
    }


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
    expert_policy_mode="none",
):
    """Evaluate a learned policy or an oracle expert.

    expert_policy_mode:
        None -> use provided policy.
        "multi_discrete" -> use discretized expert (indices into dx, dy, dyaw bins).
        "continuous" -> use raw continuous expert deltas (dx, dy, dyaw) directly.
    """
    logging.info("Evaluating the policy...")

    if policy:
        policy.eval()

    evaluator = WOSACEvaluator(config=config)

    # gt_trajectories_list = []
    # simulated_trajectories_list = []

    # for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        # try:
        #     env.swap_data_batch(batch)
        #     logging.info(f"Swapped in batch {i} successfully.")
        # except ValueError as e:
        #     logging.warning(f"Stop swapping due to ValueError in swap_data_batch: {e}. Done")
        #     break

    # Collect ground truth trajectories from the dataset
    gt_trajectories = evaluator.collect_ground_truth_trajectories(env)

    # Roll out trained policy in the simulator
    simulated_trajectories = evaluator.collect_simulated_trajectories(env, policy, expert_policy_mode)

    print(f"\nCollected trajectories on {len(np.unique(gt_trajectories['scenario_id']))} scenarios.")

    if config["eval"]["wosac_sanity_check"]:
        evaluator._quick_sanity_check(gt_trajectories, simulated_trajectories)

    agent_state = get_agent_size(env)
    road_edge_polylines = get_road_edge_polylines(env)

    # Analyze and compute metrics
    results = evaluator.compute_metrics(
        gt_trajectories,
        simulated_trajectories,
        agent_state,
        road_edge_polylines,
        config["eval"]["wosac_aggregate_results"],
    )

    if config["eval"]["wosac_aggregate_results"]:

        print("WOSAC_METRICS_START")
        print(json.dumps(results))
        print("WOSAC_METRICS_END")

    return results

# eval_run_dir = [
#     "runs/sprl_singapore_to_pittsburgh_seed0__C__R_200000__12_29_08_25_39_617",
#     "runs/sprl_singapore_to_pittsburgh_seed1__C__R_200000__12_29_08_26_32_947",
#     "runs/sprl_singapore_to_pittsburgh_seed2__C__R_200000__12_29_08_27_06_771",
#     "runs/sprl_singapore_to_pittsburgh_seed4__C__R_200000__12_29_08_27_34_273",
#     "runs/sprl_singapore_to_pittsburgh_seed42__C__R_200000__12_30_11_18_51_987",
# ]

# eval_run_dir = [
#     "runs/kl0.08_mirrorx_false_generate_scene__C__R_200000__12_10_22_16_34_075",
#     "runs/kl0.08_mirrorx_false_generate_scene_boston_to_singapore_seed0__C__R_200000__12_15_16_12_21_636",
#     "runs/kl0.08_mirrorx_false_generate_scene_boston_to_singapore_seed1__C__R_200000__12_15_16_13_50_263",
#     "runs/kl0.08_mirrorx_false_generate_scene_boston_to_singapore_seed2__C__R_200000__12_15_16_14_19_428",
#     "runs/kl0.08_mirrorx_false_generate_scene_boston_to_singapore_seed4__C__R_200000__12_16_02_17_53_187",
# ]

eval_run_dir = [
    "runs/sprl_singapore_to_boston_seed0__C__R_200000__12_26_07_58_08_544",
    "runs/sprl_singapore_to_boston_seed1__C__R_200000__12_26_07_59_13_665",
    "runs/sprl_singapore_to_boston_seed2__C__R_200000__12_26_07_59_37_493",
    "runs/sprl_singapore_to_boston_seed4__C__R_200000__12_26_08_01_53_874",
    "runs/sprl_singapore_to_boston_seed42__C__R_200000__12_26_07_57_31_026",
]

# frontier checkpoints for Pittsburgh_valid_800
# eval_epochs = [
#     6040, 6340, 6080, 6550, 6030, 7470, 7010, 7370, 7070, 7020, 7520, 7050
# ]

# frontier checkpoints for singapore_valid_800_newtest
# eval_epochs = [
#     3710, 3660, 3640, 3670, 4310, 4090, 3750, 5040, 4980, 4990, 6330, 5380, 7460, 6510, 6570, 6520, 7470, 6040, 6930, 7030, 7040, 7170, 7400
# ]

# frontier checkpoints for singapore_valid_800
# eval_epochs = [
#     3710, 3740, 3690, 4300, 3930, 4610, 3980, 4010, 4700, 4670, 6160, 5000,
#     6150, 6570, 4800, 7480, 6040, 7490, 7400
# ]

# frontier checkpoints for boston_valid_800
eval_epochs = [
    4910, 4860, 4920, 4940, 6180, 6310, 6240, 6250, 6370, 6290, 6800, 6840, 6880, 7530, 7520, 7540, 7040
]


if __name__ == "__main__":


    parse = argparse.ArgumentParser(
        description="Generate expert actions and observations from Waymo Open Dataset."
    )
    parse.add_argument("--config", "-c", default="baselines/eval/config/wosac_evaluate_final.yaml", type=str, help="Path to the configuration file.")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    seed_everything(
        seed=config.train.seed,
        torch_deterministic=config.train.torch_deterministic,
    )

    results_dir = Path("eval_wosac_icml") / config.environment.collision_behavior / str(config.environment.mirror_x) / "assigned_checkpoints" / eval_run_dir[0].split('/')[-1]
    results_dir = results_dir / config.data_loader.root.split('/')[-1]  # Use last part of data root as subdir
    results_dir.mkdir(parents=True, exist_ok=True)

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
    config.train.action_type = "multi_discrete"
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=config.environment.max_controlled_agents,
        device=config.train.device,
        action_type=config.train.action_type,
    )

    # Iterate over run directories and epochs
    config.train.continue_training = True
    aggregate_results = []

    for run_dir_str in eval_run_dir:
        run_path = Path(run_dir_str)
        run_basename = run_path.name
        
        for epoch in eval_epochs:
            ckpt_filename = f"model_{run_basename}_{epoch:06d}.pt"
            ckpt_path = run_path / ckpt_filename
            
            if not ckpt_path.exists():
                print(f"Checkpoint not found: {ckpt_path}, skipping.")
                continue

            print(f"Evaluating model checkpoint: {ckpt_path}")
            config.train.model_cpt = str(ckpt_path)

            # Create policy
            policy = make_agent(env=env, config=config).to(
                config.train.device
            )
            
            config.train.network.num_parameters = get_model_parameters(policy)

            wosac_result = evaluate(
                config,
                env, 
                data_loader, 
                policy, 
            )

            # Save per-checkpoint JSON
            ckpt_name = ckpt_path.stem
            with open(results_dir / f"{ckpt_name}.wosac_stats.json", "w") as f:
                json.dump({"checkpoint": ckpt_name, "results": wosac_result}, f, indent=2)
            print(f"Saved results to {results_dir / f'{ckpt_name}.wosac_stats.json'}")

            # Add to aggregate
            row = {"checkpoint": ckpt_name}
            row.update(wosac_result)
            aggregate_results.append(row)

    # Save aggregate CSV
    if aggregate_results:
        csv_path = results_dir / "aggregate_results.csv"
        keys = list(aggregate_results[0].keys())
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(aggregate_results)
        print(f"Aggregate CSV written to {csv_path}")

        # Generate Markdown table with bold best values
        lower_is_better = {"ade", "min_ade", "num_collisions_sim", "num_collisions_ref"}
        
        # Find best values
        best_values = {}
        for key in keys:
            if key == "checkpoint":
                continue
            # Filter for numeric values only
            values = [row[key] for row in aggregate_results if isinstance(row[key], (int, float))]
            if not values:
                continue
            
            if key in lower_is_better:
                best_values[key] = min(values)
            else:
                best_values[key] = max(values)

        md_lines = []
        md_lines.append(f"| {' | '.join(keys)} |")
        md_lines.append(f"| {' | '.join(['---'] * len(keys))} |")

        for row in aggregate_results:
            line = []
            for key in keys:
                val = row[key]
                if key == "checkpoint":
                    line.append(str(val))
                    continue
                
                if isinstance(val, (int, float)):
                    is_best = False
                    if key in best_values:
                        if val == best_values[key]:
                            is_best = True
                    
                    val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                    if is_best:
                        line.append(f"**{val_str}**")
                    else:
                        line.append(val_str)
                else:
                    line.append(str(val))
            md_lines.append(f"| {' | '.join(line)} |")
        
        md_path = results_dir / "aggregate_results.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))
        print(f"Aggregate Markdown table written to {md_path}")
        print("\n".join(md_lines))

    env.close()
    del env
