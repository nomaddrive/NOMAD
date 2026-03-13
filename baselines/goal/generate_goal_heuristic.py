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
import json
import copy
import madrona_gpudrive

from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

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
from gpudrive.datatypes.control import ResponseType
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
    LidarObs,
    BevObs,
)
from gpudrive.datatypes.metadata import Metadata
from gpudrive.visualize import utils
from goal_generation import (
    GoalGenerationParams,
    LaneResources,
    build_lane_resources,
    generate_goal_sets,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_road_edges_and_lines(road_graph: GlobalRoadGraphPoints, env_idx, ax, base_alpha: float = 0.5):
    """Plot the road graph."""

    ROAD_GRAPH_COLORS = {
        1: "#000000",  # 'RoadEdgeBoundary' (Gray)
        2: np.array([120, 120, 120])
        / 255.0,  # 'RoadLine-BrokenSingleYellow' (Light Purple)
        3: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-Freeway' (Light Gray)
        4: np.array([200, 200, 200]) / 255.0,  # 'Crosswalk' (Light Gray)
        5: np.array([0.85, 0.65, 0.13]),  # 'SpeedBump' (Dark yellow)
        6: np.array([255, 0, 0]) / 255.0,  # 'StopSign' (Red)
    }
    def endpoints(x, y, length, theta):
        dx = length * np.cos(theta)
        dy = length * np.sin(theta)
        start = np.stack([x - dx, y - dy], axis=-1)
        end = np.stack([x + dx, y + dy], axis=-1)
        return start, end

    for road_point_type in road_graph.type.unique().tolist():
        if road_point_type == int(madrona_gpudrive.EntityType._None):
            continue

        road_mask = road_graph.type[env_idx, :] == road_point_type

        # Get coordinates and metadata for the current road type
        x_coords = road_graph.x[env_idx, road_mask].tolist()
        y_coords = road_graph.y[env_idx, road_mask].tolist()
        segment_lengths = road_graph.segment_length[
            env_idx, road_mask
        ].tolist()
        segment_widths = road_graph.segment_width[
            env_idx, road_mask
        ].tolist()
        segment_orientations = road_graph.orientation[
            env_idx, road_mask
        ].tolist()

        road_ids = road_graph.id[env_idx, road_mask].int().tolist()

        if road_point_type in [
            int(madrona_gpudrive.EntityType.RoadEdge),
            int(madrona_gpudrive.EntityType.RoadLine),
            int(madrona_gpudrive.EntityType.RoadLane),
        ]:
            for x, y, length, orientation, road_id in zip(
                x_coords,
                y_coords,
                segment_lengths,
                segment_orientations,
                road_ids
            ):
                start, end = endpoints(
                    x, y, length, orientation
                )
                line_width = (
                    1.1 * 1.0
                    if road_point_type
                    == int(madrona_gpudrive.EntityType.RoadEdge)
                    else 0.75 * 1.0
                )

                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=ROAD_GRAPH_COLORS[road_point_type],
                    linewidth=line_width,
                    alpha=base_alpha,
                    zorder=1,
                )
        elif road_point_type == int(madrona_gpudrive.EntityType.SpeedBump):
            utils.plot_speed_bumps(
                x_coords,
                y_coords,
                segment_lengths,
                segment_widths,
                segment_orientations,
                ax,
            )

        elif road_point_type == int(madrona_gpudrive.EntityType.StopSign):
            for x, y in zip(x_coords, y_coords):
                utils.plot_stop_sign(
                    point=np.array([x, y]),
                    ax=ax,
                    radius=1.5,
                    facecolor="#c04000",
                    edgecolor="none",
                    linewidth=3.0,
                    alpha=0.9,
                )

        elif road_point_type == int(madrona_gpudrive.EntityType.CrossWalk):
            for x, y, length, width, orientation in zip(
                x_coords,
                y_coords,
                segment_lengths,
                segment_widths,
                segment_orientations,
            ):
                def _get_corners_polygon(x, y, length, width, orientation):
                    """Calculate the four corners of a speed bump (can be any) polygon."""
                    # Compute the direction vectors based on orientation
                    # print(length)
                    c = np.cos(orientation)
                    s = np.sin(orientation)
                    u = np.array((c, s))  # Unit vector along the orientation
                    ut = np.array((-s, c))  # Unit vector perpendicular to the orientation

                    # Center point of the speed bump
                    pt = np.array([x, y])

                    # corners
                    tl = pt + (length / 2) * u - (width / 2) * ut
                    tr = pt + (length / 2) * u + (width / 2) * ut
                    br = pt - (length / 2) * u + (width / 2) * ut
                    bl = pt - (length / 2) * u - (width / 2) * ut

                    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]
                    
                points = _get_corners_polygon(
                    x, y, length, width, orientation
                )
                utils.plot_crosswalk(
                    points=points,
                    ax=ax,
                    facecolor="none",
                    edgecolor="xkcd:bluish grey",
                    alpha=0.4,
                )


def extract_scene_state(env: GPUDriveTorchEnv, device: str = "cpu"):
    agent_states = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(), backend="torch", device=device
    )
    local_states = LocalEgoState.from_tensor(
        env.sim.self_observation_tensor(), backend="torch", device=device
    )
    road_graph = GlobalRoadGraphPoints.from_tensor(
        env.sim.map_observation_tensor(), backend="torch", device=device
    )
    response_type = ResponseType.from_tensor(
        tensor=env.sim.response_type_tensor(), backend="torch", device=device
    )
    filenames = env.get_env_filenames()
    cont_mask = env.cont_agent_mask.to(device)
    world_means = env.sim.world_means_tensor().to_torch().to(device)
    return agent_states, local_states, road_graph, cont_mask, world_means, filenames, response_type


def update_goal_positions(original_json: Dict, goal_map: Dict[int, Tuple[float, float]]):
    if not goal_map:
        return None

    updated = copy.deepcopy(original_json)
    id_to_object = {obj["id"]: obj for obj in updated.get("objects", [])}

    for agent_id, (gx, gy) in goal_map.items():
        obj = id_to_object.get(agent_id)
        if obj is None:
            continue
        if obj.get("type") not in {"vehicle"}:
            continue
        goal_pos = obj.get("goalPosition", {"x": 0.0, "y": 0.0, "z": 0.0})
        goal_pos["x"] = float(gx)
        goal_pos["y"] = float(gy)
        goal_pos.setdefault("z", 0.0)
        obj["goalPosition"] = goal_pos

    return updated


def denormalize_goals(
    goal_sets: List[Dict[int, Tuple[float, float]]],
    env_idx: int,
    world_means: torch.Tensor,
):
    mean_x, mean_y, _ = world_means[env_idx]
    denorm_sets = []
    for per_set in goal_sets:
        denorm_sets.append(
            {
                agent_id: (gx + mean_x.item(), gy + mean_y.item())
                for agent_id, (gx, gy) in per_set.items()
            }
        )
    return denorm_sets


def write_goalsets(
    scene_path: Path,
    output_dir: Path,
    goal_sets: List[Dict[int, Tuple[float, float]]],
    suffix: str,
):
    output_dir = output_dir / (scene_path.parent.name + "_heuristic_goals")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_path, "r", encoding="utf-8") as f:
        original_json = json.load(f)

    written_files = []
    for idx, goal_map in enumerate(goal_sets):
        # Update original json with new goal positions, name, and scenario_id
        original_json["name"] = f"{scene_path.stem}h{idx:02d}.json"
        original_json["scenario_id"] = f"{original_json["scenario_id"]}h{idx:02d}"
        updated = update_goal_positions(original_json, goal_map)
        if updated is None:
            continue
        new_name = f"{scene_path.stem}h{idx:02d}.json"
        output_path = output_dir / new_name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
        written_files.append(output_path)
    return written_files


def plot_generated_goals(
    agent_states,
    road_graph,
    response_type,
    filenames,
    cont_mask,
    env_idx,
    per_agent_goals,
):
    fig, ax = plt.subplots(figsize=(6,6), dpi=200)
    plot_road_edges_and_lines(road_graph, env_idx, ax, base_alpha=0.35)
    # Plot log replay agents
    log_replay = (
        response_type.static[env_idx, :] | response_type.moving[env_idx, :]
    ) & ~cont_mask[env_idx, :]

    pos_x = agent_states.pos_x[env_idx, log_replay]
    pos_y = agent_states.pos_y[env_idx, log_replay]
    rotation_angle = agent_states.rotation_angle[env_idx, log_replay]
    vehicle_length = agent_states.vehicle_length[env_idx, log_replay]
    vehicle_width = agent_states.vehicle_width[env_idx, log_replay]
    ids = agent_states.id[env_idx, log_replay]

    valid_mask = (
        (torch.abs(pos_x) < 1000)
        & (torch.abs(pos_y) < 1000)
        & (
            (vehicle_length > 0.5)
            & (vehicle_length < 15)
            & (vehicle_width > 0.5)
            & (vehicle_width < 15)
        )
    )

    bboxes_static = np.stack(
        (
            pos_x[valid_mask].numpy(),
            pos_y[valid_mask].numpy(),
            vehicle_length[valid_mask].numpy(),
            vehicle_width[valid_mask].numpy(),
            rotation_angle[valid_mask].numpy(),
            ids[valid_mask].numpy()
        ),
        axis=1,
    )

    utils.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=bboxes_static,
        color="#c7c7c7",
        alpha=0.8,
        line_width_scale=0.5,
        as_center_pts=False,
        label=None,
    )
    # Build list of controllable agents (we don't filter offroad/collision as requested)
    controlled_idx = torch.where(cont_mask[env_idx])[0].cpu().numpy().tolist()

    # Assign a consistent color per agent index
    cmap = plt.get_cmap("tab20")
    def agent_color(k):
        return cmap(k % cmap.N)

    # Draw vehicles and original goals per agent using per-agent color
    for k, aidx in enumerate(controlled_idx):
        color = agent_color(k)
        bx = np.array([
            [
                agent_states.pos_x[env_idx, aidx].item(),
                agent_states.pos_y[env_idx, aidx].item(),
                agent_states.vehicle_length[env_idx, aidx].item(),
                agent_states.vehicle_width[env_idx, aidx].item(),
                agent_states.rotation_angle[env_idx, aidx].item(),
                agent_states.id[env_idx, aidx].item(),
            ]
        ])
        utils.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=bx,
            color=color,
            alpha=0.9,
            line_width_scale=0.5,
            as_center_pts=False,
            label=None,
        )
        # Original goal marker
        gx0 = agent_states.goal_x[env_idx, aidx].item()
        gy0 = agent_states.goal_y[env_idx, aidx].item()
        ax.scatter([gx0], [gy0], s=18, marker="s", facecolors="none", edgecolors=color, linewidths=0.5, zorder=4)
        gx, gy = [], []
        goals = per_agent_goals[int(agent_states.id[env_idx, aidx].item())]
        for g in goals:
            gx.append(g[0])
            gy.append(g[1])
        sc = ax.scatter(gx, gy, color=color, s=18, marker="x", zorder=5, linewidths=0.5)
        try:
            sc.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
        except Exception:
            pass
    center_x = agent_states.pos_x[
        env_idx, 0
    ].item()
    center_y = agent_states.pos_y[
        env_idx, 0
    ].item()

    # Use configured zoom radius around ego
    R = float(getattr(args, "zoom_radius", 100))
    ax.set_xlim(center_x - R, center_x + R)
    ax.set_ylim(center_y - R, center_y + R)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.05)
    save_folder = Path("visualization") / "heuristic" / "render_resample"
    save_folder.mkdir(parents=True, exist_ok=True)
    out_file = save_folder / f"env{env_idx}_file{filenames[env_idx]}.png"
    plt.savefig(out_file, dpi=400, bbox_inches='tight', facecolor="white")
    plt.close(fig)

def generate_goal_dataset(
    env,
    data_loader,
    config,
    np_rng,
    py_rng,
    render=False,
    debug=False,
):

    for i, batch in tqdm(enumerate(data_loader), desc="Loading data batches"):
        try:
            env.swap_data_batch(batch)
            logging.info(f"Swapped in batch {i} successfully.")
        except ValueError as e:
            logging.warning(f"Skipping a batch due to ValueError in swap_data_batch: {e}. Done")
            break

        # Logic
        if render:
            config.train.device = "cpu"
        agent_states, local_states, road_graph, cont_mask, world_means, filenames, response_type = extract_scene_state(env, config.train.device)

        num_worlds = agent_states.shape[0]
        for env_idx in range(num_worlds):
            scene_path = Path(config.data_loader.root) / (env.get_env_filenames()[env_idx] + 'n')
            lane_resources: LaneResources | None = build_lane_resources(
                road_graph, env_idx=env_idx, params=config.goal_params
            )
            try:
                per_agent_goals, goal_sets = generate_goal_sets(
                    agent_states,
                    local_states,
                    cont_mask,
                    env_idx=env_idx,
                    lane_resources=lane_resources,
                    params=config.goal_params,
                    np_rng=np_rng,
                    py_rng=py_rng,
                )
            except ValueError as e:
                logging.warning(f"Skipping scene {scene_path.stem} due to ValueError in goal generation: {e}.")
                continue
            if render:
                plot_generated_goals(
                    agent_states,
                    road_graph,
                    response_type,
                    filenames,
                    cont_mask,
                    env_idx,
                    per_agent_goals,
                )
                continue
            goal_sets = denormalize_goals(
                goal_sets, env_idx, world_means
            )
            written = write_goalsets(scene_path, Path(config.goal_dataset.save_path), goal_sets, config.goal_dataset.file_suffix)
            logging.info(f"Wrote {len(written)} goal set files for scene {scene_path.stem}.")

        if render or debug:
            break  # Only render the first batch

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
    parse.add_argument("--config", "-c", default="baselines/goal/config/heuristic_goal_generation.yaml", type=str, help="Path to the configuration file.")
    parse.add_argument("--render", "-r", action="store_true", help="Whether to render the environment.")
    parse.add_argument("--debug", "-d", action="store_true", help="Whether to render debug information.")
    args = parse.parse_args()
    # Load default configs
    config = load_config(args.config)

    goal_params = GoalGenerationParams(
        num_sets=config.goal_generation.num_sets,
        distance_mean_coeff=config.goal_generation.distance_mean_coeff,
        distance_std_coeff=config.goal_generation.distance_std_coeff,
        min_distance=config.goal_generation.min_distance,
        max_distance=config.goal_generation.max_distance,
        prefer_straight=config.goal_generation.prefer_straight,
        max_hops=config.goal_generation.max_hops,
        max_heading_mismatch_deg=config.goal_generation.max_heading_mismatch_deg,
        heading_penalty=config.goal_generation.heading_penalty,
        add_goal_noise=config.goal_generation.add_goal_noise,
        goal_noise_std_x=config.goal_generation.goal_noise_std_x,
        goal_noise_std_y=config.goal_generation.goal_noise_std_y,
        goal_noise_corr=config.goal_generation.goal_noise_corr,
        goal_noise_max_abs=config.goal_generation.goal_noise_max_abs,
        eps=config.goal_generation.eps,
        snap_tol=config.goal_generation.snap_tol,
        max_angle_deg=config.goal_generation.max_angle_deg,
        rng_seed=config.goal_generation.rng_seed,
        minimum_speed_threshold=config.goal_generation.minimum_speed_threshold,
        global_seed=config.goal_generation.global_seed,
        projection_sample_mode=config.goal_generation.projection_sample_mode,
        projection_softmax_temp=config.goal_generation.projection_softmax_temp,
        projection_k=config.goal_generation.projection_k,
    )
    config.goal_params = goal_params

    master_seed = config.seed
    set_seed_all(master_seed)
    np_rng = np.random.default_rng(master_seed)
    py_rng = random.Random(master_seed)

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
        config=config,
        np_rng=np_rng,
        py_rng=py_rng,
        render=args.render,
        debug=args.debug,
    )
    env.close()
    del env
    del env_config