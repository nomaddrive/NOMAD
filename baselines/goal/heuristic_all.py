import os
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib
from matplotlib.patches import Circle

from sklearn.neighbors import KDTree
import numpy as np

import math, random
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union
from scipy.spatial import cKDTree
import numpy as np
from dataclasses import dataclass
import networkx as nx

import torch
import madrona_gpudrive

from gpudrive.env.config import EnvConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.datatypes.control import ResponseType
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState, LocalEgoState
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints
from gpudrive.visualize import utils

# Import our goal prediction models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from baselines.goal.model import GoalPredictor, GoalPredictorMDN


def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(
    data_path: str,
    num_envs: int,
    dynamics_model: str = "delta_local",
    action_type: str = "multi_discrete",
    controllable_agent_selection: str = "no_static",
    mirror_x: bool = False,
    max_num_objects: int = 64,
    device: str = "cuda",
) -> GPUDriveTorchEnv:
    env_config = EnvConfig(
        dynamics_model=dynamics_model,
        controllable_agent_selection=controllable_agent_selection,
        collision_behavior="ignore",
        mirror_x=mirror_x,
    )

    data_loader = SceneDataLoader(
        root=data_path,
        batch_size=num_envs,
        dataset_size=num_envs,
        sample_with_replacement=False,
        seed=42,
        shuffle=True,
        file_prefix="",
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=data_loader,
        max_cont_agents=max_num_objects,
        device=device,
        action_type=action_type,
    )
    return env


# def get_inputs_from_env(env: GPUDriveTorchEnv, device: torch.device):
#     # Read tensors directly from the underlying simulator via the visualizer handle
#     sim = env.vis.sim_object

#     # Global agent state: build (B, A, 3) = (x, y, yaw)
#     ges = GlobalEgoState.from_tensor(
#         sim.absolute_self_observation_tensor(), backend="torch", device="cpu"
#     )
#     agent_ids = ges.id.to(device)
#     # These are CPU torch tensors; stack into (B,A,3) and move to device
#     agent_states = torch.stack([ges.pos_x, ges.pos_y, ges.rotation_angle], dim=-1).to(device)
#     # Find mask for valid agents
#     cont_agent_mask = env.cont_agent_mask.to(device)

#     raw_types = env.sim.info_tensor().to_torch().clone().to(env.device)[:, :, 4].long()
#     valid_mask = raw_types != int(madrona_gpudrive.EntityType._None)

#     # Roadgraph features: match collect_dataset.py pipeline exactly
#     # - one_hot types
#     # - normalize() roadgraph (demean/scale as defined in datatype)
#     # - concatenate features into 27-D vector per point
#     grp = GlobalRoadGraphPoints.from_tensor(
#         roadgraph_tensor=sim.map_observation_tensor(), backend="torch", device="cpu"
#     )
#     roadgraph_mask = grp.segment_height > 0.
#     roadgraph_mask = roadgraph_mask.to(device)
#     grp.one_hot_encode_road_point_types()
#     grp.normalize()
#     roadgraph = torch.cat(
#         [
#             grp.x.unsqueeze(-1),
#             grp.y.unsqueeze(-1),
#             grp.segment_length.unsqueeze(-1),
#             grp.segment_width.unsqueeze(-1),
#             grp.segment_height.unsqueeze(-1),
#             grp.orientation.unsqueeze(-1),
#             grp.type,
#         ],
#         dim=-1,
#     ).to(device)

#     return agent_states, valid_mask, cont_agent_mask, roadgraph, roadgraph_mask, agent_ids


def plot_filtered_agent_bounding_boxes(
    env_idx: int,
    ax: matplotlib.axes.Axes,
    agent_states: GlobalEgoState,
    cont_agent_mask: torch.Tensor,
    is_ok_mask: torch.Tensor,
    is_offroad_mask: torch.Tensor,
    is_collided_mask: torch.Tensor,
    response_type: Any,
    alpha: Optional[float] = 1.0,
    as_center_pts: bool = False,
    label: Optional[str] = None,
    plot_goal_points: bool = True,
    line_width_scale: int = 1.0,
    show_ids: bool = False,
) -> None:
    """Plots bounding boxes for agents filtered by environment index and mask.

    Args:
        env_idx: Environment indices to select specific environments.
        ax: Matplotlib axis for plotting.
        agent_state: The global state of agents from `GlobalEgoState`.
        is_ok_mask: Mask for agents that are in a valid state.
        is_offroad_mask: Mask for agents that are off-road.
        is_collided_mask: Mask for agents that have collided.
        response_type: Mask to filter static agents.
        alpha: Alpha value for drawing, i.e., 0 means fully transparent.
        as_center_pts: If True, only plot center points instead of full boxes.
        label: Label for the plotted elements.
        plot_goal_points: If True, plot goal points for agents.
        line_width_scale: Scale factor for line width.
        cont_agent_mask: Mask for controlled agents.
    """

    AGENT_COLOR_BY_STATE = {
        "ok": "#4B77BE",  # Controlled and doing fine
        "collided": "r",  # Controlled and collided
        "off_road": "orange",  # Controlled and off-road
        "log_replay": "#c7c7c7",  # Agents marked as expert controlled or static
    }

    def plot_agent_group_2d(bboxes, color,by_policy = False):
        """Helper function to plot a group of agents in 2D"""
        if not by_policy:
            utils.plot_numpy_bounding_boxes(
                ax=ax,
                bboxes=bboxes,
                color=color,
                alpha=alpha,
                line_width_scale=line_width_scale,
                as_center_pts=as_center_pts,
                label=label,
            )
            if show_ids:
                utils.plot_object_ids(
                    ax=ax,
                    bboxes=bboxes,
                    color='black',
                )
        else:
            num_policies = len(bboxes)
            utils.plot_numpy_bounding_boxes_multiple_policy(            
            ax=ax,
            bboxes_s=bboxes,
            colors=color[:num_policies],
            alpha=alpha,
            line_width_scale=line_width_scale,
            as_center_pts=as_center_pts,
            label=label,
                )
    # Off-road agents
    bboxes_controlled_offroad = np.stack(
        (
            agent_states.pos_x[env_idx, is_offroad_mask].numpy(),
            agent_states.pos_y[env_idx, is_offroad_mask].numpy(),
            agent_states.vehicle_length[env_idx, is_offroad_mask].numpy(),
            agent_states.vehicle_width[env_idx, is_offroad_mask].numpy(),
            agent_states.rotation_angle[env_idx, is_offroad_mask].numpy(),
            agent_states.id[env_idx, is_offroad_mask].numpy(),
        ),
        axis=1,
    )

    plot_agent_group_2d(
        bboxes_controlled_offroad, AGENT_COLOR_BY_STATE["off_road"]
    )

    # Plot goals
    if plot_goal_points:
        for mask, color in [
            (is_ok_mask, AGENT_COLOR_BY_STATE["ok"]),
            (is_offroad_mask, AGENT_COLOR_BY_STATE["off_road"]),
            (is_collided_mask, AGENT_COLOR_BY_STATE["collided"]),
        ]:
            if not mask.any():
                continue

            goal_x = agent_states.goal_x[env_idx, mask].numpy()
            goal_y = agent_states.goal_y[env_idx, mask].numpy()

            # Original 2D goal plotting
            ax.scatter(
                goal_x,
                goal_y,
                s=5 * 1.0,
                linewidth=1.5 * 1.0,
                c=color,
                marker="o",
            )
            for x, y in zip(goal_x, goal_y):
                circle = Circle(
                    (x, y),
                    radius=2.0,
                    color=color,
                    fill=False,
                    linestyle="--",
                )
                ax.add_patch(circle)

    # Collided agents
    bboxes_controlled_collided = np.stack(
        (
            agent_states.pos_x[env_idx, is_collided_mask].numpy(),
            agent_states.pos_y[env_idx, is_collided_mask].numpy(),
            agent_states.vehicle_length[env_idx, is_collided_mask].numpy(),
            agent_states.vehicle_width[env_idx, is_collided_mask].numpy(),
            agent_states.rotation_angle[env_idx, is_collided_mask].numpy(),
            agent_states.id[env_idx, is_collided_mask].numpy(),
        ),
        axis=1,
    )

    plot_agent_group_2d(
        bboxes_controlled_collided, AGENT_COLOR_BY_STATE["collided"]
    )

    # Living agents
    bboxes_controlled_ok = np.stack(
        (
            agent_states.pos_x[env_idx, is_ok_mask].numpy(),
            agent_states.pos_y[env_idx, is_ok_mask].numpy(),
            agent_states.vehicle_length[env_idx, is_ok_mask].numpy(),
            agent_states.vehicle_width[env_idx, is_ok_mask].numpy(),
            agent_states.rotation_angle[env_idx, is_ok_mask].numpy(),
            agent_states.id[env_idx, is_ok_mask].numpy()
        ),
        axis=1,
    )

    plot_agent_group_2d(
        bboxes_controlled_ok, AGENT_COLOR_BY_STATE["ok"]
    )

    # Plot log replay agents
    log_replay = (
        response_type.static[env_idx, :] | response_type.moving[env_idx, :]
    ) & ~cont_agent_mask[env_idx, :]

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

    plot_agent_group_2d(
        bboxes_static, AGENT_COLOR_BY_STATE["log_replay"]
    )


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


def endpoints(x, y, length, theta):
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    start = np.stack([x - dx, y - dy], axis=-1)
    end   = np.stack([x + dx, y + dy], axis=-1)
    return start, end


def order_segments_by_follow(start, end, eps=0.01):
    # Build “end -> next start” links
    tree = KDTree(start)
    n = len(start)
    succ, pred = [-1]*n, [-1]*n
    for i in range(n):
        idx = tree.query_radius(end[i][None], r=eps)[0]
        idx = [j for j in idx if j != i]
        if idx:
            j = idx[0]         # assume single chain inside a lane
            succ[i] = j
            pred[j] = i

    # head = the one with no predecessor
    heads = [i for i in range(n) if pred[i] == -1]
    head = heads[0] if heads else 0

    order = []
    cur = head
    while cur != -1:
        order.append(cur)
        cur = succ[cur]
    return order

def build_lane_polylines(x_coords, y_coords, segment_length, orientations, ids, eps=0.01):
    starts, ends = endpoints(x_coords, y_coords, segment_length, orientations)

    lanes = {}
    for i, lane_id in enumerate(np.unique(ids)):
        mask = (ids == lane_id)
        idx = np.where(mask)[0]
        order = order_segments_by_follow(starts[idx], ends[idx], eps)
        idx = idx[order]

        # stitch into a centerline polyline [p0, p1, ..., pN]
        pts = [starts[idx[0]].tolist()]
        for k in idx:
            pts.append(ends[k].tolist())

        lanes[lane_id] = dict(
            centerline=np.array(pts, float),
            start=pts[0],
            end=pts[-1],
            start_heading=orientations[idx[0]],
            end_heading=orientations[idx[-1]],
            seg_indices=idx,
        )
    return lanes


def initialize_vehicles_on_network(
    lanes: Dict,
    seg_tree: KDTree,
    seg_records: List,
    density_per_km: float,
    max_speed: float,
    heading_noise_std: float = 0.1,
    min_speed: float = 0.0,
    min_time_gap: float = 1.2,
    lateral_deviation_std: float = 0.3,
    min_distance: float = 5.0,
    speed_mean_percentile: float = 0.85,
    speed_std_percentile: float = 0.22,
) -> List[Dict[str, float]]:
    """
    Initialize vehicles on the lane network based on density with collision avoidance.
    
    Args:
        lanes: Dictionary of lane data with geometry
        seg_tree: KD-tree for spatial queries
        seg_records: Records mapping tree indices to lanes
        density_per_km: Number of vehicles per kilometer of lane
        max_speed: Maximum speed for vehicles (m/s)
        heading_noise_std: Std dev for heading noise in radians
        min_speed: Minimum speed for vehicles (m/s)
        min_time_gap: Minimum time gap between vehicles in seconds
        lateral_deviation_std: Std dev of lateral offset from lane center in meters
        min_distance: Minimum center-to-center distance between vehicles in meters
        speed_mean_percentile: Mean speed as percentile of max_speed (e.g., 0.85)
        speed_std_percentile: Std dev as percentile of max_speed (e.g., 0.22)
        
    Returns:
        List of vehicle dictionaries with position, heading, speed, lane_id, s
    """
    # Calculate total network length in kilometers
    total_length_m = sum(lane["length"] for lane in lanes.values())
    total_length_km = total_length_m / 1000.0
    
    # Calculate max number of vehicles based on density
    max_vehicles = int(np.round(density_per_km * total_length_km))
    
    if max_vehicles == 0:
        return []
    
    # Randomly sample number of vehicles between 1 and max_vehicles
    num_vehicles = np.random.randint(1, max_vehicles + 1)
    
    # Collect all lane segments with their cumulative probabilities (weighted by length)
    lane_ids = []
    segment_starts = []
    segment_lengths = []
    
    for lane_id, lane_data in lanes.items():
        n_segs = len(lane_data["seglen"])
        for seg_idx in range(n_segs):
            lane_ids.append(lane_id)
            segment_starts.append(lane_data["cumlen"][seg_idx])
            segment_lengths.append(lane_data["seglen"][seg_idx])
    
    segment_lengths = np.array(segment_lengths)
    # Probability of sampling each segment proportional to its length
    segment_probs = segment_lengths / segment_lengths.sum()
    
    vehicles = []
    # Track occupied positions per lane for efficient collision checking
    # Format: {lane_id: [(s_position, speed), ...]} sorted by s_position
    lane_occupancy = {lane_id: [] for lane_id in lanes.keys()}
    
    # Build KDTree for spatial collision checking (O(log n) query instead of O(n))
    vehicle_positions = []  # List of (x, y, required_distance) tuples
    vehicle_kdtree = None  # Will be rebuilt after each successful placement
    
    max_attempts = num_vehicles * 10  # Prevent infinite loop
    attempts = 0
    
    while len(vehicles) < num_vehicles and attempts < max_attempts:
        attempts += 1
        
        # Sample a segment
        seg_idx = np.random.choice(len(lane_ids), p=segment_probs)
        lane_id = lane_ids[seg_idx]
        lane_data = lanes[lane_id]
        
        # Sample position uniformly along the segment
        s_start = segment_starts[seg_idx]
        s_offset = np.random.uniform(0, segment_lengths[seg_idx])
        s = s_start + s_offset
        
        # Generate speed first (needed for collision checking)
        # Log-normal distribution based on traffic engineering literature
        target_mean = speed_mean_percentile * max_speed
        target_std = speed_std_percentile * max_speed
        variance = target_std ** 2
        mu = np.log(target_mean ** 2 / np.sqrt(target_mean ** 2 + variance))
        sigma = np.sqrt(np.log(1 + variance / target_mean ** 2))
        speed = np.random.lognormal(mu, sigma)
        speed = float(np.clip(speed, max(min_speed, 0.05 * max_speed), 1.1 * max_speed))
        
        # Calculate minimum distance: max of time-gap-based and absolute minimum
        time_gap_distance = speed * min_time_gap
        required_min_distance = max(time_gap_distance, min_distance)
        
        # Get candidate position WITH lateral deviation for accurate collision checking
        centerline_point, heading = point_at_s(lane_data, s)
        
        # Apply lateral deviation NOW (before collision checks)
        lateral_offset = np.random.normal(0, lateral_deviation_std)
        perp_x = -np.sin(heading)
        perp_y = np.cos(heading)
        candidate_point = centerline_point + np.array([lateral_offset * perp_x, lateral_offset * perp_y])
        
        collision = False
        
        # Check 1: Same-lane collision (efficient via lane_occupancy)
        occupied = lane_occupancy[lane_id]
        for (occupied_s, occupied_speed) in occupied:
            # Check distance in both directions
            distance = abs(s - occupied_s)
            # Use the larger of the two minimum distances for safety
            occupied_required_distance = max(occupied_speed * min_time_gap, min_distance)
            final_required_distance = max(required_min_distance, occupied_required_distance)
            
            if distance < final_required_distance:
                collision = True
                break
        
        # Check 2: Spatial collision with ALL vehicles (handles cross-lane conflicts)
        if not collision and vehicle_kdtree is not None:
            # Query KDTree for vehicles within required distance
            # Add small buffer to ensure we catch all potential collisions
            query_radius = required_min_distance * 1.1
            indices = vehicle_kdtree.query_radius([[candidate_point[0], candidate_point[1]]], 
                                                   r=query_radius)[0]
            
            # Check each nearby vehicle
            for idx in indices:
                vx, vy, v_required_dist = vehicle_positions[idx]
                spatial_distance = np.sqrt((candidate_point[0] - vx)**2 + (candidate_point[1] - vy)**2)
                # Use maximum of both vehicles' required distances
                final_required_distance = max(required_min_distance, v_required_dist)
                
                if spatial_distance < final_required_distance:
                    collision = True
                    break
        
        if not collision:
            # Use the already-calculated deviated position and heading
            position_x = float(candidate_point[0])
            position_y = float(candidate_point[1])
            
            # Add heading noise
            heading_noisy = heading + np.random.normal(0, heading_noise_std)
            
            vehicle = {
                "position_x": position_x,
                "position_y": position_y,
                "heading": float(heading_noisy),
                "speed": float(speed),
                "lane_id": lane_id,
                "s": float(s),
            }
            vehicles.append(vehicle)
            
            # Add to occupancy tracking and keep sorted
            lane_occupancy[lane_id].append((s, speed))
            lane_occupancy[lane_id].sort(key=lambda x: x[0])
            
            # Add to spatial tracking for cross-lane collision detection
            vehicle_positions.append((position_x, position_y, required_min_distance))
            
            # Rebuild KDTree with all vehicle positions for efficient spatial queries
            if len(vehicle_positions) > 0:
                positions_array = np.array([[vx, vy] for vx, vy, _ in vehicle_positions])
                vehicle_kdtree = KDTree(positions_array)
    
    if len(vehicles) < num_vehicles:
        print(f"Warning: Only placed {len(vehicles)}/{num_vehicles} vehicles without collisions after {attempts} attempts")
    
    return vehicles


def angle_diff(a, b):
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return abs(d)

def classify_turn(a):  # radians
    deg = np.degrees(a)
    if deg < 25:   
        return "straight"
    if 25 <= deg <= 45:  
        return "turn"
    else:
        raise ValueError("Angle too large for turn classification")

def build_lane_graph(lanes, snap_tol=0.30, max_angle=np.deg2rad(45)):
    starts = np.array([lanes[k]["start"] for k in lanes])
    start_ids = list(lanes.keys())
    tree = KDTree(starts)

    G = nx.DiGraph()
    for k, d in lanes.items():
        G.add_node(k, **d)

    for a, da in lanes.items():
        idxs = tree.query_radius(np.array(da["end"])[None], r=snap_tol)[0]
        for j in idxs:
            b = start_ids[j]
            if b == a: 
                continue
            ang = angle_diff(da["end_heading"], lanes[b]["start_heading"])
            if ang <= max_angle:
                G.add_edge(a, b, turn_angle=ang, turn_type=classify_turn(ang))
    return G


# ---- coloring helpers ----
def component_colors(G, cmap_name="tab20"):
    """Return {lane_id: rgba} so lanes in the same (weak) component share a color."""
    comps = list(nx.weakly_connected_components(G))  # or strongly_connected_components(G)
    cmap = plt.get_cmap(cmap_name, len(comps))
    id2color = {}
    for ci, comp in enumerate(comps):
        for nid in comp:
            id2color[nid] = cmap(ci)
    return id2color, len(comps)


# --- add arc-length bookkeeping to each lane ---
def prepare_lane_geometry(lanes):
    for k, d in lanes.items():
        pts = np.asarray(d["centerline"], float)         # (N,2)
        segs = pts[1:] - pts[:-1]                         # (N-1,2)
        seglen = np.linalg.norm(segs, axis=1)             # (N-1,)
        cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
        d.update(dict(points=pts, segments=segs, seglen=seglen,
                      cumlen=cumlen, length=float(cumlen[-1])))


# --- global segment index for fast nearest-lane queries ---
def build_segment_index(lanes):
    mids, records = [], []  # records[i] = (lane_id, segment_idx)
    for lane_id, d in lanes.items():
        a = d["points"][:-1]
        b = d["points"][1:]
        mid = 0.5*(a+b)
        mids.append(mid)
        records += [(lane_id, i) for i in range(len(mid))]
    mids = np.vstack(mids) if len(mids) else np.empty((0,2))
    tree = KDTree(mids) if len(mids) else None
    return tree, records


def project_to_segment(p, a, b):
    v = b - a
    l2 = float((v*v).sum())
    if l2 == 0.0:
        return a, 0.0, np.linalg.norm(p - a)
    t = np.clip(((p - a) @ v) / l2, 0.0, 1.0)
    proj = a + t * v
    return proj, t, float(np.linalg.norm(p - proj))

def closest_point_on_network(p, lanes, seg_tree, seg_records,
                             vehicle_heading: Optional[float] = None,
                             k: int = 24,
                             max_heading_mismatch: float = np.deg2rad(90.0),
                             heading_penalty: float = 5.0):
    """Find closest point on lane network considering optional vehicle heading.

    When vehicle_heading is provided, segments whose heading deviates more than
    max_heading_mismatch are ignored, and a combined cost = distance + heading_penalty * angle_diff
    is minimized.
    """
    # candidate segments by midpoint proximity, then exact projection
    idxs = seg_tree.query(p[None], k=min(k, len(seg_records)))[1][0]
    best = None
    for idx in idxs:
        lane_id, i = seg_records[idx]
        d = lanes[lane_id]
        a, b = d["points"][i], d["points"][i+1]
        proj, t, dist = project_to_segment(p, a, b)
        s = d["cumlen"][i] + t * d["seglen"][i]            # curvilinear s on the lane
        # segment heading
        seg_heading = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
        if vehicle_heading is not None:
            ang_mis = angle_diff(seg_heading, vehicle_heading)
            # gate out segments pointing nearly opposite direction
            if ang_mis > max_heading_mismatch:
                continue
            cost = float(dist + heading_penalty * ang_mis)
        else:
            cost = float(dist)
        if (best is None) or (cost < best[-1]):
            best = (lane_id, float(s), proj, cost)

    # Fallback: if all candidates were gated out, ignore heading and use pure distance
    if best is None:
        for idx in idxs:
            lane_id, i = seg_records[idx]
            d = lanes[lane_id]
            a, b = d["points"][i], d["points"][i+1]
            proj, t, dist = project_to_segment(p, a, b)
            s = d["cumlen"][i] + t * d["seglen"][i]
            if (best is None) or (dist < best[-1]):
                best = (lane_id, float(s), proj, float(dist))

    lane_id, s, proj, _ = best
    return lane_id, s, proj

def point_at_s(lane_dict, s):
    # return point & heading at curvilinear position s on the lane
    s = float(np.clip(s, 0.0, lane_dict["length"]))
    i = int(np.searchsorted(lane_dict["cumlen"], s, side="right") - 1)
    i = min(i, len(lane_dict["seglen"]) - 1)
    ds = s - lane_dict["cumlen"][i]
    t = ds / max(lane_dict["seglen"][i], 1e-12)
    a = lane_dict["points"][i]; b = lane_dict["points"][i+1]
    p = a + t * (b - a)
    heading = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
    return p, heading


def advance_along_graph(G, lanes, start_lane, s0, distance,
                        prefer_straight=False, max_hops=200):
    remain = float(distance)
    lane = start_lane
    s = float(s0)                         # lane-relative curvilinear coord at start
    path = []                             # each item: {lane_id, s_from, s_to, traveled}

    for _ in range(max_hops):
        # distance we can still move on this lane from current s
        avail = lanes[lane]["length"] - s
        step = min(remain, max(avail, 0.0))
        s1 = s + step

        # log only the portion we actually travel on this lane
        if step > 0.0:
            path.append(dict(lane_id=lane, s_from=s, s_to=s1, traveled=step, tol_length=lanes[lane]["length"]))

        # if we’ve satisfied the requested distance, stop here
        if step >= remain - 1e-9:
            p, hdg = point_at_s(lanes[lane], s1)
            return {
                "point": p, "heading": hdg,
                "lane_id": lane, "s": s1,
                "path": path, "finished": True
            }

        # otherwise we consumed this lane; hop to a successor
        remain -= step
        succs = list(G.successors(lane))
        if not succs:
            # dead end: stop at lane end
            p, hdg = point_at_s(lanes[lane], s1)
            return {
                "point": p, "heading": hdg,
                "lane_id": lane, "s": s1,
                "path": path, "finished": False, "reason": "dead_end"
            }

        if prefer_straight:
            ws = []
            for b in succs:
                ttype = G.edges[lane, b].get("turn_type", "straight")
                ws.append(3.0 if ttype == "straight" else (1.5 if ttype == "turn" else 0.8))
            ws = np.array(ws, float); ws /= ws.sum()
            lane = random.choices(succs, weights=ws)[0]
        else:
            random.random()
            lane = random.choice(succs)
            # print(f"Selected lane {lane} from successors {succs}")

        s = 0.0  # start of next lane

    # safety exit
    p, hdg = point_at_s(lanes[lane], s)
    return {"point": p, "heading": hdg, "lane_id": lane, "s": s,
            "path": path, "finished": False, "reason": "max_hops"}


@torch.no_grad()
def main():
    pa = argparse.ArgumentParser(description="Visualize predicted goal points over simulator scenes")
    pa.add_argument("--data_path", type=str, default="data/nuplan/test", help="Dataset root for scenes")
    pa.add_argument("--num_envs", type=int, default=2, help="Number of scenes/worlds to visualize")
    pa.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for model")
    pa.add_argument("--mirror_x", action="store_true", help="Mirror scenes across X axis")
    pa.add_argument("--controllable_agent_selection", type=str, default="no_static",
                    choices=["all_agents", "no_static", "no_expert", "no_static_no_expert"],
                    help="Which agents to control/plot")
    pa.add_argument("--action_type", type=str, default="multi_discrete", choices=["multi_discrete", "continuous", "discrete"])
    pa.add_argument("--dynamics_model", type=str, default="delta_local", choices=["delta_local", "state", "classic"])
    pa.add_argument("--max_num_objects", type=int, default=64)
    pa.add_argument("--zoom_radius", type=int, default=100)
    pa.add_argument("--out_dir", type=str, default="visualization/predicted_goals/", help="Directory to save images")
    pa.add_argument("--save_path", type=str, default="visualization/heuristic", help="Path to save images")
    pa.add_argument("--snap_tol", type=float, default=0.1, help="Tolerance for snapping lane segment endpoints")
    pa.add_argument("--eps", type=float, default=0.01, help="Tolerance for connecting lane segments")
    pa.add_argument("--max_angle", type=float, default=45.0, help="Max angle (degrees) for connecting lane segments")
    pa.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    pa.add_argument("--max_hops", type=int, default=200, help="Maximum hops when advancing along lane graph")
    pa.add_argument("--prefer_straight", action="store_true", help="Prefer straight connections when advancing along graph")
    pa.add_argument("--num_goals", type=int, default=8, help="Number of goal samples to generate")
    pa.add_argument("--distance_mean_coeff", type=float, default=9.0, help="Coefficient k for mean = k * speed")
    pa.add_argument("--distance_std_coeff", type=float, default=3.0, help="Coefficient c for std = c * speed")
    pa.add_argument("--min_distance", type=float, default=5.0, help="Lower clamp for sampled driving distance")
    pa.add_argument("--max_distance", type=float, default=200.0, help="Upper clamp for sampled driving distance")
    pa.add_argument("--add_goal_noise", type=bool, default=True, help="Add 2D Gaussian noise to goal points")
    pa.add_argument("--goal_noise_std_x", type=float, default=1.0, help="Std dev of goal noise in x (meters)")
    pa.add_argument("--goal_noise_std_y", type=float, default=1.0, help="Std dev of goal noise in y (meters)")
    pa.add_argument("--goal_noise_corr", type=float, default=0.0, help="Correlation [-1,1] between x and y noise")
    pa.add_argument("--goal", action="store_true", help="Generate goal points for vehicles (required)")
    pa.add_argument("--start", action="store_true", help="Generate custom vehicle start positions (requires --goal)")
    pa.add_argument("--vehicle_density_per_km", type=float, default=4.0, help="Number of vehicles per km of lane network (used with --start)")
    pa.add_argument("--heading_noise_std", type=float, default=0.15, help="Std dev of heading noise for custom vehicles (radians, used with --start)")

    args = pa.parse_args()

    # Validate flag combinations
    if args.start and not args.goal:
        print("Error: --start requires --goal flag")
        print("Usage: --goal (for goal generation only) or --goal --start (for custom vehicles + goals)")
        return
    
    if not args.goal:
        print("Error: --goal flag is required")
        print("Usage: --goal (for goal generation only) or --goal --start (for custom vehicles + goals)")
        return

    device = torch.device("cpu")
    # 1) Create environment
    env = build_env(
        data_path=args.data_path,
        num_envs=args.num_envs,
        dynamics_model=args.dynamics_model,
        action_type=args.action_type,
        controllable_agent_selection=args.controllable_agent_selection,
        mirror_x=args.mirror_x,
        max_num_objects=args.max_num_objects,
        device="cuda",
    )

    print("Environment generated.")

    # Global agent state: build (B, A, 3) = (x, y, yaw)
    ges = GlobalEgoState.from_tensor(
        env.sim.absolute_self_observation_tensor(), backend="torch", device="cpu"
    )
    # Find mask for valid agents
    cont_agent_mask = env.cont_agent_mask.to("cpu")

    road_graph = GlobalRoadGraphPoints.from_tensor(
        roadgraph_tensor=env.sim.map_observation_tensor(), backend="torch", device="cpu"
    )

    response_type = ResponseType.from_tensor(
        tensor=env.sim.response_type_tensor(), backend="torch", device="cpu"
    )

    self_obs = LocalEgoState.from_tensor(
        self_obs_tensor=env.sim.self_observation_tensor(), backend="torch", device="cpu"
    )
    agent_speed = self_obs.speed.to(device)

    # Ensure output directory exists
    save_dir = Path(args.save_path) / args.data_path.split('/')[-1]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get scenario file paths and extract names
    scenario_paths = env.data_loader.dataset[:args.num_envs]
    scenario_names = [Path(path).stem for path in scenario_paths]

    for env_idx in range(args.num_envs):
        scenario_name = scenario_names[env_idx]
        print(f"\n{'='*80}")
        print(f"Processing Scenario: {scenario_name}")
        print(f"{'='*80}")

        # create mask for road lane
        road_mask = road_graph.type[env_idx] == int(madrona_gpudrive.EntityType.RoadLane)

        # get valid road points
        x_coords = road_graph.x[env_idx, road_mask].cpu().numpy()
        y_coords = road_graph.y[env_idx, road_mask].cpu().numpy()
        segment_length = road_graph.segment_length[env_idx, road_mask].cpu().numpy()
        orientations = road_graph.orientation[env_idx, road_mask].cpu().numpy()
        ids = road_graph.id[env_idx, road_mask].cpu().numpy().astype(int)

        lanes = build_lane_polylines(x_coords, y_coords, segment_length, orientations, ids, eps=args.eps)
        G = build_lane_graph(lanes, snap_tol=args.snap_tol, max_angle=np.deg2rad(args.max_angle))
        print(f"lanes: {len(lanes)}, edges: {G.number_of_edges()}")

        some_lane = next(iter(lanes.keys()))
        print("Successors:", list(G.successors(some_lane)))
        print("Predecessors:", list(G.predecessors(some_lane)))
        print("Lane attrs:", G.nodes[some_lane].keys())  # centerline, width, ...

        id2color, ncomp = component_colors(G)

        fig, ax = plt.subplots(figsize=(6,6), dpi=200)
        # keep map as soft background
        plot_road_edges_and_lines(road_graph, env_idx, ax, base_alpha=0.35)
        # Plot log replay agents
        log_replay = (
            response_type.static[env_idx, :] | response_type.moving[env_idx, :]
        ) & ~cont_agent_mask[env_idx, :]

        pos_x = ges.pos_x[env_idx, log_replay]
        pos_y = ges.pos_y[env_idx, log_replay]
        rotation_angle = ges.rotation_angle[env_idx, log_replay]
        vehicle_length = ges.vehicle_length[env_idx, log_replay]
        vehicle_width = ges.vehicle_width[env_idx, log_replay]
        ids = ges.id[env_idx, log_replay]

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

        prepare_lane_geometry(lanes)
        seg_tree, seg_records = build_segment_index(lanes)

        # Plot scenario vehicles if not generating custom start positions
        if not args.start:
            # Build list of controllable agents
            controlled_idx = torch.where(cont_agent_mask[env_idx])[0].cpu().numpy().tolist()

            # Assign a consistent color per agent index
            cmap = plt.get_cmap("tab20")
            def agent_color(k):
                return cmap(k % cmap.N)

            # Draw vehicles and original goals per agent using per-agent color
            for k, aidx in enumerate(controlled_idx):
                color = agent_color(k)
                bx = np.array([
                    [
                        ges.pos_x[env_idx, aidx].item(),
                        ges.pos_y[env_idx, aidx].item(),
                        ges.vehicle_length[env_idx, aidx].item(),
                        ges.vehicle_width[env_idx, aidx].item(),
                        ges.rotation_angle[env_idx, aidx].item(),
                        ges.id[env_idx, aidx].item(),
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
                gx0 = ges.goal_x[env_idx, aidx].item()
                gy0 = ges.goal_y[env_idx, aidx].item()
                ax.scatter([gx0], [gy0], s=18, marker="s", facecolors="none", edgecolors=color, linewidths=0.5, zorder=4)

        # Determine vehicles to use
        if args.start:
            # Get max speed from scenario for realistic speed sampling
            max_speed_scenario = float(agent_speed[env_idx, :].max().item())
            if max_speed_scenario <= 0:
                max_speed_scenario = 15.0  # Default ~54 km/h if no valid speeds
            
            # Initialize custom vehicles
            custom_vehicles = initialize_vehicles_on_network(
                lanes=lanes,
                seg_tree=seg_tree,
                seg_records=seg_records,
                density_per_km=args.vehicle_density_per_km,
                max_speed=max_speed_scenario,
                heading_noise_std=args.heading_noise_std,
                min_speed=0.0,
            )
            
            print(f"\nInitialized {len(custom_vehicles)} custom vehicles (density: {args.vehicle_density_per_km} veh/km)")
            print(f"Vehicle initialization details:")
            for i, veh in enumerate(custom_vehicles[:5]):  # Show first 5
                print(f"  Vehicle {i}: lane_id={veh['lane_id']}, s={veh['s']:.2f}m, speed={veh['speed']:.2f}m/s")
            if len(custom_vehicles) > 5:
                print(f"  ... and {len(custom_vehicles) - 5} more vehicles")
            
            # Create list of vehicle data
            vehicle_list = custom_vehicles
        else:
            # Use scenario vehicles
            controlled_idx = torch.where(cont_agent_mask[env_idx])[0].cpu().numpy().tolist()
            vehicle_list = []
            for aidx in controlled_idx:
                vehicle_list.append({
                    "position_x": ges.pos_x[env_idx, aidx].item(),
                    "position_y": ges.pos_y[env_idx, aidx].item(),
                    "heading": ges.rotation_angle[env_idx, aidx].item(),
                    "speed": float(agent_speed[env_idx, aidx].item()),
                    "agent_idx": int(aidx),
                    "original_goal_x": ges.goal_x[env_idx, aidx].item(),
                    "original_goal_y": ges.goal_y[env_idx, aidx].item(),
                })
        
        # Shared noise covariance for goal jitter
        std_x = max(0.0, float(args.goal_noise_std_x))
        std_y = max(0.0, float(args.goal_noise_std_y))
        rho = float(np.clip(args.goal_noise_corr, -1.0, 1.0))
        cov_xy = rho * std_x * std_y
        Sigma = np.array([[std_x**2, cov_xy], [cov_xy, std_y**2]], dtype=float)

        # Assign colors
        cmap = plt.get_cmap("tab20")
        def agent_color(k):
            return cmap(k % cmap.N)
        
        # Plot vehicles and generate goals
        for k, vehicle in enumerate(vehicle_list):
            color = agent_color(k)
            sx = vehicle["position_x"]
            sy = vehicle["position_y"]
            veh_hdg = vehicle["heading"]
            speed_i = vehicle["speed"]
            
            # Plot vehicle (as a point or small marker for custom vehicles)
            if args.start:
                # Plot as scatter point with heading arrow
                ax.scatter([sx], [sy], s=30, color=color, marker='o', zorder=4, edgecolors='black', linewidths=0.5)
                # Draw heading arrow
                arrow_len = 3.0
                dx = arrow_len * np.cos(veh_hdg)
                dy = arrow_len * np.sin(veh_hdg)
                ax.arrow(sx, sy, dx, dy, head_width=1.0, head_length=1.5, 
                        fc=color, ec=color, alpha=0.7, zorder=4, length_includes_head=True)
            else:
                # Plot as bounding box for scenario vehicles
                bx = np.array([[
                    sx, sy,
                    ges.vehicle_length[env_idx, vehicle["agent_idx"]].item(),
                    ges.vehicle_width[env_idx, vehicle["agent_idx"]].item(),
                    veh_hdg,
                    ges.id[env_idx, vehicle["agent_idx"]].item(),
                ]])
                utils.plot_numpy_bounding_boxes(
                    ax=ax, bboxes=bx, color=color, alpha=0.9,
                    line_width_scale=0.5, as_center_pts=False, label=None,
                )
                # Plot original goal
                gx0 = vehicle["original_goal_x"]
                gy0 = vehicle["original_goal_y"]
                ax.scatter([gx0], [gy0], s=18, marker="s", facecolors="none", 
                          edgecolors=color, linewidths=0.5, zorder=4)
            
            # Generate goals
            mean_dist = args.distance_mean_coeff * max(speed_i, 0.0)
            std_dist = max(1e-3, args.distance_std_coeff * max(speed_i, 0.0))

            # For custom vehicles, use their assigned lane_id and s directly
            # For scenario vehicles, need to snap to network
            if args.start:
                lane_id = vehicle["lane_id"]
                s_on_lane = vehicle["s"]
                if k < 3:  # Log first 3 vehicles
                    print(f"\n  Vehicle {k}: Using assigned lane_id={lane_id}, s={s_on_lane:.2f}m for graph traversal")
            else:
                p0 = np.asarray((sx, sy), float)
                lane_id, s_on_lane, _ = closest_point_on_network(p0, lanes, seg_tree, seg_records, vehicle_heading=veh_hdg)
                if k < 3:  # Log first 3 vehicles
                    print(f"\n  Vehicle {k}: Snapped to lane_id={lane_id}, s={s_on_lane:.2f}m for graph traversal")

            goals = []
            for n in range(int(args.num_goals)):
                sample = np.random.normal(mean_dist, std_dist)
                sample = float(np.clip(sample, args.min_distance, args.max_distance))
                res = advance_along_graph(G, lanes, lane_id, s_on_lane, sample, prefer_straight=args.prefer_straight, max_hops=args.max_hops)
                target_xy  = np.asarray(res["point"], float)
                if args.add_goal_noise:
                    noise = np.random.multivariate_normal([0.0, 0.0], Sigma)
                    target_xy = target_xy + noise
                goals.append(target_xy)

            # Plot generated goals with the same color
            if goals:
                gx = [g[0] for g in goals]
                gy = [g[1] for g in goals]
                sc = ax.scatter(gx, gy, color=color, s=18, marker="x", zorder=5, linewidths=0.5)
                try:
                    sc.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
                except Exception:
                    pass

        # Set view center
        if args.start and len(vehicle_list) > 0:
            center_x = vehicle_list[0]["position_x"]
            center_y = vehicle_list[0]["position_y"]
        else:
            center_x = ges.pos_x[env_idx, 0].item()
            center_y = ges.pos_y[env_idx, 0].item()

        # Use configured zoom radius around center
        R = float(getattr(args, "zoom_radius", 100))
        ax.set_xlim(center_x - R, center_x + R)
        ax.set_ylim(center_y - R, center_y + R)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_aspect("equal")
        fig.tight_layout(pad=0.05)
        out_file = save_dir / f"{scenario_name}_eps{args.eps}_snaptol{args.snap_tol}_ang{int(args.max_angle)}.png"
        plt.savefig(out_file, dpi=400, bbox_inches='tight', facecolor="white")
        print(f"\nSaved visualization to: {out_file}")
        plt.close(fig)



if __name__ == "__main__":
    main()
