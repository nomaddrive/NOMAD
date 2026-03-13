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
from tqdm import tqdm

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

                if road_point_type == int(madrona_gpudrive.EntityType.RoadEdge):
                    arrow_len = length
                    hw = 0.3 * arrow_len
                    hl = 0.4 * arrow_len
                    ax.arrow(
                        x, y,
                        arrow_len * np.cos(orientation), arrow_len * np.sin(orientation),
                        head_width=0.5, head_length=0.5,
                        fc='grey', ec='grey',
                        length_includes_head=True, alpha=0.9, zorder=2
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


def sample_truncated_bivariate_normal(Sigma: np.ndarray, max_abs: float = 1.5,
                                      mean: Optional[np.ndarray] = None,
                                      max_tries: int = 200,
                                      rng: Optional[Any] = None) -> np.ndarray:
    """Sample from a 2D Gaussian N(mean, Sigma) truncated to box [-max_abs, max_abs]^2.

    Uses simple rejection sampling; falls back to clipping if no sample is found
    within max_tries draws. Preserves correlation structure of Sigma when accepted.
    """
    if mean is None:
        mean = np.zeros(2, dtype=float)
    if rng is None:
        # fall back to global numpy RNG (less reproducible if mixed with other code)
        local_rng = np.random
        multivar = lambda m, S: local_rng.multivariate_normal(m, S)
    else:
        multivar = lambda m, S: rng.multivariate_normal(m, S)
    for _ in range(max_tries):
        z = multivar(mean, Sigma)
        if np.all(np.abs(z) <= max_abs):
            return z
    # Fallback: clip a final draw to ensure bound is respected
    z = multivar(mean, Sigma)
    return np.clip(z, -max_abs, max_abs)


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
                             heading_penalty: float = 5.0,
                             sample_mode: str = "argmin",  # "argmin" or "softmax"
                             softmax_temp: float = 1.0,
                             rng: Optional[Any] = None):
    """Find closest point on lane network considering optional vehicle heading.

    When vehicle_heading is provided, segments whose heading deviates more than
    max_heading_mismatch are ignored, and a combined cost = distance + heading_penalty * angle_diff
    is minimized.
    """
    # candidate segments by midpoint proximity, then exact projection
    search_k = k
    max_search_k = max(k * 4, 200)

    while True:
        idxs = seg_tree.query(p[None], k=min(search_k, len(seg_records)))[1][0]
        candidates = []  # list of (lane_id, s, proj, cost)
        for idx in idxs:
            lane_id, i = seg_records[idx]
            d = lanes[lane_id]
            a, b = d["points"][i], d["points"][i+1]
            proj, t, dist = project_to_segment(p, a, b)
            s = d["cumlen"][i] + t * d["seglen"][i]            # curvilinear s on the lane
            seg_heading = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            if vehicle_heading is not None:
                ang_mis = angle_diff(seg_heading, vehicle_heading)
                if ang_mis > max_heading_mismatch:
                    continue
                cost = float(dist + heading_penalty * ang_mis)
            else:
                cost = float(dist)
            candidates.append((lane_id, float(s), proj, cost))

        if len(candidates) >= int(k / 2) or search_k >= len(seg_records) or search_k >= max_search_k:
            break
        search_k *= 2

    # If heading gating removed all candidates, allow pure distance retry
    if not candidates:
        for idx in idxs:
            lane_id, i = seg_records[idx]
            d = lanes[lane_id]
            a, b = d["points"][i], d["points"][i+1]
            proj, t, dist = project_to_segment(p, a, b)
            s = d["cumlen"][i] + t * d["seglen"][i]
            candidates.append((lane_id, float(s), proj, float(dist)))

    # Argmin fallback if sampling disabled or only one candidate
    if sample_mode not in ("softmax", "argmin"):
        sample_mode = "argmin"
    if (sample_mode == "argmin") or len(candidates) == 1:
        best = min(candidates, key=lambda x: x[-1])
    else:
        # softmax over negative cost (lower cost -> higher weight)
        temp = max(1e-6, float(softmax_temp))
        costs = np.array([c[-1] for c in candidates], float)
        w = np.exp(-costs / temp)
        w_sum = w.sum()
        if w_sum <= 0:
            best = min(candidates, key=lambda x: x[-1])
        else:
            w /= w_sum
            if rng is None:
                # fall back to numpy global RNG
                choice_idx = int(np.random.choice(len(candidates), p=w))
            else:
                # Support both numpy Generator and random.Random
                if hasattr(rng, "choice"):
                    choice_idx = int(rng.choice(len(candidates), p=w))  # numpy Generator
                else:
                    # random.Random has no weighted choice; emulate
                    # cumulative method
                    r = rng.random()
                    cum = 0.0
                    choice_idx = 0
                    for i, wi in enumerate(w):
                        cum += wi
                        if r <= cum:
                            choice_idx = i
                            break
            best = candidates[choice_idx]

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
                        prefer_straight=False, max_hops=200,
                        rng_choice: Optional[Any] = None):
    remain = float(distance)
    lane = start_lane
    s = float(s0)                         # lane-relative curvilinear coord at start
    path = []                             # each item: {lane_id, s_from, s_to, traveled}
    # branching_stack holds frames for lanes where multiple successors existed.
    # Each frame: {lane, remaining, remain_after_end, path_len}
    branching_stack = []
    # Avoid repeating the same (u->v) edge many times
    visited_edges = set()

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
            # Dead end: attempt backtracking to last branching point with untried successors
            while branching_stack:
                frame = branching_stack.pop()  # last branching point
                if frame["remaining"]:
                    # Truncate path to the state at branching
                    path = path[:frame["path_len"]]
                    # Choose next untried successor
                    nxt = frame["remaining"].pop(0)
                    # push updated frame back for further alternatives (if any remain)
                    if frame["remaining"]:
                        branching_stack.append(frame)
                    # restore remaining distance and start new lane
                    remain = frame["remain_after_end"]
                    lane = nxt
                    s = 0.0
                    visited_edges.add((frame["lane"], nxt))
                    # Continue traversal from this alternative successor
                    break
            else:
                # No branching point left -> true dead end
                p, hdg = point_at_s(lanes[lane], s1)
                return {
                    "point": p, "heading": hdg,
                    "lane_id": lane, "s": s1,
                    "path": path, "finished": False, "reason": "dead_end"
                }
            # proceed to next hop iteration
            continue

        # Decide successor order
        if rng_choice is None:
            # fallback to python's random (expects external seeding via set_seed_all)
            local_rand = random
        else:
            local_rand = rng_choice

        if prefer_straight:
            ws = []
            for b in succs:
                ttype = G.edges[lane, b].get("turn_type", "straight")
                ws.append(3.0 if ttype == "straight" else (1.5 if ttype == "turn" else 0.8))
            ws = np.array(ws, float); denom = max(ws.sum(), 1e-9); ws /= denom
            # Weighted sampling once; remaining order excludes chosen successor
            chosen_index = int(local_rand.choices(range(len(succs)), weights=ws, k=1)[0])
            remaining_indices = [i for i in range(len(succs)) if i != chosen_index]
            # Keep remaining in weight-desc order for deterministic exploration if backtracked
            remaining_indices.sort(key=lambda i: -ws[i])
            order = [succs[chosen_index]] + [succs[i] for i in remaining_indices]
        else:
            order = list(succs)
            local_rand.shuffle(order)

        # Filter out already-visited edges from current lane
        order = [b for b in order if (lane, b) not in visited_edges]

        if not order:
            # Nothing new to try from here; treat like dead-end and backtrack
            # Put a dummy empty frame so the same logic applies below
            while branching_stack:
                frame = branching_stack.pop()
                if frame["remaining"]:
                    path = path[:frame["path_len"]]
                    nxt = frame["remaining"].pop(0)
                    if frame["remaining"]:
                        branching_stack.append(frame)
                    remain = frame["remain_after_end"]
                    lane = nxt
                    s = 0.0
                    visited_edges.add((frame["lane"], nxt))
                    break
            else:
                p, hdg = point_at_s(lanes[lane], s1)
                return {"point": p, "heading": hdg, "lane_id": lane, "s": s1,
                        "path": path, "finished": False, "reason": "dead_end"}
            continue

        chosen_successor = order[0]
        visited_edges.add((lane, chosen_successor))
        # Record branching info if multiple untried successors available
        if len(order) > 1:
            branching_stack.append(dict(
                lane=lane,
                remaining=order[1:],
                remain_after_end=remain,
                path_len=len(path),
            ))
        lane = chosen_successor

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
    pa.add_argument("--goal_noise_max_abs", type=float, default=1.2, help="Max absolute value per axis for goal noise")
    pa.add_argument("--minimum_speed_threshold", type=float, default=2.0, help="Minimum speed to consider for goal distance sampling")
    pa.add_argument("--global_seed", type=int, default=None, help="Overrides --seed; used for all RNGs if set")
    pa.add_argument("--projection_sample_mode", type=str, default="softmax", choices=["argmin", "softmax"], help="Mode for selecting closest lane segment")
    pa.add_argument("--projection_softmax_temp", type=float, default=1.0, help="Temperature for softmax sampling of lane projection")
    pa.add_argument("--projection_k", type=int, default=8, help="Number of candidate segments to consider for lane projection")
    pa.add_argument("--heading_penalty", type=float, default=5.0, help="Penalty factor for heading mismatch in lane projection")

    args = pa.parse_args()

    # Unified seeding
    master_seed = args.global_seed if args.global_seed is not None else args.seed
    set_seed_all(master_seed)
    np_rng = np.random.default_rng(master_seed)
    py_rng = random.Random(master_seed)
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
    save_dir = Path(args.save_path) / args.data_path.split('/')[-1] / str(args.mirror_x)
    save_dir.mkdir(parents=True, exist_ok=True)

    for _, env_idx in tqdm(enumerate(range(args.num_envs)), total=args.num_envs, desc="Visualizing scenes"):

        filename = env.get_env_filenames()[env_idx]

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
        # print(f"lanes: {len(lanes)}, edges: {G.number_of_edges()}")

        # some_lane = next(iter(lanes.keys()))
        # print("Successors:", list(G.successors(some_lane)))
        # print("Predecessors:", list(G.predecessors(some_lane)))
        # print("Lane attrs:", G.nodes[some_lane].keys())  # centerline, width, ...

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
        # Build list of controllable agents (we don't filter offroad/collision as requested)
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

            # Start point marker
            # sx, sy = bx[0, 0], bx[0, 1]
            # ax.scatter([sx], [sy], s=14, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, zorder=4)

            # Original goal marker
            gx0 = ges.goal_x[env_idx, aidx].item()
            gy0 = ges.goal_y[env_idx, aidx].item()
            ax.scatter([gx0], [gy0], s=18, marker="s", facecolors="none", edgecolors=color, linewidths=0.5, zorder=4)
        # Choose a reasonable arrow length relative to typical segment length
        arrow_len = max(3.0, float(np.median(segment_length)) * 0.5)
        for k, d in lanes.items():
            c = d["centerline"]
            color = id2color[k]
            # ax.plot(c[:,0], c[:,1], color=color, linewidth=1)
            # ax.scatter(c[:,0], c[:,1], color=color, s=0.5, alpha=0.7, zorder=2)
            # Draw arrows for start and end headings of each lane polyline
            sx, sy = d["start"]
            ex, ey = d["end"]
            sh, eh = d["start_heading"], d["end_heading"]
            hw = 0.30 * arrow_len
            hl = 0.40 * arrow_len
            ax.arrow(
                sx, sy,
                arrow_len * np.cos(sh), arrow_len * np.sin(sh),
                head_width=hw, head_length=hl,
                fc=color, ec=color,
                length_includes_head=True, alpha=0.9
            )
            ax.arrow(
                ex, ey,
                arrow_len * np.cos(eh), arrow_len * np.sin(eh),
                head_width=hw, head_length=hl,
                fc=color, ec=color,
                length_includes_head=True, alpha=0.9
            )

        prepare_lane_geometry(lanes)
        seg_tree, seg_records = build_segment_index(lanes)

        # Shared noise covariance for goal jitter
        std_x = max(0.0, float(args.goal_noise_std_x))
        std_y = max(0.0, float(args.goal_noise_std_y))
        rho = float(np.clip(args.goal_noise_corr, -1.0, 1.0))
        cov_xy = rho * std_x * std_y
        Sigma = np.array([[std_x**2, cov_xy], [cov_xy, std_y**2]], dtype=float)

        # Generate and plot goals per controllable agent with consistent color
        for k, aidx in enumerate(controlled_idx):
            color = agent_color(k)
            sx = ges.pos_x[env_idx, aidx].item()
            sy = ges.pos_y[env_idx, aidx].item()
            speed_i = float(agent_speed[env_idx, aidx].item())
            # Set a minimum speed to avoid zero-mean/stdev
            speed_i = max(speed_i, args.minimum_speed_threshold)
            mean_dist = args.distance_mean_coeff * max(speed_i, 0.0)
            std_dist = max(1e-3, args.distance_std_coeff * max(speed_i, 0.0))

            p0 = np.asarray((sx, sy), float)
            veh_hdg = float(ges.rotation_angle[env_idx, aidx].item())

            goals = []
            for n in range(int(args.num_goals)):
                lane_id, s_on_lane, _ = closest_point_on_network(
                    p0, lanes, seg_tree, seg_records,
                    vehicle_heading=veh_hdg, k=args.projection_k,
                    sample_mode=args.projection_sample_mode,
                    softmax_temp=args.projection_softmax_temp,
                    heading_penalty=args.heading_penalty,
                    rng=np_rng
                )
                sample = float(np_rng.normal(mean_dist, std_dist))
                sample = float(np.clip(sample, args.min_distance, args.max_distance))
                res = advance_along_graph(G, lanes, lane_id, s_on_lane, sample,
                                           prefer_straight=args.prefer_straight,
                                           max_hops=args.max_hops,
                                           rng_choice=py_rng)
                target_xy  = np.asarray(res["point"], float)
                if args.add_goal_noise:
                    noise = sample_truncated_bivariate_normal(Sigma, max_abs=float(args.goal_noise_max_abs), rng=np_rng)
                    target_xy = target_xy + noise
                goals.append(target_xy)

            # Plot generated goals with the same color
            # if goals:
            #     gx = [g[0] for g in goals]
            #     gy = [g[1] for g in goals]
            #     sc = ax.scatter(gx, gy, color=color, s=18, marker="x", zorder=5, linewidths=0.5)
            #     try:
            #         sc.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
            #     except Exception:
            #         pass

        center_x = ges.pos_x[
            env_idx, 0
        ].item()
        center_y = ges.pos_y[
            env_idx, 0
        ].item()

        # ax.scatter([x0], [y0], color="green", s=10, label="start")
        # Plot goals with white stroke for contrast
        # gx = [x for x, y in goals]
        # gy = [y for x, y in goals]
        # sc = ax.scatter(gx, gy, color="red", s=24, marker="x", zorder=5)
        try:
            sc.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
        except Exception:
            pass

        # Use configured zoom radius around ego
        R = float(getattr(args, "zoom_radius", 100))
        ax.set_xlim(center_x - R, center_x + R)
        ax.set_ylim(center_y - R, center_y + R)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_aspect("equal")
        fig.tight_layout(pad=0.05)
        out_file = save_dir / f"lanes_env{env_idx}_file{filename}_probstart{args.projection_k}_headingpenalty{args.heading_penalty}.png"
        plt.savefig(out_file, dpi=400, bbox_inches='tight', facecolor="white")
        plt.close(fig)



if __name__ == "__main__":
    main()
