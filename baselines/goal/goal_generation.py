"""Utilities for generating new goal positions for controllable agents.

This module factors out the geometry helpers from ``baselines/goal/heuristic.py``
and exposes a small API for programmatic goal generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple, Any

import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
import torch

import madrona_gpudrive


@dataclass
class LaneResources:
    """Pre-computed lane data structures for fast goal projection."""

    lanes: Dict[int, Dict[str, np.ndarray]]
    graph: nx.DiGraph
    segment_tree: Optional[KDTree]
    segment_records: List[Tuple[int, int]]


@dataclass
class GoalGenerationParams:
    """Hyper-parameters controlling the goal sampling process."""

    num_sets: int = 4
    distance_mean_coeff: float = 9.0
    distance_std_coeff: float = 3.0
    min_distance: float = 5.0
    max_distance: float = 200.0
    prefer_straight: bool = False
    max_hops: int = 200
    max_heading_mismatch_deg: float = 90.0
    heading_penalty: float = 5.0
    add_goal_noise: bool = True
    goal_noise_std_x: float = 1.0
    goal_noise_std_y: float = 1.0
    goal_noise_corr: float = 0.0
    goal_noise_max_abs: float = 1.2
    eps: float = 0.01
    snap_tol: float = 0.1
    max_angle_deg: float = 45.0
    rng_seed: int = 42
    global_seed: int = 42
    minimum_speed_threshold: float = 2.0
    projection_sample_mode: str = "softmax"
    projection_softmax_temp: float = 1.0
    projection_k: int = 8


    def heading_mismatch_rad(self) -> float:
        return math.radians(self.max_heading_mismatch_deg)

    def max_angle_rad(self) -> float:
        return math.radians(self.max_angle_deg)

    def goal_noise_cov(self) -> np.ndarray:
        std_x = max(0.0, float(self.goal_noise_std_x))
        std_y = max(0.0, float(self.goal_noise_std_y))
        rho = float(np.clip(self.goal_noise_corr, -1.0, 1.0))
        cov_xy = rho * std_x * std_y
        return np.array([[std_x**2, cov_xy], [cov_xy, std_y**2]], dtype=float)


def endpoints(x, y, length, theta):
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    start = np.stack([x - dx, y - dy], axis=-1)
    end = np.stack([x + dx, y + dy], axis=-1)
    return start, end


def order_segments_by_follow(start, end, eps=0.01):
    tree = KDTree(start)
    n = len(start)
    succ, pred = [-1] * n, [-1] * n
    for i in range(n):
        idx = tree.query_radius(end[i][None], r=eps)[0]
        idx = [j for j in idx if j != i]
        if idx:
            j = idx[0]
            succ[i] = j
            pred[j] = i

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

    lanes: Dict[int, Dict[str, np.ndarray]] = {}
    for lane_id in np.unique(ids):
        mask = ids == lane_id
        idx = np.where(mask)[0]
        order = order_segments_by_follow(starts[idx], ends[idx], eps)
        idx = idx[order]

        pts = [starts[idx[0]].tolist()]
        for k in idx:
            pts.append(ends[k].tolist())

        lanes[int(lane_id)] = dict(
            centerline=np.array(pts, float),
            start=pts[0],
            end=pts[-1],
            start_heading=orientations[idx[0]],
            end_heading=orientations[idx[-1]],
            seg_indices=idx,
        )
    return lanes


def angle_diff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def classify_turn(angle_rad: float) -> str:
    deg = np.degrees(angle_rad)
    if deg < 25:
        return "straight"
    if 25 <= deg <= 45:
        return "turn"
    return "sharp"


def build_lane_graph(lanes, snap_tol=0.30, max_angle=np.deg2rad(45)):
    starts = np.array([lanes[k]["start"] for k in lanes])
    start_ids = list(lanes.keys())
    tree = KDTree(starts) if len(starts) else None

    G = nx.DiGraph()
    for k, d in lanes.items():
        G.add_node(k, **d)

    if tree is None:
        return G

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


def prepare_lane_geometry(lanes):
    for d in lanes.values():
        pts = np.asarray(d["centerline"], float)
        if len(pts) < 2:
            d.update(points=pts, segments=np.empty((0, 2)), seglen=np.array([0.0]), cumlen=np.array([0.0]), length=0.0)
            continue
        segs = pts[1:] - pts[:-1]
        seglen = np.linalg.norm(segs, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
        d.update(
            points=pts,
            segments=segs,
            seglen=seglen,
            cumlen=cumlen,
            length=float(cumlen[-1]),
        )


def build_segment_index(lanes):
    mids: List[np.ndarray] = []
    records: List[Tuple[int, int]] = []
    for lane_id, d in lanes.items():
        if len(d.get("points", [])) < 2:
            continue
        a = d["points"][:-1]
        b = d["points"][1:]
        mid = 0.5 * (a + b)
        mids.append(mid)
        records += [(lane_id, i) for i in range(len(mid))]
    if not mids:
        return None, []
    mids_arr = np.vstack(mids)
    tree = KDTree(mids_arr)
    return tree, records


def project_to_segment(p, a, b):
    v = b - a
    l2 = float((v * v).sum())
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

    return candidates


def point_at_s(lane_dict, s):
    s = float(np.clip(s, 0.0, lane_dict["length"]))
    if lane_dict["length"] <= 0:
        return lane_dict["points"][0], 0.0
    i = int(np.searchsorted(lane_dict["cumlen"], s, side="right") - 1)
    i = min(i, len(lane_dict["seglen"]) - 1)
    ds = s - lane_dict["cumlen"][i]
    seg_len = max(lane_dict["seglen"][i], 1e-12)
    t = ds / seg_len
    a = lane_dict["points"][i]
    b = lane_dict["points"][i + 1]
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
                return None
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
                return None
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



def build_lane_resources(
    road_graph,
    env_idx: int,
    params: GoalGenerationParams,
) -> Optional[LaneResources]:
    lane_type = int(madrona_gpudrive.EntityType.RoadLane)
    lane_mask = (road_graph.type[env_idx] == lane_type).cpu()
    if not lane_mask.any():
        return None

    x_coords = road_graph.x[env_idx, lane_mask].cpu().numpy()
    y_coords = road_graph.y[env_idx, lane_mask].cpu().numpy()
    segment_length = road_graph.segment_length[env_idx, lane_mask].cpu().numpy()
    orientations = road_graph.orientation[env_idx, lane_mask].cpu().numpy()
    ids = road_graph.id[env_idx, lane_mask].cpu().numpy().astype(int)

    lanes = build_lane_polylines(
        x_coords,
        y_coords,
        segment_length,
        orientations,
        ids,
        eps=params.eps,
    )
    if not lanes:
        return None

    graph = build_lane_graph(lanes, snap_tol=params.snap_tol, max_angle=params.max_angle_rad())
    prepare_lane_geometry(lanes)
    seg_tree, seg_records = build_segment_index(lanes)
    if seg_tree is None:
        return None

    return LaneResources(lanes=lanes, graph=graph, segment_tree=seg_tree, segment_records=seg_records)


def _sample_distance(speed: float, params: GoalGenerationParams, rng: np.random.Generator) -> float:
    # Ensure positive support even when the agent is slow/stopped.
    mean = params.distance_mean_coeff * speed
    std = params.distance_std_coeff * speed
    dist = rng.normal(loc=mean, scale=max(std, 1e-3))
    return float(np.clip(dist, params.min_distance, params.max_distance))


def sample_truncated_bivariate_normal(Sigma: np.ndarray, max_abs: float = 1.2,
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


def generate_goal_sets(
    agent_states,
    local_states,
    cont_agent_mask: torch.Tensor,
    env_idx: int,
    lane_resources: Optional[LaneResources],
    params: GoalGenerationParams,
    np_rng: np.random.Generator,
    py_rng: random.Random,
) -> List[Dict[int, Tuple[float, float]]]:
    if lane_resources is None or lane_resources.segment_tree is None:
        raise ValueError("Lane resources are not available for goal generation.")

    controllable = torch.where(cont_agent_mask[env_idx])[0]
    if controllable.numel() == 0:
        return []

    cov = params.goal_noise_cov()

    goal_sets: List[Dict[int, Tuple[float, float]]] = []
    seg_tree = lane_resources.segment_tree
    seg_records = lane_resources.segment_records

    per_agent_goals: Dict[int, List[Tuple[float, float]]] = {}

    for agent_idx in controllable.tolist():
        
        px = float(agent_states.pos_x[env_idx, agent_idx].item())
        py = float(agent_states.pos_y[env_idx, agent_idx].item())
        heading = float(agent_states.rotation_angle[env_idx, agent_idx].item())
        speed = float(local_states.speed[env_idx, agent_idx].item())
        speed = max(speed, params.minimum_speed_threshold)
        agent_id = int(agent_states.id[env_idx, agent_idx].item())
        per_agent_goals[agent_id] = []

        # Get all candidates once
        candidates = closest_point_on_network(
            np.array([px, py], dtype=float),
            lane_resources.lanes,
            seg_tree,
            seg_records,
            vehicle_heading=heading,
            k=params.projection_k,
            max_heading_mismatch=params.heading_mismatch_rad(),
            heading_penalty=params.heading_penalty,
        )
        
        best_candidate_backup = None
        if candidates:
             best_candidate_backup = min(candidates, key=lambda x: x[-1])

        for _ in range(params.num_sets):
            target = None
            
            # Retry loop to find a valid path (not a dead end)
            for _retry in range(params.projection_k // 2):
                if not candidates:
                    break

                # Sample a candidate
                sample_mode = params.projection_sample_mode
                softmax_temp = params.projection_softmax_temp
                
                if sample_mode not in ("softmax", "argmin"):
                    sample_mode = "argmin"
                
                if (sample_mode == "argmin") or len(candidates) == 1:
                    best_idx = 0
                    min_cost = candidates[0][-1]
                    for i in range(1, len(candidates)):
                        if candidates[i][-1] < min_cost:
                            min_cost = candidates[i][-1]
                            best_idx = i
                    choice_idx = best_idx
                else:
                    # softmax
                    temp = max(1e-6, float(softmax_temp))
                    costs = np.array([c[-1] for c in candidates], float)
                    w = np.exp(-costs / temp)
                    w_sum = w.sum()
                    if w_sum <= 0:
                         choice_idx = int(np_rng.integers(0, len(candidates)))
                    else:
                        w /= w_sum
                        choice_idx = int(np_rng.choice(len(candidates), p=w))

                lane_id, s, proj, cost = candidates[choice_idx]
                
                travel_dist = _sample_distance(speed, params, np_rng)
                adv = advance_along_graph(
                    G=lane_resources.graph,
                    lanes=lane_resources.lanes,
                    start_lane=lane_id,
                    s0=s,
                    distance=travel_dist,
                    prefer_straight=params.prefer_straight,
                    max_hops=params.max_hops,
                    rng_choice=py_rng,
                )
                
                if adv is not None:
                    target = np.array(adv["point"], dtype=float)
                    if np.linalg.norm(target - np.array([px, py])) < params.min_distance:
                        target = None
                        continue
                    break
                else:
                    # Dead end, remove this candidate forever
                    candidates.pop(choice_idx)
            
            # Fallback if no valid path found
            if target is None:
                if best_candidate_backup is not None:
                    target = best_candidate_backup[2]  # proj
                else:
                    target = np.array([px, py], dtype=float)

            if params.add_goal_noise:
                noise = sample_truncated_bivariate_normal(
                    Sigma=cov, 
                    max_abs=params.goal_noise_max_abs,
                    rng=np_rng
                )
                target += noise

            per_agent_goals[agent_id].append((float(target[0]), float(target[1])))

    for i in range(params.num_sets):
        goal_set = {aid: per_agent_goals[aid][i] for aid in per_agent_goals}
        goal_sets.append(goal_set)

    return per_agent_goals, goal_sets
