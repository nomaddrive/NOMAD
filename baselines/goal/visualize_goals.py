import os
from pathlib import Path
import argparse
from typing import List, Optional

import matplotlib.pyplot as plt

from shapely.geometry import LineString, Point
from shapely.ops import unary_union, polygonize, snap
from shapely.ops import polygonize_full
from shapely.strtree import STRtree
from shapely.validation import make_valid
import numpy as np

import torch
import madrona_gpudrive

from gpudrive.env.config import EnvConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.datatypes.observation import GlobalEgoState
from gpudrive.datatypes.roadgraph import GlobalRoadGraphPoints

# Import our goal prediction models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from baselines.goal.model import GoalPredictor, GoalPredictorMDN


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


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt.get("args", {})
    model_type = args.get("model", "det")

    model_kwargs = dict(
        road_feat_dim=args.get("road_feat_dim", 27),
        d_model=args.get("d_model", 128),
        nhead=args.get("nhead", 4),
        enc_layers=args.get("enc_layers", 2),
        dec_layers=args.get("dec_layers", 2),
        dropout=args.get("dropout", 0.1),
    )

    if model_type == "mdn":
        model = GoalPredictorMDN(num_components=args.get("num_components", 5), **model_kwargs)
    else:
        model = GoalPredictor(**model_kwargs)

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    norm = ckpt.get("norm", {})
    mean_state = norm.get("mean_state")
    std_state = norm.get("std_state")
    mean_goal = norm.get("mean_goal")
    std_goal = norm.get("std_goal")
    if mean_state is None or std_state is None or mean_goal is None or std_goal is None:
        raise RuntimeError("Normalization statistics not found in checkpoint under 'norm'.")
    # Move to device
    mean_state = mean_state.to(device)
    std_state = std_state.to(device)
    mean_goal = mean_goal.to(device)
    std_goal = std_goal.to(device)

    return model, model_type, (mean_state, std_state, mean_goal, std_goal)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


@torch.no_grad()
def sample_goals_from_mdn(
    mu: torch.Tensor,            # (B,A,K,2)
    log_sigma: torch.Tensor,     # (B,A,K,2)
    rho: torch.Tensor,           # (B,A,K)
    logits: torch.Tensor,        # (B,A,K)
    mean_goal: torch.Tensor,     # (A,2)
    std_goal: torch.Tensor,      # (A,2)
    n_samples: int = 20,
) -> torch.Tensor:
    """Sample multiple goal points per agent from an MDN and denormalize.

    Returns: samples (B, A, S, 2) in original coordinate space.
    """
    B, A, K, _ = mu.shape
    device = mu.device

    probs = torch.softmax(logits, dim=-1)  # (B,A,K)
    comp_idx = torch.distributions.Categorical(probs=probs).sample((n_samples,)).permute(1, 2, 0)  # (B,A,S)

    S = n_samples
    idx_k = comp_idx.unsqueeze(-1).unsqueeze(-1)  # (B,A,S,1,1)

    mu_exp = mu.unsqueeze(2).expand(B, A, S, K, 2)
    log_sigma_exp = log_sigma.unsqueeze(2).expand(B, A, S, K, 2)
    rho_exp = rho.unsqueeze(2).expand(B, A, S, K)

    mu_sel = torch.gather(mu_exp, 3, idx_k.expand(B, A, S, 1, 2)).squeeze(3)  # (B,A,S,2)
    log_sigma_sel = torch.gather(log_sigma_exp, 3, idx_k.expand(B, A, S, 1, 2)).squeeze(3)  # (B,A,S,2)
    rho_sel = torch.gather(rho_exp, 3, comp_idx.unsqueeze(-1)).squeeze(-1)  # (B,A,S)

    sigma_sel = torch.exp(log_sigma_sel)  # (B,A,S,2)

    z1 = torch.randn(B, A, S, device=device)
    z2 = torch.randn(B, A, S, device=device)
    x = mu_sel[..., 0] + sigma_sel[..., 0] * z1
    eps = 1e-6
    y = mu_sel[..., 1] + sigma_sel[..., 1] * (
        rho_sel * z1 + torch.sqrt(torch.clamp(1 - rho_sel ** 2, min=eps)) * z2
    )
    samples_n = torch.stack([x, y], dim=-1)  # (B,A,S,2)

    # Denormalize to original space
    mg = mean_goal.view(1, A, 1, 2).to(device)
    sg = std_goal.view(1, A, 1, 2).to(device)
    samples = samples_n * sg + mg
    return samples


def get_endpoints(x, y, length, yaw):
    """Compute the start and end points of a road segment."""
    center = np.array([x, y])
    start = center - np.array([length * np.cos(yaw), length * np.sin(yaw)])
    end = center + np.array([length * np.cos(yaw), length * np.sin(yaw)])
    return start, end


def drivable_from_band(band, edge_lines, pad=30.0, keep_within=6.0, min_area=4.0):
    """
    band: Polygon/MultiPolygon returned by curb_band_from_edges
    pad:  how far the outer frame extends beyond the map bounds
    keep_within: distance (buffer) from the network to keep interiors; removes 'outside-the-map' area
    min_area: drop tiny polygons
    """
    G = unary_union(edge_lines)
    frame = box(*band.bounds).buffer(pad)
    candidates = frame.difference(band)                  # outside + road interiors
    near_net   = candidates.intersection(G.buffer(keep_within))
    # keep only reasonably sized pieces
    geoms = list(getattr(near_net, "geoms", [near_net]))
    geoms = [g for g in geoms if g.area >= min_area]
    from shapely.ops import unary_union as uu
    return uu(geoms) if geoms else near_net


def build_drivable_area(env_idx: int, road_graph: GlobalRoadGraphPoints, snap_tol: float=0.05):
    """Build a 2D polygon of the drivable area from road graph points for a given env index.

    Returns: shapely.geometry.Polygon object representing the drivable area.
    """

    # Create mask for road edges
    road_mask = road_graph.type[env_idx, :] == int(madrona_gpudrive.EntityType.RoadEdge)

    # Get coordinates and metadata for the current road type
    x_coords = road_graph.x[env_idx, road_mask].tolist()
    y_coords = road_graph.y[env_idx, road_mask].tolist()
    segment_lengths = road_graph.segment_length[
        env_idx, road_mask
    ].tolist()
    segment_orientations = road_graph.orientation[
        env_idx, road_mask
    ].tolist()

    edge_lines = []

    for x, y, L, theta in zip(x_coords, y_coords, segment_lengths, segment_orientations):
        a, b = get_endpoints(x, y, L, theta)
        edge_lines.append(LineString([a, b]))

    merged = unary_union(edge_lines)
    closed = snap(merged, merged, snap_tol)
    closed = merged.buffer(distance=0.1, cap_style=2, join_style=2).buffer(distance=--0.1, cap_style=2, join_style=2)
    try:
        closed = make_valid(closed)
    except Exception as e:
        print(f"Error making closed geometry valid: {e}")
    # keep as MultiPolygon/Polygon; drop tiny bits
    faces = list(getattr(closed, "geoms", [closed]))
    faces = [p for p in faces if p.area >= 0.1]

    polys, dangles, cuts, invalid = polygonize_full(closed)

    fig, ax = plt.subplots(figsize=(8,10))

    # base linework (light gray)
    for ls in _iter_lines(merged):
        x, y = ls.xy
        ax.plot(x, y, lw=0.8, c="#BBBBBB", zorder=1)

    # polygons (green outlines)
    for p in _iter_polys(polys):
        x, y = p.exterior.xy
        ax.plot(x, y, lw=1.6, c="green", zorder=2)

    # dangles (red) + endpoints
    dx, dy = [], []
    for ls in _iter_lines(dangles):
        x, y = ls.xy
        ax.plot(x, y, lw=2.0, c="red", zorder=3)
        dx += [ls.coords[0][0], ls.coords[-1][0]]
        dy += [ls.coords[0][1], ls.coords[-1][1]]
    if dx:
        ax.scatter(dx, dy, s=12, c="red", label="dangle endpoints", zorder=4)

    # cuts (orange dashed)
    for ls in _iter_lines(cuts):
        x, y = ls.xy
        ax.plot(x, y, lw=2.0, c="orange", ls="-", zorder=3)

    # invalid rings (magenta)
    for ls in _iter_lines(invalid):
        x, y = ls.xy
        ax.plot(x, y, lw=2.0, c="magenta", zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"polygonize_full → polys={len(list(_iter_polys(polys)))}, "
        f"dangles={sum(1 for _ in _iter_lines(dangles))}, "
        f"cuts={sum(1 for _ in _iter_lines(cuts))}, "
        f"invalid={sum(1 for _ in _iter_lines(invalid))}"
    )
    # build a simple legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="#BBBBBB", lw=1, label="base lines"),
        Line2D([], [], color="green", lw=2, label="polygons"),
        Line2D([], [], color="red", lw=2, label="dangles"),
        Line2D([], [], color="orange", lw=2, ls="--", label="cuts"),
        Line2D([], [], color="magenta", lw=2, label="invalid rings"),
    ]
    ax.legend(handles=handles, loc="best")
    plt.savefig(f"drivable_area_env_{env_idx}.png")
    return unary_union(faces)


def is_point_drivable(x, y, drivable_area) -> bool:
    pt = Point(x, y)
    return drivable_area.contains(pt)


def get_inputs_from_env(env: GPUDriveTorchEnv, device: torch.device):
    # Read tensors directly from the underlying simulator via the visualizer handle
    sim = env.vis.sim_object

    # Global agent state: build (B, A, 3) = (x, y, yaw)
    ges = GlobalEgoState.from_tensor(
        sim.absolute_self_observation_tensor(), backend="torch", device="cpu"
    )
    agent_ids = ges.id.to(device)
    # These are CPU torch tensors; stack into (B,A,3) and move to device
    agent_states = torch.stack([ges.pos_x, ges.pos_y, ges.rotation_angle], dim=-1).to(device)
    # Find mask for valid agents
    cont_agent_mask = env.cont_agent_mask.to(device)

    raw_types = env.sim.info_tensor().to_torch().clone().to(env.device)[:, :, 4].long()
    valid_mask = raw_types != int(madrona_gpudrive.EntityType._None)

    # Roadgraph features: match collect_dataset.py pipeline exactly
    # - one_hot types
    # - normalize() roadgraph (demean/scale as defined in datatype)
    # - concatenate features into 27-D vector per point
    grp = GlobalRoadGraphPoints.from_tensor(
        roadgraph_tensor=sim.map_observation_tensor(), backend="torch", device="cpu"
    )
    roadgraph_mask = grp.segment_height > 0.
    roadgraph_mask = roadgraph_mask.to(device)
    grp.one_hot_encode_road_point_types()
    grp.normalize()
    roadgraph = torch.cat(
        [
            grp.x.unsqueeze(-1),
            grp.y.unsqueeze(-1),
            grp.segment_length.unsqueeze(-1),
            grp.segment_width.unsqueeze(-1),
            grp.segment_height.unsqueeze(-1),
            grp.orientation.unsqueeze(-1),
            grp.type,
        ],
        dim=-1,
    ).to(device)

    return agent_states, valid_mask, cont_agent_mask, roadgraph, roadgraph_mask, agent_ids


def plot_road_edge_segments(
    env_idx: int,
    road_graph: GlobalRoadGraphPoints,
    snap_tol: float = 0.0,
    out_path: Optional[Path] = None,
    show: bool = False,
):
    """Generate a debug plot of road edge segments for a given environment.

    - Left subplot: raw edge segments with endpoints
    - Right subplot: snapped/merged lines and any polygonized faces (outlines)
    """
    # Local import to avoid global matplotlib dependency when not debugging

    # Create mask for road edges
    road_mask = road_graph.type[env_idx, :] == int(madrona_gpudrive.EntityType.RoadEdge)

    # Get coordinates and metadata for the current road type
    x_coords = road_graph.x[env_idx, road_mask].tolist()
    y_coords = road_graph.y[env_idx, road_mask].tolist()
    segment_lengths = road_graph.segment_length[env_idx, road_mask].tolist()
    segment_orientations = road_graph.orientation[env_idx, road_mask].tolist()

    edge_lines = []
    endpoints = []
    for x, y, L, theta in zip(x_coords, y_coords, segment_lengths, segment_orientations):
        a, b = get_endpoints(x, y, L, theta)
        edge_lines.append(LineString([a, b]))
        endpoints.append(a)
        endpoints.append(b)

    import pdb; pdb.set_trace()

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax0, ax1, ax2 = axes

    # Left: raw edges
    for ln in edge_lines:
        xs, ys = ln.xy
        ax0.plot(xs, ys, color="#1f77b4", alpha=0.8, linewidth=0.1)
    if endpoints:
        ex = [p[0] for p in endpoints]
        ey = [p[1] for p in endpoints]
        ax0.scatter(ex, ey, s=6, c="#d62728", alpha=0.6, label="endpoints")
    ax0.set_title(f"Env {env_idx} - Raw edge segments")
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.2)

    # Right: snapped/merged and polygonized outlines
    if len(edge_lines) > 0:
        merged = unary_union(edge_lines)
        merged = snap(merged, merged, snap_tol)

        plot_polygonize_diagnostics(merged, ax2)

        # Plot merged lines (could be LineString or MultiLineString)
        def _plot_lines(geom, ax, color="#2ca02c"):
            gtype = getattr(geom, "geom_type", "")
            if gtype == "LineString":
                xs, ys = geom.xy
                ax.plot(xs, ys, color=color, alpha=0.9, linewidth=0.1)
            elif gtype == "MultiLineString":
                for ln in geom.geoms:
                    xs, ys = ln.xy
                    ax.plot(xs, ys, color=color, alpha=0.9, linewidth=0.1)

        _plot_lines(merged, ax1)

        polys = list(polygonize(merged))
        plotted_poly_label = False
        for poly in polys:
            x, y = poly.exterior.xy
            ax1.plot(
                x,
                y,
                color="#ff7f0e",
                linewidth=0.1,
                label=("polygon" if not plotted_poly_label else None),
            )
            plotted_poly_label = True
    ax1.set_title(f"Env {env_idx} - Snapped + faces (tol={snap_tol})")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.2)
    # Build legend for right subplot if polygons plotted
    handles, labels = ax1.get_legend_handles_labels()
    if handles and labels:
        ax1.legend(loc="upper right")

    # Tight limits
    for ax in axes:
        ax.autoscale(enable=True, tight=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ------- helpers -------
def _iter_lines(g):
    """Yield LineString/LinearRing pieces from any Shapely geometry/collection."""
    if g is None or g.is_empty:
        return
    gt = g.geom_type
    if gt in ("LineString", "LinearRing"):
        yield g
    elif gt in ("MultiLineString", "GeometryCollection"):
        for s in g.geoms:
            yield from _iter_lines(s)
    elif gt == "Polygon":
        yield g.exterior
        for hole in g.interiors:
            yield LineString(hole)
    elif gt == "MultiPolygon":
        for p in g.geoms:
            yield from _iter_lines(p)

def _iter_polys(g):
    if g is None or g.is_empty:
        return
    if g.geom_type == "Polygon":
        yield g
    elif g.geom_type in ("MultiPolygon", "GeometryCollection"):
        for p in g.geoms:
            yield from _iter_polys(p)

# ------- plotting -------
def plot_polygonize_diagnostics(merged_lines, ax=None):
    """
    merged_lines: the noded linework you pass to polygonize_full
                  (e.g., unary_union(lines) after any snapping)
    """
    polys, dangles, cuts, invalid = polygonize_full(merged_lines)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,10))

    # base linework (light gray)
    for ls in _iter_lines(merged_lines):
        x, y = ls.xy
        ax.plot(x, y, lw=0.8, c="#BBBBBB", zorder=1)

    # polygons (green outlines)
    for p in _iter_polys(polys):
        x, y = p.exterior.xy
        ax.plot(x, y, lw=1.6, c="green", zorder=2)

    # dangles (red) + endpoints
    dx, dy = [], []
    for ls in _iter_lines(dangles):
        x, y = ls.xy
        ax.plot(x, y, lw=2.0, c="red", zorder=3)
        dx += [ls.coords[0][0], ls.coords[-1][0]]
        dy += [ls.coords[0][1], ls.coords[-1][1]]
    if dx:
        ax.scatter(dx, dy, s=12, c="red", label="dangle endpoints", zorder=4)

    # cuts (orange dashed)
    for ls in _iter_lines(cuts):
        x, y = ls.xy
        ax.plot(x, y, lw=2.0, c="orange", ls="-", zorder=3)

    # invalid rings (magenta)
    for ls in _iter_lines(invalid):
        x, y = ls.xy
        ax.plot(x, y, lw=2.0, c="magenta", zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"polygonize_full → polys={len(list(_iter_polys(polys)))}, "
        f"dangles={sum(1 for _ in _iter_lines(dangles))}, "
        f"cuts={sum(1 for _ in _iter_lines(cuts))}, "
        f"invalid={sum(1 for _ in _iter_lines(invalid))}"
    )
    # build a simple legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="#BBBBBB", lw=1, label="base lines"),
        Line2D([], [], color="green", lw=2, label="polygons"),
        Line2D([], [], color="red", lw=2, label="dangles"),
        Line2D([], [], color="orange", lw=2, ls="--", label="cuts"),
        Line2D([], [], color="magenta", lw=2, label="invalid rings"),
    ]
    ax.legend(handles=handles, loc="best")

    return polys, dangles, cuts, invalid, ax


def overlay_predicted_goals_on_fig(fig, pred_xy: torch.Tensor, zoom_radius: int = 100, color: str = "magenta"):
    # pred_xy: (N, 2) tensor on any device
    import matplotlib.pyplot as plt

    ax = fig.axes[0] if len(fig.axes) > 0 else plt.gca()
    px = pred_xy[:, 0].detach().cpu().numpy()
    py = pred_xy[:, 1].detach().cpu().numpy()

    ax.scatter(px, py, s=20, c=color, marker="x", label="Pred goal")
    # Optionally, draw small circles around each predicted goal
    # for x, y in zip(px, py):
    #     circle = plt.Circle((x, y), radius=zoom_radius * 0.02, color=color, fill=False, linestyle=":")
    #     ax.add_patch(circle)

    # Add legend if not present
    handles, labels = ax.get_legend_handles_labels()
    if "Pred goal" not in labels:
        ax.legend(loc="upper right")


def overlay_multiagent_samples(
    fig,
    samples_env: torch.Tensor,       # (A,S,2)
    agent_positions: torch.Tensor,   # (A,2)
    agent_mask: torch.Tensor,        # (A,)
    agent_id: torch.Tensor,          # (A,)
    drivable_area,
    zoom_radius: int = 100,
    label_agents: bool = True,
    draw_lines: bool = False,
    alpha: float = 0.6,
):
    """Overlay multiple sampled goals per agent with clear agent association.

    - Uses per-agent colors (tab20 cycling), optional labels at current agent position,
      and optional faint lines connecting agent position to each of its samples.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ax = fig.axes[0] if len(fig.axes) > 0 else plt.gca()

    # Build color cycle per agent
    cmap = plt.get_cmap("tab20")
    A = samples_env.shape[0]
    colors = [cmap(i % 20) for i in range(A)]

    valid_idx = torch.where(agent_mask)[0].tolist()
    valid_agent_id = agent_id[agent_mask].tolist()

    # Plot per agent
    for i in valid_idx:
        c = colors[i]
        pts = samples_env[i].detach().cpu().numpy()  # (S,2)
        # inroad_pts = []
        # for (x, y) in pts:
        #     if is_point_drivable(x, y, drivable_area):
        #         inroad_pts.append((x, y))
        # if len(inroad_pts) == 0:
        #     raise ValueError("No valid in-road points found.")
        # pts = np.array(inroad_pts)

        ax.scatter(pts[:, 0], pts[:, 1], s=12, c=[c], alpha=alpha, marker="x", label=None)

        # # Label agent near its current position
        pos = agent_positions[i].detach().cpu().numpy()
        # if label_agents:
        #     ax.text(pos[0], pos[1], f"a{i}", color=c, fontsize=8, ha="center", va="center",
        #             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.5))

        if draw_lines:
            # Thin lines from agent pos to each sample (can be heavy if many samples)
            for p in pts:
                ax.plot([pos[0], p[0]], [pos[1], p[1]], color=c, alpha=alpha * 0.4, linewidth=0.5)

    # Add a compact legend with a subset of agents to avoid clutter
    # Show up to 10 agent color entries
    handles = []
    labels = []
    for j in range(len(valid_agent_id)):
        handles.append(plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=colors[valid_idx[j]], markeredgecolor=colors[valid_idx[j]],
                                    markersize=6, linestyle=''))
        labels.append(int(valid_agent_id[j]))
    ax.legend(handles, labels, title="Agents", loc="upper right", framealpha=0.8)


@torch.no_grad()
def main():
    pa = argparse.ArgumentParser(description="Visualize predicted goal points over simulator scenes")
    pa.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    pa.add_argument("--data_path", type=str, default="data/nuplan/singapore_valid_train", help="Dataset root for scenes")
    pa.add_argument("--num_envs", type=int, default=50, help="Number of scenes/worlds to visualize")
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
    pa.add_argument("--num_samples", type=int, default=0, help="If >0 and model is MDN, sample this many goals per agent and overlay")
    pa.add_argument("--label_agents", action="store_true", help="Label agent indices near their current positions")
    pa.add_argument("--draw_lines", action="store_true", help="Draw faint lines from agent to each sampled goal (may clutter)")
    pa.add_argument("--sample_alpha", type=float, default=0.6, help="Alpha (transparency) for sampled goals")
    pa.add_argument("--snap_tol", type=float, default=0.0, help="Snap tolerance for building drivable area polygons")

    args = pa.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) Create environment
    env = build_env(
        data_path=args.data_path,
        num_envs=args.num_envs,
        dynamics_model=args.dynamics_model,
        action_type=args.action_type,
        controllable_agent_selection=args.controllable_agent_selection,
        mirror_x=args.mirror_x,
        max_num_objects=args.max_num_objects,
        device="cuda" if device.type == "cuda" else "cpu",
    )

    global_roadgraph = GlobalRoadGraphPoints.from_tensor(
        roadgraph_tensor=env.sim.map_observation_tensor(),
        backend=env.backend,
        device=env.device,
    )

    control_mask = env.cont_agent_mask

    # 2) Load model and normalization
    model, model_type, (mean_state, std_state, mean_goal, std_goal) = load_model(args.checkpoint, device)

    # 3) Reset env and collect inputs
    env.reset(control_mask)
    env_indices: List[int] = list(range(args.num_envs))
    time_steps: List[int] = [0] * args.num_envs
    figs = env.vis.plot_simulator_state(
        env_indices=env_indices,
        time_steps=time_steps,
        zoom_radius=args.zoom_radius,
        plot_log_replay_trajectory=False,
    )
    agent_states, valid_mask, cont_agent_mask, roadgraph, roadgraph_mask, agent_ids = get_inputs_from_env(env, device)
    road_mask = roadgraph_mask
    agent_mask = valid_mask

    # 4) Run inference
    agent_states_n = normalize(agent_states, mean_state, std_state)
    if model_type == "mdn":
        mu, log_sigma, rho, logits = model(agent_states_n, roadgraph, agent_mask=agent_mask, road_mask=road_mask)
        if args.num_samples and args.num_samples > 0:
            # Multi-sample per agent
            samples = sample_goals_from_mdn(mu, log_sigma, rho, logits, mean_goal, std_goal, n_samples=args.num_samples)
            pred = None  # Not used in multi-sample path
        else:
            # Take MAP component
            k = torch.argmax(logits, dim=-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            mu_sel = torch.gather(mu, 2, k).squeeze(2)
            pred = denormalize(mu_sel, mean_goal, std_goal)
            samples = None
    else:
        pred_n = model(agent_states_n, roadgraph, agent_mask=agent_mask, road_mask=road_mask)
        pred = denormalize(pred_n, mean_goal, std_goal)
        samples = None

    # 5) Visualize base scene and overlay predictions
    out_dir = Path(args.out_dir)
    out_dir = out_dir / model_type / args.checkpoint.split("/")[-2] / args.data_path.split("/")[-1]  # Use parent folder name of checkpoint
    os.makedirs(out_dir, exist_ok=True)

    # 6) Overlay predicted goals per env and save
    filenames = env.get_env_filenames() if hasattr(env, "get_env_filenames") else [f"env_{i}" for i in env_indices]

    for i, env_idx in enumerate(env_indices):
        try:
            drivable_area = build_drivable_area(env_idx, global_roadgraph, snap_tol=args.snap_tol)
        except RuntimeError as e:
            # Produce a debug plot to help diagnose missing closed faces
            debug_dir = Path(args.out_dir) / "debug_failed_polygonization"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_plot_path = debug_dir / f"env_{env_idx}_road_edges.png"
            print(f"[WARN] {e} -- writing debug plot to {debug_plot_path}")
            plot_road_edge_segments(env_idx, global_roadgraph, out_path=debug_plot_path, show=False, snap_tol=args.snap_tol)
            # Skip this env for now
            continue
        valid = cont_agent_mask[env_idx]
        agent_id = agent_ids[env_idx]
        if model_type == "mdn" and samples is not None:
            # Gather positions and samples for this env
            # Current positions from env: use agent_states (x,y,yaw) -> (x,y)
            agent_pos_xy = agent_states[env_idx, :, :2]
            samples_env = samples[env_idx]  # (A,S,2)
            overlay_multiagent_samples(
                figs[i], samples_env, agent_pos_xy, valid, agent_id, 
                drivable_area=None,
                zoom_radius=args.zoom_radius,
                label_agents=args.label_agents,
                draw_lines=args.draw_lines,
                alpha=args.sample_alpha,
            )
        else:
            pred_xy = pred[env_idx][valid]
            overlay_predicted_goals_on_fig(figs[i], pred_xy)

        out_path = Path(out_dir) / (filenames[i][:-5] if isinstance(filenames[i], str) and filenames[i].endswith(".json") else str(filenames[i]))
        out_path = out_path.with_suffix(".png")
        figs[i].savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
