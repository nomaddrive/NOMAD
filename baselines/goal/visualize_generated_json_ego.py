#!/usr/bin/env python3
"""
Visualize generated scenario JSON files with Ego-centric observation view.
Highlights objects within the observation radius of a selected ego vehicle.

Example usage:
    python3 baselines/goal/visualize_generated_json_ego.py \
    data/scenario.json \
    --output viz_ego.png \
    --frame 0 \
    --ego 0 \
    --radius 50.0 \
    --mirror
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Default observation radius from config
DEFAULT_OBS_RADIUS = 50.0

def validate_json_format(data: Dict) -> List[str]:
    """
    Validate JSON has all required fields and correct format.
    Returns list of errors (empty if valid).
    """
    errors = []
    
    # Check top-level keys
    required_keys = ['objects', 'roads']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing top-level key: {key}")
    
    if 'objects' not in data:
        return errors
    
    return errors

def plot_map_ego(ax, roads, ego_pos: Tuple[float, float], obs_radius: float):
    """
    Plot map with different opacity for inside/outside observation radius.
    """
    if not roads:
        return

    x_inside, y_inside = [], []
    x_outside, y_outside = [], []
    
    ego_x, ego_y = ego_pos
    radius_sq = obs_radius**2
    
    for road in roads:
        for point in road.get('geometry', []):
            px, py = point.get('x', 0), point.get('y', 0)
            
            # Check distance squared
            dist_sq = (px - ego_x)**2 + (py - ego_y)**2
            
            if dist_sq <= radius_sq:
                x_inside.append(px)
                y_inside.append(py)
            else:
                x_outside.append(px)
                y_outside.append(py)
                
    # Plot outside points (low opacity)
    if x_outside:
        ax.scatter(x_outside, y_outside, c='lightgray', s=1, alpha=0.5, label='Road (Outside)')
        
    # Plot inside points (high opacity)
    if x_inside:
        ax.scatter(x_inside, y_inside, c='gray', s=2, alpha=0.8, label='Road (Inside)')

def get_rotated_corners(x, y, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    
    # Half dimensions
    l2 = length / 2
    w2 = width / 2
    
    # Corners relative to center (Front-Left, Rear-Left, Rear-Right, Front-Right)
    corners_x = np.array([l2, -l2, -l2, l2])
    corners_y = np.array([w2, w2, -w2, -w2])
    
    # Rotate
    rot_x = corners_x * c - corners_y * s
    rot_y = corners_x * s + corners_y * c
    
    # Translate
    final_x = rot_x + x
    final_y = rot_y + y
    
    return final_x, final_y

def visualize_json_scenario_ego(json_path: Path, output_path: Path = None, frame: int = 0, 
                              ego_idx: int = 0, obs_radius: float = DEFAULT_OBS_RADIUS, mirror: bool = False):
    """
    Load and visualize a scenario from JSON file with ego-centric view.
    """
    print(f"Loading: {json_path}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Validate format
    errors = validate_json_format(data)
    if errors:
        print("\n❌ JSON Validation Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nProceeding with visualization anyway...\n")
    
    # Extract data
    vehicles = data.get('objects', [])
    roads = data.get('roads', [])

    # Apply mirroring if requested
    if mirror:
        print("Applying mirror transformation (flipping X axis)...")
        # Mirror roads
        for road in roads:
            # geometry is a list of dicts with x, y
            for point in road.get('geometry', []):
                if 'x' in point:
                    point['x'] = -point['x']
        
        # Mirror vehicles
        for vehicle in vehicles:
            # Mirror position history
            if 'position' in vehicle:
                for pos in vehicle['position']:
                    if 'x' in pos:
                        pos['x'] = -pos['x']
            
            # Mirror goal position
            if 'goalPosition' in vehicle:
                goal = vehicle['goalPosition']
                if 'x' in goal:
                    goal['x'] = -goal['x']
            
            # Reflect heading across Y-axis: new_heading = pi - old_heading
            # Note: heading is a list of floats
            if 'heading' in vehicle:
                vehicle['heading'] = [np.pi - h for h in vehicle['heading']]
    
    if not vehicles:
        print("No vehicles found in scenario.")
        return

    # Find Ego Vehicle
    if ego_idx >= len(vehicles):
        print(f"Warning: Ego index {ego_idx} out of range (0-{len(vehicles)-1}). Using 0.")
        ego_idx = 0
    
    ego_vehicle = vehicles[ego_idx]
    
    # Get Ego Position at frame
    if frame < len(ego_vehicle.get('position', [])):
        ego_pos_data = ego_vehicle['position'][frame]
        ego_x, ego_y = ego_pos_data['x'], ego_pos_data['y']
    else:
        ego_x, ego_y = 0, 0
        
    print(f"Found {len(vehicles)} vehicles")
    print(f"Visualizing frame {frame}/90")
    print(f"Ego Vehicle: ID {ego_vehicle.get('id')} at ({ego_x:.1f}, {ego_y:.1f})")
    print(f"Observation Radius: {obs_radius}m")
    
    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 1. Plot Map (with opacity based on radius)
    plot_map_ego(ax, roads, (ego_x, ego_y), obs_radius)
    
    # 2. Plot Vehicles
    visible_vehicles = 0
    for i, vehicle in enumerate(vehicles):
        is_ego = (i == ego_idx)
        
        if is_ego:
            color = 'red'
        else:
            color = 'black'
        
        # Get position at specified frame
        if frame < len(vehicle.get('position', [])):
            pos = vehicle['position'][frame]
            pos_x, pos_y = pos['x'], pos['y']
        else:
            pos_x, pos_y = 0, 0
        
        # Check if position is valid
        if pos_x <= -9000 or pos_y <= -9000:
            continue
            
        # Calculate distance to ego
        dist_sq = (pos_x - ego_x)**2 + (pos_y - ego_y)**2
        is_visible = dist_sq <= obs_radius**2
        
        # Set opacity
        if is_ego:
            alpha = 1.0
            edgecolor = 'black' # Black edge for red ego vehicle
            linewidth = 2
        elif is_visible:
            alpha = 0.9
            edgecolor = 'none' # No edge for black vehicles
            linewidth = 1
        else:
            alpha = 0.15 # Faded for outside vehicles
            edgecolor = 'none'
            linewidth = 1
            
        visible_vehicles += 1

        # Get heading
        if frame < len(vehicle.get('heading', [])):
            heading = vehicle['heading'][frame]
        else:
            heading = 0
            
        if heading <= -9000:
            heading = 0

        # Dimensions (Double size as requested previously)
        length = vehicle.get('length', 4.5) * 2.0
        width = vehicle.get('width', 2.0) * 2.0
        
        # Draw Vehicle
        corners_x, corners_y = get_rotated_corners(pos_x, pos_y, heading, length, width)
        poly = patches.Polygon(np.column_stack([corners_x, corners_y]), 
                             closed=True, facecolor=color, edgecolor=edgecolor, 
                             alpha=alpha, linewidth=linewidth,
                             label=f'V{i}' if i < 10 else None)
        ax.add_patch(poly)
        
        # Goal Position (only if visible or ego)
        if is_visible or is_ego:
            goal = vehicle.get('goalPosition', {})
            goal_x, goal_y = goal.get('x', 0), goal.get('y', 0)
            
            # Draw Goal
            ax.plot(goal_x, goal_y, 'o', markersize=8, markerfacecolor='none', 
                   markeredgecolor=color, markeredgewidth=2, alpha=alpha)
            ax.plot(goal_x, goal_y, '.', markersize=3, color=color, alpha=alpha)
            
            # Connection line
            ax.plot([pos_x, goal_x], [pos_y, goal_y], '--', color=color, alpha=0.3 * alpha, linewidth=1)

    # 3. Draw Observation Radius Circle
    obs_circle = patches.Circle((ego_x, ego_y), obs_radius, 
                              fill=False, edgecolor='red', linestyle='--', linewidth=2, alpha=0.5,
                              label='Obs Radius')
    ax.add_patch(obs_circle)

    # Labels and Title
    ax.set_xlabel('X (m)', fontsize=20)
    ax.set_ylabel('Y (m)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    title = f'{json_path.stem}\nFrame {frame}/90, Ego V{ego_idx}'
    ax.set_title(title, fontsize=24)
    
    # Custom Legend
    legend_elements = [
        patches.Patch(facecolor='red', edgecolor='black', linewidth=2, alpha=1.0, label='Ego Vehicle'),
        patches.Patch(facecolor='black', edgecolor='none', alpha=0.9, label='Visible Vehicle'),
        patches.Patch(facecolor='black', edgecolor='none', alpha=0.15, label='Outside Vehicle'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Obs Radius ({obs_radius}m)'),
        Line2D([0], [0], marker='o', color='w', label='Goal',
               markerfacecolor='none', markeredgecolor='black', markersize=10, markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16)
    
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Center view on Ego Vehicle with some padding
    view_radius = obs_radius * 1.5
    ax.set_xlim(ego_x - view_radius, ego_x + view_radius)
    ax.set_ylim(ego_y - view_radius, ego_y + view_radius)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated scenario JSON with Ego-centric observation view"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for visualization image"
    )
    parser.add_argument(
        "--frame", "-f",
        type=int,
        default=0,
        help="Which frame to visualize (0-90, default: 0)"
    )
    parser.add_argument(
        "--ego", "-e",
        type=int,
        default=0,
        help="Index of the ego vehicle to center on (default: 0)"
    )
    parser.add_argument(
        "--radius", "-r",
        type=float,
        default=DEFAULT_OBS_RADIUS,
        help=f"Observation radius in meters (default: {DEFAULT_OBS_RADIUS})"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the scenario across the Y-axis"
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    
    if json_path.is_file():
        output_path = Path(args.output) if args.output else None
        visualize_json_scenario_ego(json_path, output_path, args.frame, args.ego, args.radius, args.mirror)
    elif json_path.is_dir():
        print("Directory processing not fully implemented for single-file focus. Please specify a JSON file.")
    else:
        print(f"Error: {json_path} not found")

if __name__ == "__main__":
    main()
