#!/usr/bin/env python3
"""
Visualize generated scenario JSON files.
Generates two images: one with vehicles/goals, one with map only.

Example usage:
    SINGLE FILE:
    python3 baselines/goal/visualize_generated_json_dual.py \
    data/scenario.json \
    --output viz.png \
    --frame 0 \
    --highlight-dynamic \
    --zoom 1.0 \
    --mirror
    
    ALL FILES IN DIRECTORY:
    python3 baselines/goal/visualize_generated_json_dual.py \
    data/my_scenarios_folder \
    --all \
    --output viz_base_name \
    --frame 0 \
    --highlight-dynamic \
    --zoom 1.0 \
    --mirror
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from typing import Dict, List


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
    
    # Validate each vehicle object
    for i, obj in enumerate(data['objects']):
        vehicle_errors = []
        
        # Required fields
        required_fields = [
            'id', 'type', 'is_sdc', 'mark_as_expert', 'valid',
            'position', 'heading', 'velocity', 'goalPosition',
            'length', 'width', 'height', 'total_distance_traveled'
        ]
        
        for field in required_fields:
            if field not in obj:
                vehicle_errors.append(f"Missing field '{field}'")
        
        # Check array lengths (should be 91 frames)
        if 'position' in obj and len(obj['position']) != 91:
            vehicle_errors.append(f"position has {len(obj['position'])} frames (expected 91)")
        
        if 'heading' in obj and len(obj['heading']) != 91:
            vehicle_errors.append(f"heading has {len(obj['heading'])} frames (expected 91)")
        
        if 'velocity' in obj and len(obj['velocity']) != 91:
            vehicle_errors.append(f"velocity has {len(obj['velocity'])} frames (expected 91)")
        
        if 'valid' in obj and len(obj['valid']) != 91:
            vehicle_errors.append(f"valid has {len(obj['valid'])} values (expected 91)")
        
        # Check velocity format (should be dict with x, y)
        if 'velocity' in obj and len(obj['velocity']) > 0:
            v = obj['velocity'][0]
            if not isinstance(v, dict) or 'x' not in v or 'y' not in v:
                vehicle_errors.append(f"velocity should be dict with 'x' and 'y' keys")
        
        # Check total_distance_traveled is float/number
        if 'total_distance_traveled' in obj and not isinstance(obj['total_distance_traveled'], (int, float)):
            vehicle_errors.append(f"total_distance_traveled should be a number")
        
        if vehicle_errors:
            errors.append(f"Vehicle {i} (ID {obj.get('id', 'unknown')}): " + ", ".join(vehicle_errors))
    
    return errors

def plot_map(ax, roads):
    if roads:
        x_pts = []
        y_pts = []
        for road in roads:
            for point in road.get('geometry', []):
                x_pts.append(point.get('x', 0))
                y_pts.append(point.get('y', 0))
        if x_pts:
            ax.scatter(x_pts, y_pts, c='lightgray', s=1, alpha=0.3, label='Road network')

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

def visualize_json_scenario(json_path: Path, output_path_base: Path = None, frame: int = 0, zoom: float = 1.0, mirror: bool = False, highlight_dynamic: bool = False, hide_static: bool = False):
    """
    Load and visualize a scenario from JSON file.
    Generates two plots: Map only, and Map + Vehicles.
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
    else:
        print("✅ JSON format valid - all required fields present\n")
    
    # Extract data
    vehicles = data.get('objects', [])
    roads = data.get('roads', [])
    
    # Extract tracks to predict indices if available
    tracks_to_predict_indices = set()
    if 'tracks_to_predict' in data:
        for t in data['tracks_to_predict']:
            if 'track_index' in t:
                tracks_to_predict_indices.add(t['track_index'])

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
    
    print(f"Found {len(vehicles)} vehicles")
    print(f"Visualizing frame {frame}/90")
    
    # --- Image 1: Map Only ---
    fig1, ax1 = plt.subplots(figsize=(16, 12))
    plot_map(ax1, roads)
    ax1.set_title(f'{json_path.stem} - Map Only', fontsize=20)
    ax1.axis('equal')
    ax1.set_xlabel('X (m)', fontsize=16)
    ax1.set_ylabel('Y (m)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True, alpha=0.3)

    # --- Image 2: With Vehicles ---
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    plot_map(ax2, roads)
    
    visible_vehicles = 0
    for i, vehicle in enumerate(vehicles):
        color = f'C{i % 10}'
        
        if highlight_dynamic or hide_static:
            is_dynamic = False
            
            # Criterion 1: Is it a track to predict?
            # Note: We assume vehicle index 'i' corresponds to 'track_index'
            if i in tracks_to_predict_indices:
                is_dynamic = True
            else:
                # Criterion 2: Distance between Start and Goal >= 4.0m (consts::staticThreshold)
                positions = vehicle.get('position', [])
                goal = vehicle.get('goalPosition')
                
                if positions and goal:
                    # Find first valid position
                    start_pos = None
                    valid_mask = vehicle.get('valid', [True] * len(positions))
                    for j, pos in enumerate(positions):
                        if j < len(valid_mask) and valid_mask[j]:
                            start_pos = pos
                            break
                    
                    if start_pos:
                        sx, sy = start_pos.get('x', 0), start_pos.get('y', 0)
                        gx, gy = goal.get('x', 0), goal.get('y', 0)
                        
                        # Check for valid coordinates (ignore padding values like -1e4)
                        if sx > -9000 and sy > -9000 and gx > -9000 and gy > -9000:
                            dist = np.sqrt((gx - sx)**2 + (gy - sy)**2)
                            if dist >= 4.0:
                                is_dynamic = True

            if hide_static and not is_dynamic:
                continue

            if is_dynamic:
                zorder = 5 # Dynamic on top
            else:
                color = 'lightgray'
                zorder = 3 # Static below
        else:
            zorder = 4
        
        # Get position at specified frame
        if frame < len(vehicle.get('position', [])):
            pos = vehicle['position'][frame]
            pos_x, pos_y = pos['x'], pos['y']
        else:
            pos_x, pos_y = 0, 0
        
        # Check if position is valid (ignore -10000.0 values)
        if pos_x <= -9000 or pos_y <= -9000:
            continue
            
        visible_vehicles += 1

        # Get heading at specified frame
        if frame < len(vehicle.get('heading', [])):
            heading = vehicle['heading'][frame]
        else:
            heading = 0
            
        if heading <= -9000:
            heading = 0

        # Dimensions
        length = vehicle.get('length', 4.5) * 2.0
        width = vehicle.get('width', 2.0) * 2.0
        
        # Draw Vehicle as Rotated Rectangle
        corners_x, corners_y = get_rotated_corners(pos_x, pos_y, heading, length, width)
        poly = patches.Polygon(np.column_stack([corners_x, corners_y]), 
                             closed=True, facecolor=color, edgecolor='black', alpha=0.7, 
                             label=f'V{i}' if i < 10 else None, zorder=zorder)
        ax2.add_patch(poly)
        
        # Goal Position
        goal = vehicle.get('goalPosition', {})
        goal_x, goal_y = goal.get('x', 0), goal.get('y', 0)
        
        # Draw Goal: Small circle with dot inside
        # Circle (hollow)
        ax2.plot(goal_x, goal_y, 'o', markersize=8, markerfacecolor='none', markeredgecolor=color, markeredgewidth=2, zorder=zorder)
        # Dot (filled)
        ax2.plot(goal_x, goal_y, '.', markersize=3, color=color, zorder=zorder)
        
        # Connection line
        ax2.plot([pos_x, goal_x], [pos_y, goal_y], '--', color=color, alpha=0.3, linewidth=1, zorder=zorder-1)
        
    
    ax2.set_xlabel('X (m)', fontsize=20)
    ax2.set_ylabel('Y (m)', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    title = f'{json_path.stem}\nFrame {frame}/90, {len(vehicles)} vehicles'
    if errors:
        title += '\n⚠️ JSON has validation errors (see console)'
    ax2.set_title(title, fontsize=24)
    
    # Custom Legend
    if highlight_dynamic:
        legend_elements = [
            patches.Patch(facecolor='C0', edgecolor='black', alpha=0.7, label='Dynamic Vehicle'),
            patches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.7, label='Static Vehicle'),
            Line2D([0], [0], marker='o', color='w', label='Goal',
                   markerfacecolor='none', markeredgecolor='gray', markersize=10, markeredgewidth=2)
        ]
    else:
        legend_elements = [
            patches.Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Vehicle'),
            Line2D([0], [0], marker='o', color='w', label='Goal',
                   markerfacecolor='none', markeredgecolor='gray', markersize=10, markeredgewidth=2)
        ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=16)
    
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Apply zoom if > 1.0
    if zoom > 1.0:
        for ax in [ax1, ax2]:
            # Get current limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Calculate center
            cx = (xlim[0] + xlim[1]) / 2
            cy = (ylim[0] + ylim[1]) / 2
            
            # Calculate new range
            rx = (xlim[1] - xlim[0]) / 2 / zoom
            ry = (ylim[1] - ylim[0]) / 2 / zoom
            
            # Set new limits
            ax.set_xlim(cx - rx, cx + rx)
            ax.set_ylim(cy - ry, cy + ry)

    plt.tight_layout()
    
    if output_path_base:
        # Construct filenames
        parent = output_path_base.parent
        stem = output_path_base.stem
        suffix = output_path_base.suffix
        
        out_map = parent / f"{stem}_map{suffix}"
        out_veh = parent / f"{stem}_vehicles{suffix}"
        
        fig1.savefig(out_map, dpi=150, bbox_inches='tight')
        print(f"Saved map visualization to: {out_map}")
        
        fig2.savefig(out_veh, dpi=150, bbox_inches='tight')
        print(f"Saved vehicle visualization to: {out_veh}")
        
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated scenario JSON files (Map and Vehicles)"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path base for visualization images (e.g. 'viz.png' -> 'viz_map.png', 'viz_vehicles.png')"
    )
    parser.add_argument(
        "--frame", "-f",
        type=int,
        default=0,
        help="Which frame to visualize (0-90, default: 0)"
    )
    parser.add_argument(
        "--zoom", "-z",
        type=float,
        default=1.0,
        help="Zoom level (default: 1.0)"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the scenario across the Y-axis"
    )
    parser.add_argument(
        "--highlight-dynamic",
        action="store_true",
        help="Color static vehicles gray and dynamic vehicles randomly"
    )
    parser.add_argument(
        "--hide-static",
        action="store_true",
        help="Do not show static vehicles at all"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all JSON files in directory"
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    
    if json_path.is_dir():
        # Process directory
        json_files = sorted(json_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {json_path}")
        
        if not json_files:
            print("No JSON files found!")
            return
        
        if args.all:
            # Visualize all files
            if args.output:
                output_dir = json_path / args.output
            else:
                output_dir = json_path / "dual_vis"
            output_dir.mkdir(exist_ok=True)
            
            for json_file in json_files:
                output_path = output_dir / f"{json_file.stem}.png"
                print(f"\n{'='*60}")
                visualize_json_scenario(json_file, output_path, args.frame, args.zoom, args.mirror, args.highlight_dynamic, args.hide_static)
        else:
            # Just visualize first file
            print(f"Visualizing first file (use --all to process all files)")
            print(f"\n{'='*60}")
            output_path = Path(args.output) if args.output else None
            visualize_json_scenario(json_files[0], output_path, args.frame, args.zoom, args.mirror, args.highlight_dynamic, args.hide_static)
    
    elif json_path.is_file():
        # Single file
        output_path = Path(args.output) if args.output else None
        visualize_json_scenario(json_path, output_path, args.frame, args.zoom, args.mirror, args.highlight_dynamic, args.hide_static)
    
    else:
        print(f"Error: {json_path} not found")


if __name__ == "__main__":
    main()
