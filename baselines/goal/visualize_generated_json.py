#!/usr/bin/env python3
"""
Visualize generated scenario JSON files.
Loads a JSON file and displays vehicles, goals, road network.
Validates JSON format and checks for required fields.
"""

import json
import argparse
import matplotlib.pyplot as plt
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


def visualize_json_scenario(json_path: Path, output_path: Path = None, frame: int = 0):
    """
    Load and visualize a scenario from JSON file.
    
    Args:
        json_path: Path to JSON file
        output_path: Optional path to save visualization (if None, display interactively)
        frame: Which frame to visualize (0-90)
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
    
    print(f"Found {len(vehicles)} vehicles")
    print(f"Visualizing frame {frame}/90")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot road network from roads geometry (matching original JSON structure)
    if roads:
        print(f"Found {len(roads)} road segments")
        x_pts = []
        y_pts = []
        for road in roads:
            for point in road.get('geometry', []):
                x_pts.append(point.get('x', 0))
                y_pts.append(point.get('y', 0))
        if x_pts:
            ax.scatter(x_pts, y_pts, c='lightgray', s=1, alpha=0.3, label='Road network')
    else:
        print("No road segments found")
    
    # Plot each vehicle
    visible_vehicles = 0
    for i, vehicle in enumerate(vehicles):
        color = f'C{i % 10}'
        
        # Get position at specified frame
        if frame < len(vehicle.get('position', [])):
            pos = vehicle['position'][frame]
            pos_x, pos_y = pos['x'], pos['y']
        else:
            pos_x, pos_y = 0, 0
        
        # Check if position is valid (ignore -10000.0 values)
        if pos_x <= -9000 or pos_y <= -9000:
            if i < 5: # Print debug for first few skipped
                print(f"Skipping vehicle {i} at frame {frame}: pos=({pos_x}, {pos_y})")
            continue
            
        visible_vehicles += 1


        # Get heading at specified frame
        if frame < len(vehicle.get('heading', [])):
            heading = vehicle['heading'][frame]
        else:
            heading = 0
            
        # Check if heading is valid
        if heading <= -9000:
            heading = 0  # Default to 0 if invalid, but position check usually handles it

        
        # Goal position
        goal = vehicle.get('goalPosition', {})
        goal_x, goal_y = goal.get('x', 0), goal.get('y', 0)
        
        # Vehicle info
        vehicle_id = vehicle.get('id', 'unknown')
        vehicle_type = vehicle.get('type', 'unknown')
        is_sdc = vehicle.get('is_sdc', False)
        distance = vehicle.get('total_distance_traveled', 0)
        
        label = f'V{i} ({"SDC" if is_sdc else vehicle_type})'
        
        # Plot vehicle position
        ax.plot(pos_x, pos_y, 'o', color=color, markersize=10, label=label if i < 10 else '')
        
        # Plot goal
        ax.plot(goal_x, goal_y, '*', color=color, markersize=15)
        
        # Connection line
        ax.plot([pos_x, goal_x], [pos_y, goal_y], '--', color=color, alpha=0.3, linewidth=1)
        
        # Orientation arrow
        arrow_len = 3.0
        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)
        ax.arrow(pos_x, pos_y, dx, dy, head_width=1.5, head_length=1.5,
                fc=color, ec=color, alpha=0.7)
        
        # Add text annotation for first few vehicles
        if i < 5:
            ax.text(pos_x + 2, pos_y + 2, f'{vehicle_id}\n{distance:.1f}m',
                   fontsize=8, color=color)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    
    title = f'{json_path.stem}\nFrame {frame}/90, {len(vehicles)} vehicles'
    if errors:
        title += '\n⚠️ JSON has validation errors (see console)'
    ax.set_title(title, fontsize=14)
    
    if len(vehicles) <= 10:
        ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated scenario JSON files and validate format"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for visualization image (if not provided, displays interactively)"
    )
    parser.add_argument(
        "--frame", "-f",
        type=int,
        default=0,
        help="Which frame to visualize (0-90, default: 0)"
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
            output_dir = json_path / "visualizations"
            output_dir.mkdir(exist_ok=True)
            
            for json_file in json_files:
                output_path = output_dir / f"{json_file.stem}_viz.png"
                print(f"\n{'='*60}")
                visualize_json_scenario(json_file, output_path, args.frame)
        else:
            # Just visualize first file
            print(f"Visualizing first file (use --all to process all files)")
            print(f"\n{'='*60}")
            visualize_json_scenario(json_files[0], args.output, args.frame)
    
    elif json_path.is_file():
        # Single file
        output_path = Path(args.output) if args.output else None
        visualize_json_scenario(json_path, output_path, args.frame)
    
    else:
        print(f"Error: {json_path} not found")


if __name__ == "__main__":
    main()
