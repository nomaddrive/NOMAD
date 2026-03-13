"""Module for initializing vehicle states on a road network."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import KDTree
from typing import Dict, List, Tuple, Optional
import torch

from goal_generation import point_at_s, LaneResources

@dataclass
class InitialGenerationParams:
    """Parameters for vehicle initialization."""
    vehicle_density_per_km: float = 1.0
    # Gamma distribution parameters for density
    use_gamma_density: bool = False
    gamma_alpha: float = 1.2630
    gamma_loc: float = 0.1363
    gamma_scale: float = 0.2500
    
    min_time_gap: float = 1.2  # seconds
    lateral_deviation_std: float = 0.3  # meters
    min_distance: float = 5.0  # meters
    heading_noise_std: float = 0.1  # radians
    min_speed: float = 0.0  # m/s
    speed_mean_percentile: float = 0.85  # 85th percentile speed
    speed_std_percentile: float = 0.22  # 22% std (traffic engineering standard)
    rng_seed: int = 42
    static_vehicle_ratio: float = 0.5
    static_vehicle_prob: float = 0.5

@dataclass
class VehicleState:
    """Complete vehicle state."""
    position_x: float
    position_y: float
    position_z: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    lane_id: int = 0
    s: float = 0.0
    vehicle_length: float = 4.5
    vehicle_width: float = 2.0
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_z: float = 0.0
    is_static: bool = False

def initialize_vehicles_on_network(
    lane_resources: LaneResources,
    params: InitialGenerationParams,
    max_speed: float,
) -> List[Dict[str, float]]:
    """
    Initialize vehicles on the lane network based on density with collision avoidance.
    """
    lanes = lane_resources.lanes
    
    # Calculate total network length in kilometers
    total_length_m = sum(lane["length"] for lane in lanes.values())
    
    if total_length_m == 0:
        return []
        
    total_length_km = total_length_m / 1000.0
    
    if params.use_gamma_density:
        # Sample density from Gamma distribution
        density = np.random.gamma(params.gamma_alpha, params.gamma_scale) + params.gamma_loc
        # Ensure density is non-negative
        density = max(0.0, density)
        num_dynamic_vehicles = int(np.round(density * total_length_km))
        # Ensure at least 1 vehicle
        num_dynamic_vehicles = max(1, num_dynamic_vehicles)
    else:
        # Calculate max number of vehicles based on density
        # Use float to handle densities < 1.0 correctly
        expected_vehicles = params.vehicle_density_per_km * total_length_km
        max_vehicles = max(1, int(np.round(expected_vehicles)))
        
        # Randomly sample number of vehicles between 1 and max_vehicles
        num_dynamic_vehicles = np.random.randint(1, max_vehicles + 1)
    
    if num_dynamic_vehicles == 0:
        return []
    
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
    
    # --- Phase 1: Generate Dynamic Vehicles ---
    max_attempts = num_dynamic_vehicles * 10  # Prevent infinite loop
    attempts = 0
    
    while len(vehicles) < num_dynamic_vehicles and attempts < max_attempts:
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
        target_mean = params.speed_mean_percentile * max_speed
        target_std = params.speed_std_percentile * max_speed
        variance = target_std ** 2
        mu = np.log(target_mean ** 2 / np.sqrt(target_mean ** 2 + variance))
        sigma = np.sqrt(np.log(1 + variance / target_mean ** 2))
        speed = np.random.lognormal(mu, sigma)
        speed = float(np.clip(speed, max(params.min_speed, 0.05 * max_speed), 1.1 * max_speed))
        
        # Calculate minimum distance: max of time-gap-based and absolute minimum
        time_gap_distance = speed * params.min_time_gap
        required_min_distance = max(time_gap_distance, params.min_distance)
        
        # Get candidate position WITH lateral deviation for accurate collision checking
        centerline_point, heading = point_at_s(lane_data, s)
        
        # Apply lateral deviation NOW (before collision checks)
        lateral_offset = np.random.normal(0, params.lateral_deviation_std)
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
            occupied_required_distance = max(occupied_speed * params.min_time_gap, params.min_distance)
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
            heading_noisy = heading + np.random.normal(0, params.heading_noise_std)
            
            vehicle = {
                "position_x": position_x,
                "position_y": position_y,
                "heading": float(heading_noisy),
                "speed": float(speed),
                "lane_id": lane_id,
                "s": float(s),
                "is_static": False,
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
    
    # --- Phase 2: Generate Static Vehicles ---
    # Only if probability condition is met
    if np.random.rand() < params.static_vehicle_prob:
        # Calculate number of static vehicles based on the number of dynamic vehicles generated
        # Static = Dynamic * Ratio
        num_static_vehicles = int(len(vehicles) * params.static_vehicle_ratio)
        
        attempts = 0
        max_attempts_static = num_static_vehicles * 10
        count_static_added = 0
        
        while count_static_added < num_static_vehicles and attempts < max_attempts_static:
            attempts += 1
            
            # Sample a segment
            seg_idx = np.random.choice(len(lane_ids), p=segment_probs)
            lane_id = lane_ids[seg_idx]
            lane_data = lanes[lane_id]
            
            # Sample position uniformly along the segment
            s_start = segment_starts[seg_idx]
            s_offset = np.random.uniform(0, segment_lengths[seg_idx])
            s = s_start + s_offset
            
            # Static vehicle speed is 0
            speed = 0.0
            
            # Calculate minimum distance: just absolute minimum since speed is 0
            required_min_distance = params.min_distance
            
            # Get candidate position WITH lateral deviation for accurate collision checking
            centerline_point, heading = point_at_s(lane_data, s)
            
            # Apply lateral deviation NOW (before collision checks)
            lateral_offset = np.random.normal(0, params.lateral_deviation_std)
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
                occupied_required_distance = max(occupied_speed * params.min_time_gap, params.min_distance)
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
                heading_noisy = heading + np.random.normal(0, params.heading_noise_std)
                
                vehicle = {
                    "position_x": position_x,
                    "position_y": position_y,
                    "heading": float(heading_noisy),
                    "speed": float(speed),
                    "lane_id": lane_id,
                    "s": float(s),
                    "is_static": True,
                }
                vehicles.append(vehicle)
                count_static_added += 1
                
                # Add to occupancy tracking and keep sorted
                lane_occupancy[lane_id].append((s, speed))
                lane_occupancy[lane_id].sort(key=lambda x: x[0])
                
                # Add to spatial tracking for cross-lane collision detection
                vehicle_positions.append((position_x, position_y, required_min_distance))
                
                # Rebuild KDTree with all vehicle positions for efficient spatial queries
                if len(vehicle_positions) > 0:
                    positions_array = np.array([[vx, vy] for vx, vy, _ in vehicle_positions])
                    vehicle_kdtree = KDTree(positions_array)
            
    return vehicles

def compute_second_position(
    vehicle: Dict[str, float],
    timestep: float = 0.1,
) -> Tuple[float, float, float]:
    """
    Compute second position based on initial speed and heading.
    """
    dx = vehicle["speed"] * timestep * np.cos(vehicle["heading"])
    dy = vehicle["speed"] * timestep * np.sin(vehicle["heading"])
    
    return (
        vehicle["position_x"] + dx,
        vehicle["position_y"] + dy,
        0.0,  # z remains 0
    )

def denormalize_position(x: float, y: float, world_means: torch.Tensor, env_idx: int) -> Tuple[float, float]:
    """Convert from normalized to world coordinates."""
    mean_x, mean_y, _ = world_means[env_idx]
    return (x + mean_x.item(), y + mean_y.item())

def create_vehicle_json_object(
    vehicle: VehicleState,
    vehicle_id: int,
    template_obj: Dict,
    world_means: torch.Tensor,
    env_idx: int,
    timestep: float = 0.1,
    num_frames: int = 91,
) -> Dict:
    """
    Create JSON object for a vehicle using template from original scenario.
    """
    # Denormalize all positions to world coordinates
    pos1_x, pos1_y = denormalize_position(vehicle.position_x, vehicle.position_y, world_means, env_idx)
    goal_x, goal_y = denormalize_position(vehicle.goal_x, vehicle.goal_y, world_means, env_idx)
    
    # Compute second position based on speed and heading
    second_pos_dict = {
        "position_x": vehicle.position_x,
        "position_y": vehicle.position_y,
        "heading": vehicle.heading,
        "speed": vehicle.speed,
    }
    pos2_x_norm, pos2_y_norm, pos2_z = compute_second_position(second_pos_dict, timestep)
    pos2_x, pos2_y = denormalize_position(pos2_x_norm, pos2_y_norm, world_means, env_idx)
    
    # Use template object for vehicle dimensions if available
    vehicle_length = template_obj.get("length", vehicle.vehicle_length)
    vehicle_width = template_obj.get("width", vehicle.vehicle_width)
    vehicle_height = template_obj.get("height", 1.5)
    
    position_array = []
    heading_array = []
    velocity_array = []
    
    # Frame 0: Initial position
    position_array.append({"x": pos1_x, "y": pos1_y, "z": vehicle.position_z})
    heading_array.append(float(vehicle.heading))
    # Velocity as {x, y} components
    vx = float(vehicle.speed * np.cos(vehicle.heading))
    vy = float(vehicle.speed * np.sin(vehicle.heading))
    velocity_array.append({"x": vx, "y": vy})
    
    # Frame 1: Second position
    position_array.append({"x": pos2_x, "y": pos2_y, "z": pos2_z})
    heading_array.append(float(vehicle.heading))
    velocity_array.append({"x": vx, "y": vy})
    
    # Frames 2-89: Set to invalid values (-10000.0)
    for i in range(2, num_frames - 1):
        position_array.append({"x": -10000.0, "y": -10000.0, "z": -10000.0})
        heading_array.append(-10000.0)
        velocity_array.append({"x": -10000.0, "y": -10000.0})
    
    # Frame 90: Goal position
    position_array.append({"x": goal_x, "y": goal_y, "z": 0.0})
    heading_array.append(float(vehicle.heading))
    velocity_array.append({"x": vx, "y": vy})
    
    total_distance = np.sqrt((goal_x - pos1_x)**2 + (goal_y - pos1_y)**2)
    
    valid_array = [True] * num_frames
    for i in range(2, num_frames - 1):
        valid_array[i] = False

    vehicle_obj = {
        "id": vehicle_id,
        "type": "vehicle",
        "is_sdc": False,
        "mark_as_expert": False,
        "valid": valid_array,
        "position": position_array,
        "heading": heading_array,
        "velocity": velocity_array,
        "goalPosition": {"x": goal_x, "y": goal_y, "z": 0.0},
        "length": float(vehicle_length),
        "width": float(vehicle_width),
        "height": float(vehicle_height),
        "total_distance_traveled": float(total_distance),
    }
    
    return vehicle_obj
