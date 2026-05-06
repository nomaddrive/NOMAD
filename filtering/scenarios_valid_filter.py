"""
This script combines off-road detection and quality checking to identify
high-quality scenarios that meet all criteria:
1. No off-road incidents
2. Duration > 5 steps
3. Has controllable agents (> 0)

It outputs a JSON file containing only the valid scenarios that pass all checks.
"""

import os
import json
import glob
import torch
import time
from pathlib import Path
import madrona_gpudrive

from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv


def find_valid_scenarios(data_path: str, max_files: int = 50):
    """
    Combined quality and off-road check to find high-quality scenarios
    """
        
    results = []
    valid_scenarios = []
    rejected_scenarios = {
        'offroad': [],
        'too_short': [],
        'no_controllable': []
    }
    
    total_scenarios = 0
    total_scenario_lengths = []
    total_controllable_agents_all_scenarios = 0
    scenario_details = {}
    processed_scenarios = set()  # Track processed scenarios to avoid duplicates

    # Environment config
    env_config = EnvConfig(
        dynamics_model="delta_local",
        controllable_agent_selection="no_static",
        collision_behavior="ignore",
        dist_to_goal_threshold=2.0
    )
    
    try:
        # Detect available device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create ONE DataLoader that will iterate through scenarios
        data_loader = SceneDataLoader(
            root=data_path,
            batch_size=64,  # Use fixed batch size for GPUDrive
            dataset_size=max_files,  # Load exactly the number we want to process
            sample_with_replacement=False,  # Ensure we get different scenarios
            seed=42,
            shuffle=True  # Shuffle to get variety
        )
        print(f"DataLoader created with {len(data_loader.dataset)} scenarios")
        
        # Create SINGLE environment
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=data_loader,
            max_cont_agents=64,
            device=device,
            action_type="continuous"
        )
        
        batch_num = 1
        
        # Manually advance through the DataLoader
        data_loader_iter = iter(data_loader)
        
        while len(processed_scenarios) < max_files:
            try:
                # Get the next batch of files from DataLoader
                try:
                    batch_files = next(data_loader_iter)
                    print(f"\nBatch {batch_num}: Got {len(batch_files)} scenarios from DataLoader")
                except StopIteration:
                    print(f"\nDataLoader exhausted after {batch_num-1} batches")
                    break
                
                # Use the environment's swap_data_batch method to load new scenarios
                env.swap_data_batch(batch_files)
                
                # Reset environment to initialize with the new batch
                obs = env.reset()
                expert_actions, _, _, _ = env.get_expert_actions()
                
                # Get scenario names for verification
                batch_filenames = env.get_env_filenames()
                actual_scenarios = list(batch_filenames.values())
                
                print("   Environment loaded:")

                # Filter out already processed scenarios
                new_scenarios = []
                for world_idx, filename in enumerate(actual_scenarios):
                    if filename not in processed_scenarios:
                        new_scenarios.append((world_idx, filename))
                        processed_scenarios.add(filename)
                
                if not new_scenarios:
                    print(f"   All scenarios in this batch were already processed, skipping...")
                    batch_num += 1
                    continue
                
                print(f"   Processing {len(new_scenarios)} new scenarios")
                
                # Get controllable agents information
                controllable_mask = env.cont_agent_mask  # Shape: (num_worlds, max_agents)
                
                # Get actual vehicle counts per scenario using info tensor
                info_tensor = env.sim.info_tensor().to_torch()
                
                # Initialize tracking for each scenario
                scenario_end_steps = {}  # {world_idx: step where scenario ended}
                scenario_offroad_vehicles = {}  # {world_idx: set of agent indices that went offroad}
                scenario_total_vehicles = {}    # {world_idx: total number of vehicles}
                scenario_controllable_vehicles = {}  # {world_idx: number of controllable vehicles}
                
                # Initialize tracking only for worlds we're processing
                for world_idx, filename in new_scenarios:
                    scenario_offroad_vehicles[world_idx] = set()
                    
                    # Count only vehicles (EntityType.Vehicle), not all entities
                    world_info = info_tensor[world_idx]  # Shape: (max_agents, info_features)
                    vehicle_mask = world_info[:, -1] == float(madrona_gpudrive.EntityType.Vehicle)
                    actual_vehicle_count = vehicle_mask.sum().item()
                    scenario_total_vehicles[world_idx] = actual_vehicle_count
                    
                    # Count controllable vehicles (vehicles that are also controllable)
                    controllable_vehicle_mask = vehicle_mask & controllable_mask[world_idx]
                    controllable_vehicle_count = controllable_vehicle_mask.sum().item()
                    scenario_controllable_vehicles[world_idx] = controllable_vehicle_count
                
                # Initialize dead agent mask
                dead_agent_mask = ~env.cont_agent_mask.clone()
                
                # Simulate to determine actual scenario lengths AND track offroad incidents
                for time_step in range(env.episode_len):
                    # Step the environment with expert actions
                    env.step_dynamics(expert_actions[:, :, time_step])
                    
                    # Get done status and infos
                    dones = env.get_dones()
                    infos = env.get_infos()
                    
                    # Check for individual scenario termination
                    for world_idx, filename in new_scenarios:
                        if world_idx not in scenario_end_steps:
                            # Check if all agents in this world are done
                            if dones[world_idx].all():
                                scenario_end_steps[world_idx] = time_step
                    
                    # Check for off-road incidents
                    offroad_mask = infos.off_road > 0  # Shape: (num_worlds, max_agents)
                    
                    for world_idx, filename in new_scenarios:
                        # Find which specific agents went offroad
                        offroad_agent_indices = torch.where(offroad_mask[world_idx])[0]
                        for agent_idx in offroad_agent_indices:
                            agent_idx_int = agent_idx.item()
                            scenario_offroad_vehicles[world_idx].add(agent_idx_int)
                    
                    # Update dead agent mask
                    dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
                    
                    # Check if all agents across all scenarios are done (for early exit)
                    if (dead_agent_mask == True).all():
                        print(f"   All agents are done at step {time_step}")
                        break
                
                # Process results for this batch
                for world_idx, filename in new_scenarios:
                    # Determine actual duration
                    if world_idx in scenario_end_steps:
                        actual_duration = scenario_end_steps[world_idx] + 1
                    else:
                        actual_duration = env.episode_len
                    
                    # Get scenario characteristics
                    controllable_count = scenario_controllable_vehicles[world_idx]
                    total_vehicle_count = scenario_total_vehicles[world_idx]
                    num_offroad_vehicles = len(scenario_offroad_vehicles[world_idx])
                    
                    # Determine if scenario is valid (passes ALL checks)
                    rejection_reasons = []
                    
                    if num_offroad_vehicles > 0:
                        rejection_reasons.append('offroad')
                    if actual_duration <= 5:
                        rejection_reasons.append('too_short')
                    if controllable_count == 0:
                        rejection_reasons.append('no_controllable')
                    
                    is_valid = len(rejection_reasons) == 0
                    
                    # Prepare result
                    result = {
                        'scenario_name': filename,
                        'is_valid': is_valid,
                        'rejection_reasons': rejection_reasons,
                        'duration_steps': actual_duration,
                        'total_vehicles': total_vehicle_count,
                        'controllable_agents': controllable_count,
                        'offroad_vehicles': num_offroad_vehicles,
                        'batch_index': batch_num - 1
                    }
                    
                    # Store results
                    results.append(result)
                    scenario_details[filename] = {
                        'total_vehicles': total_vehicle_count,
                        'controllable_agents': controllable_count,
                        'duration_steps': actual_duration,
                        'offroad_vehicles': num_offroad_vehicles,
                        'is_valid': is_valid,
                        'rejection_reasons': rejection_reasons
                    }
                    
                    # Update statistics
                    total_scenario_lengths.append(actual_duration)
                    total_controllable_agents_all_scenarios += controllable_count
                    
                    # Categorize scenarios
                    if is_valid:
                        valid_scenarios.append(filename)
                        print(f"  ✅ {filename}: VALID (duration: {actual_duration}, controllable: {controllable_count}, vehicles: {total_vehicle_count})")
                    else:
                        # Add to rejection categories
                        for reason in rejection_reasons:
                            rejected_scenarios[reason].append(filename)
                        
                        reasons_str = ", ".join(rejection_reasons)
                        print(f"  ❌ {filename}: REJECTED ({reasons_str}) (duration: {actual_duration}, controllable: {controllable_count}, offroad: {num_offroad_vehicles})")
                
                total_processed = len(processed_scenarios)
                print(f"   Processed {len(new_scenarios)} new scenarios ({total_processed}/{max_files} total)")
                
                # Clear some memory
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch {batch_num}: {str(e)}")
                print(f"   Attempting to continue with next batch...")
                torch.cuda.empty_cache()
            
            batch_num += 1
            
            # Check if we've processed enough scenarios
            if len(processed_scenarios) >= max_files:
                break
        
        # Clean up
        if 'env' in locals():
            env.close()
            del env
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Processing stopped. Partial results will be saved. CRITICAL ERROR: {str(e)}")
            
    # Summary calculations
    total_scenarios = len(results)
    avg_duration = sum(total_scenario_lengths) / max(1, len(total_scenario_lengths))
    min_duration = min(total_scenario_lengths) if total_scenario_lengths else 0
    max_duration = max(total_scenario_lengths) if total_scenario_lengths else 0
    avg_controllable_agents = total_controllable_agents_all_scenarios / max(1, total_scenarios)
    
    # Print detailed summary
    print(f"\nDETAILED SUMMARY:")
    print("=" * 60)
    print(f"SCENARIO FILTERING RESULTS:")
    print(f"   Total scenarios analyzed: {total_scenarios}")
    print(f"   VALID scenarios (passed all checks): {len(valid_scenarios)} ({len(valid_scenarios)/max(1,total_scenarios)*100:.1f}%)")
    print(f"   REJECTED scenarios: {total_scenarios - len(valid_scenarios)} ({(total_scenarios - len(valid_scenarios))/max(1,total_scenarios)*100:.1f}%)")
    
    print(f"\nREJECTION BREAKDOWN:")
    print(f"   Scenarios with off-road incidents: {len(rejected_scenarios['offroad'])} ({len(rejected_scenarios['offroad'])/max(1,total_scenarios)*100:.1f}%)")
    print(f"   Scenarios too short (≤ 5 steps): {len(rejected_scenarios['too_short'])} ({len(rejected_scenarios['too_short'])/max(1,total_scenarios)*100:.1f}%)")
    print(f"   Scenarios with no controllable agents: {len(rejected_scenarios['no_controllable'])} ({len(rejected_scenarios['no_controllable'])/max(1,total_scenarios)*100:.1f}%)")
    
    print(f"\nVALID SCENARIOS STATISTICS:")
    valid_results = [r for r in results if r['is_valid']]
    if valid_results:
        valid_durations = [r['duration_steps'] for r in valid_results]
        valid_controllable = [r['controllable_agents'] for r in valid_results]
        valid_vehicles = [r['total_vehicles'] for r in valid_results]
        
        print(f"   Average duration: {sum(valid_durations)/len(valid_durations):.1f} steps")
        print(f"   Duration range: {min(valid_durations)} - {max(valid_durations)} steps")
        print(f"   Average controllable agents: {sum(valid_controllable)/len(valid_controllable):.1f}")
        print(f"   Average total vehicles: {sum(valid_vehicles)/len(valid_vehicles):.1f}")
    
    # Create final results structure focused on valid scenarios
    valid_scenario_details = [
        {
            'scenario_name': result['scenario_name'],
            'duration_steps': result['duration_steps'],
            'total_vehicles': result['total_vehicles'],
            'controllable_agents': result['controllable_agents'],
            'quality_score': {
                'has_controllable_agents': result['controllable_agents'] > 0,
                'sufficient_duration': result['duration_steps'] > 5,
                'no_offroad_incidents': result['offroad_vehicles'] == 0
            }
        }
        for result in results if result['is_valid']
    ]
    
    final_results = {
        'analysis_metadata': {
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_path': data_path,
            'max_files_analyzed': max_files,
            'actual_files_processed': total_scenarios,
            'filtering_criteria': {
                'no_offroad_incidents': 'No vehicles go off-road during simulation',
                'sufficient_duration': 'Scenario duration > 5 steps',
                'has_controllable_agents': 'At least 1 controllable agent present'
            }
        },
        'summary_statistics': {
            'total_scenarios_analyzed': total_scenarios,
            'valid_scenarios_count': len(valid_scenarios),
            'valid_scenarios_percentage': round(len(valid_scenarios)/max(1,total_scenarios)*100, 2),
            'rejected_scenarios_count': total_scenarios - len(valid_scenarios),
            'rejection_breakdown': {
                'offroad_incidents': len(rejected_scenarios['offroad']),
                'too_short_duration': len(rejected_scenarios['too_short']),
                'no_controllable_agents': len(rejected_scenarios['no_controllable'])
            }
        },
        'valid_scenarios': {
            'scenario_names': valid_scenarios,
            'detailed_info': valid_scenario_details,
            'statistics': {
                'average_duration': round(sum([s['duration_steps'] for s in valid_scenario_details])/max(1, len(valid_scenario_details)), 2) if valid_scenario_details else 0,
                'average_total_vehicles': round(sum([s['total_vehicles'] for s in valid_scenario_details])/max(1, len(valid_scenario_details)), 2) if valid_scenario_details else 0,
                'average_controllable_agents': round(sum([s['controllable_agents'] for s in valid_scenario_details])/max(1, len(valid_scenario_details)), 2) if valid_scenario_details else 0
            }
        },
        'rejected_scenarios': {
            'by_reason': rejected_scenarios,
            'all_rejected_details': [
                {
                    'scenario_name': result['scenario_name'],
                    'rejection_reasons': result['rejection_reasons'],
                    'duration_steps': result['duration_steps'],
                    'total_vehicles': result['total_vehicles'],
                    'controllable_agents': result['controllable_agents'],
                    'offroad_vehicles': result['offroad_vehicles']
                }
                for result in results if not result['is_valid']
            ]
        },
        'all_scenarios_raw_data': results
    }
    
    # Create output directory relative to the script's location (./output)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract folder name from data_path for file naming
    folder_name = os.path.basename(os.path.normpath(data_path))
    
    # Save comprehensive JSON
    json_path = os.path.join(output_dir, f'valid_scenarios_comprehensive_{folder_name}.json')
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save focused valid-scenarios-only JSON (what you requested)
    valid_only_results = {
        'analysis_info': {
            'analysis_date': final_results['analysis_metadata']['analysis_date'],
            'data_path': data_path,
            'total_scenarios_analyzed': final_results['summary_statistics']['total_scenarios_analyzed'],
            'valid_scenarios_found': final_results['summary_statistics']['valid_scenarios_count'],
            'filtering_criteria': final_results['analysis_metadata']['filtering_criteria']
        },
        'valid_scenarios': final_results['valid_scenarios']['scenario_names'],
        'valid_scenarios_detailed': final_results['valid_scenarios']['detailed_info'],
        'statistics': final_results['valid_scenarios']['statistics']
    }
    
    valid_json_path = os.path.join(output_dir, f'valid_scenarios_only_{folder_name}.json')
    with open(valid_json_path, 'w') as f:
        json.dump(valid_only_results, f, indent=2)
    
    # Create CSV for valid scenarios
    import csv
    csv_path = os.path.join(output_dir, f'valid_scenarios_{folder_name}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['scenario_name', 'duration_steps', 'total_vehicles', 'controllable_agents']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in valid_scenario_details:
            writer.writerow({
                'scenario_name': scenario['scenario_name'],
                'duration_steps': scenario['duration_steps'],
                'total_vehicles': scenario['total_vehicles'],
                'controllable_agents': scenario['controllable_agents']
            })
    
    # Save text summary
    txt_path = os.path.join(output_dir, f'valid_scenarios_summary_{folder_name}.txt')
    with open(txt_path, 'w') as f:
        f.write("VALID SCENARIOS ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("FILTERING RESULTS:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total scenarios analyzed: {total_scenarios}\n")
        f.write(f"VALID scenarios (passed all checks): {len(valid_scenarios)} ({len(valid_scenarios)/max(1,total_scenarios)*100:.1f}%)\n")
        f.write(f"REJECTED scenarios: {total_scenarios - len(valid_scenarios)} ({(total_scenarios - len(valid_scenarios))/max(1,total_scenarios)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("REJECTION BREAKDOWN:\n")
        f.write(f"   Scenarios with off-road incidents: {len(rejected_scenarios['offroad'])} ({len(rejected_scenarios['offroad'])/max(1,total_scenarios)*100:.1f}%)\n")
        f.write(f"   Scenarios too short (≤ 5 steps): {len(rejected_scenarios['too_short'])} ({len(rejected_scenarios['too_short'])/max(1,total_scenarios)*100:.1f}%)\n")
        f.write(f"   Scenarios with no controllable agents: {len(rejected_scenarios['no_controllable'])} ({len(rejected_scenarios['no_controllable'])/max(1,total_scenarios)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("VALID SCENARIOS LIST:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Scenario Name':<40} {'Duration':<10} {'Vehicles':<10} {'Controllable':<12}\n")
        f.write("-" * 72 + "\n")
        for scenario in valid_scenario_details:
            f.write(f"{scenario['scenario_name']:<40} {scenario['duration_steps']:<10} {scenario['total_vehicles']:<10} {scenario['controllable_agents']:<12}\n")
        
        if valid_scenario_details:
            f.write(f"\nVALID SCENARIOS STATISTICS:\n")
            f.write(f"   Average duration: {sum([s['duration_steps'] for s in valid_scenario_details])/len(valid_scenario_details):.1f} steps\n")
            f.write(f"   Average total vehicles: {sum([s['total_vehicles'] for s in valid_scenario_details])/len(valid_scenario_details):.1f}\n")
            f.write(f"   Average controllable agents: {sum([s['controllable_agents'] for s in valid_scenario_details])/len(valid_scenario_details):.1f}\n")
    
    print(f"\nResults saved to:")
    print(f"   {valid_json_path} (valid scenarios only - main output)")
    print(f"   {json_path} (comprehensive analysis)")
    print(f"   {csv_path} (csv format)")
    print(f"   {txt_path} (txt summary)")

    return final_results


if __name__ == "__main__":
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/nuplan/boston_train"
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    find_valid_scenarios(data_path, max_files)