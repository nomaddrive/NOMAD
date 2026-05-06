#!/usr/bin/env python3
"""
Script to copy filtered scenario files by reading from a JSON file. 
Prepared for working with collect_multi_discrete_demo.py where some scenarios are filtered out.
"""

import json
import shutil
from pathlib import Path


# Configuration
SOURCE_DIR = "data/nuplan/boston_train"
TARGET_DIR = "data/nuplan/valid_scenarios" 
FILTER_JSON = "data_utils/output/valid_scenarios_only.json"


def main():
    """Copy scenario files listed in the filter JSON to a new directory."""
    
    # Convert to Path objects
    source_dir = Path(SOURCE_DIR)
    target_dir = Path(TARGET_DIR)
    filter_json_path = Path(FILTER_JSON)
    
    print("🚀 Starting filtered scenario copy process...")
    print(f"📂 Source directory: {source_dir.absolute()}")
    print(f"📁 Target directory: {target_dir.absolute()}")
    print(f"📋 Filter JSON: {filter_json_path.absolute()}")
    
    # Validate paths exist
    if not source_dir.exists():
        print(f"❌ Error: Source directory does not exist: {source_dir}")
        return
    
    if not filter_json_path.exists():
        print(f"❌ Error: Filter JSON file does not exist: {filter_json_path}")
        return
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the filter JSON
    try:
        with open(filter_json_path, 'r') as f:
            filtered_scenarios = json.load(f)
        print(f"✅ Successfully loaded filter JSON")
    except json.JSONDecodeError as e:
        print(f"❌ Error reading JSON file: {e}")
        return
    except Exception as e:
        print(f"❌ Error opening file: {e}")
        return
    
    # Extract scenario filenames
    scenario_files = [entry['scenario_name'] for entry in filtered_scenarios if 'scenario_name' in entry]
    print(f"📋 Found {len(scenario_files)} scenario files to copy\n")
    
    # Copy files with real-time feedback
    copied_count = 0
    missing_count = 0
    error_count = 0
    
    for i, scenario_file in enumerate(scenario_files, 1):
        source_path = source_dir / scenario_file
        target_path = target_dir / scenario_file
        
        if not source_path.exists():
            print(f"❌ [{i:3d}/{len(scenario_files)}] Missing: {scenario_file}")
            missing_count += 1
            continue
        
        try:
            shutil.copy2(source_path, target_path)
            copied_count += 1
            print(f"📄 [{i:3d}/{len(scenario_files)}] Copied: {scenario_file}")
            
            # Progress indicator every 25 files
            if copied_count % 25 == 0:
                print(f"🔄 Progress: {copied_count}/{len(scenario_files)} files copied...")
                
        except Exception as e:
            print(f"❌ [{i:3d}/{len(scenario_files)}] Error copying {scenario_file}: {e}")
            error_count += 1
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"📊 COPY SUMMARY")
    print(f"{'='*50}")
    print(f"📋 Total files to copy: {len(scenario_files)}")
    print(f"✅ Successfully copied: {copied_count}")
    print(f"❌ Missing from source: {missing_count}")
    print(f"⚠️  Copy errors: {error_count}")
    print(f"📁 Target directory: {target_dir.absolute()}")
    print(f"{'='*50}")
    
    if error_count == 0 and missing_count == 0:
        print("🎉 All files copied successfully!")
    else:
        print("⚠️  Some files could not be copied. Check the output above for details.")


if __name__ == "__main__":
    main()