# Scenario Filtering

Filters nuPlan→GPUDrive scenarios down to a clean, valid subset.

A scenario is **valid** if it satisfies all three criteria:
1. No off-road incidents during expert playback
2. Duration > 5 simulation steps
3. At least one controllable agent

## Pipeline

```bash
# 1. Run the combined filter — emits valid_scenarios_only_<folder>.json under ./output/
python -m filtering.scenarios_valid_filter <data_path> <max_files>

# 2. Copy the valid scenario files into a clean directory
python -m filtering.copy_filtered_scenarios
```

`scenarios_valid_filter.py` runs a single GPUDrive simulation pass that checks all three
criteria simultaneously. `copy_filtered_scenarios.py` reads the JSON list it produces and
copies the matching scenario files from `SOURCE_DIR` to `TARGET_DIR` (paths configured at
the top of the script).

## Requirements

GPUDrive must be installed (see top-level repo). The filter imports `madrona_gpudrive`,
`gpudrive.env.config`, `gpudrive.env.dataset`, and `gpudrive.env.env_torch`.
