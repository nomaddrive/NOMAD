This is the implementation of *Learning to Drive in New Cities Without Human Demonstrations*

## Environment Configuration
### Compile Docker image
```
DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=true --tag gpudrive:latest --progress=plain .
```
### Install dependencies
```
# Run Docker container
docker run --gpus all -it --rm --shm-size=20G -v ${PWD}:/workspace gpudrive:latest /bin/bash
# Build gpudrive
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && find external -type f -name "*.tar" -delete
make -j
```
## Dataset Download
NuPlan scenarios are available as the [Official Website](https://www.nuscenes.org/nuplan).

## Data Preparation
The training pipeline consumes GPUDrive-format JSON scenarios. Two preparation
stages turn raw nuPlan logs into the clean subset used in our experiments:

### 1. Convert nuPlan → GPUDrive JSON (`conversion/`)
```
export NUPLAN_DATA_ROOT=/path/to/nuplan/dataset   # directory with .db files
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

cd conversion
make nuplan
scenariomax-convert \
    --nuplan_src $NUPLAN_DATA_ROOT \
    --dst /path/to/output \
    --target_format gpudrive \
    --num_workers 16 \
    --nuplan_scenario_duration 15.0
```
See `conversion/docs/nuplan_to_gpudrive.md` for the full schema and CLI options.

### 2. Filter to valid scenarios (`filtering/`)
A scenario is kept only if it has no off-road incidents, runs for more than 5
simulation steps, and contains at least one controllable agent.
```
# Run the combined filter -- emits valid_scenarios_only_<folder>.json under ./output/
python -m filtering.scenarios_valid_filter <data_path> <max_files>

# Copy the valid scenario files into a clean directory
python -m filtering.copy_filtered_scenarios
```

## Training
### Demonstration Generation
```
python baselines/bc/collect_multi_discrete_demo.py
```
### Imitation Learning
```
python baselines/bc/bc.py
```
### Scenario Generation
```
python baselines/goal/generate_scene_heuristic_modular.py
```
### Reinforcement Learning
```
python baselines/bcsp/ppo_finetuning.py
```
## WOSAC Evaluation
```
python baselines/eval/wosac_evaluation.py
```
