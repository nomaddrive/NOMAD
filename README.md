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
