# ManiSkill-HAB

_A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks_

<video src="https://github.com/user-attachments/assets/133e86e0-c355-412e-8fec-00bb908e53c5" width="100%">
</video>



Official repository for the ManiSkill-HAB project by

[Arth Shukla](https://arth.website/), [Stone Tao](https://stoneztao.com/), [Hao Su](https://cseweb.ucsd.edu/~haosu/)

**[Paper TBA]()** | **[Website](https://arth-shukla.github.io/mshab/)** | **[Models](https://huggingface.co/arth-shukla/mshab_checkpoints)** | **[Dataset TBA]()** | **[Supplementary](https://sites.google.com/view/maniskill-hab)**


## Setup and Installation

1. **Install Environments**

   First, set up a conda environment.
    ```bash
    conda create -n mshab python=3.9
    conda activate mshab
    ```
  
    Next, install ManiSkill3. ManiSkill3 is currently in beta, and new changes can sometimes break MS-HAB. Until ManiSkill3 is officially released, we recommend cloning and checking out a "safe" branch:
    ```bash
    git clone https://github.com/haosulab/ManiSkill.git -b mshab --single-branch
    pip install -e ManiSkill
    pip install -e . # NOTE: you can optionally install train and dev dependencies via `pip install -e .[train,dev]`
    ```
  
    We also host an altered version of the ReplicaCAD dataset necessary for low-level manipulation, which can be downloaded with ManiSkill's download utils. This may take some time:
    ```bash
    # Default installs to ~/.maniskill/data. To change this, add `export MS_ASSET_DIR=[path]`
    python -m mani_skill.utils.download_asset ycb ReplicaCAD ReplicaCADRearrange
    ```
  
    Now the environments can be imported to your script with just one line.
    ```python
    import mshab.envs
    ```

2. **[Optional] Checkpoints, Dataset, and Data Generation**

    To generate data, first make sure you have the envs installed (above). Then, you will need to download our pretrained policies:
    ```bash
    huggingface-cli login   # in case not already authenticated
    huggingface-cli download arth-shukla/mshab_checkpoints --local-dir mshab_checkpoints
    ```
  
    While the dataset is not yet available for download, you can generate the data with trajectory filtering by running the following. However, please note that the full dataset can take up ~490GB (after compression). One can speed up data generation by reducing compression requirements (compression and writing are the main bottlenecks), but this will increase the dataset storage requirements.
    ```bash
    bash scripts/gen_dataset.sh
    ```
  
    You can also edit `mshab/utils/label_dataset.py` to use custom criteria for filtering the dataset (e.g. stricter collision requirements, allow failure data for RL, etc).

3. **[Optional] Training Dependencies**

    To install dependencies for train scripts, simply install the extra dependencies as follows:
    ```bash
    pip install -e .[train]
    ```

4. **[Optional] Dev Dependencies**

    If you'd like to contribute, please run the following to install necessary formatting and testing dependencies:
    ```bash
    pip install -e .[dev]
    ```

## Usage

### Environments

MS-HAB provides an evaluation environment, `SequentialTask-v0` which defines tasks and success/fail conditions. The evaluation environment is ideal for evaluating the HAB's long-horizon tasks.

MS-HAB also provides training environments per subtask `[Name]SubtaskTrain-v0` which add rewards, spawn rejection pipelines, etc (e.g. `PickSubtaskTrain-v0`). Training environments do not support long-horizon tasks (i.e. no skill chaining), however they are ideal for training or evaluating individual skill policies.

Tasks are defined using the `TaskPlan` dataclass (in `mshab/envs/planner.py`). Train environments use precomputed feasible spawn points. `TaskPlan`s and spawn data are installed with `ReplicaCADRearrange` to the directory listed in env var `MS_ASSET_DIR` (defaults to `~/.maniskill/data`) (see [Setup and Installation](#setup-and-installation)).

In this repo, environments are made using code in `mshab/envs/make.py`, including additional useful wrappers. Below we provide an example for making a training environment for TidyHouse Pick on the train split.
```python
import gymnasium as gym

from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mshab.envs
from mshab.envs.planner import plan_data_from_file


REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

task = "tidy_house"
subtask = "pick"
split = "train"

plan_data = plan_data_from_file(
    REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json"
)
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

env = gym.make(
    f"{subtask.capitalize()}SubtaskTrain-v0",
    # Simulation args
    num_envs=256,
    obs_mode="rgbd",
    sim_backend="gpu",
    robot_uids="fetch",
    control_mode="pd_joint_delta_pos",
    # Rendering argsa
    reward_mode="normalized_dense",
    render_mode="rgb_array",
    shader_dir="minimal",
    # TimeLimit args
    max_episode_steps=1000,
    # SequentialTask args
    task_plans=plan_data.plans,
    scene_builder_cls=plan_data.dataset,
    # SubtaskTrain args
    spawn_data_fp=spawn_data_fp,
    # optional: additioanl env_kwargs
)

# add env wrappers here

venv = ManiSkillVectorEnv(
    env,
    max_episode_steps=1000,
    ignore_terminations=True,  # set to False for partial resets
)

# add vector env wrappers here
```

### Training

To run SAC, PPO, BC and Diffusion Policy training with default hyperparamters, you can run

```bash
bash scripts/train_[algo].sh
```

Each `scripts/train_[algo].sh` file also contains additional examples for running and changing hyperparameters.

Default train configs are located under `configs/`. If you have the checkpoints downloaded, you can train using the same hyperparemters using the included train configs by running the following:
```bash
python -m mshab.train_[algo] \
  [path-to-checkpoint-cfg]
  # optionally change specific parameters with CLI

# For example, including overriding
python -m mshab.train_sac \
  mshab_checkpoints/rl/tidy_house/pick/all/config.yml \
  algo.gamma=0.8  # overrides algo.gamma to 0.8
```

You can also resume training using a previous checkpoint. All checkpoints include model weights, optimizer/scheduler states, and other trainable parameters (e.g. log alpha for SAC). To resume training, run the following

```bash
# From checkpoint
python -m mshab.train_[algo] \
  [path-to-checkpoint-cfg] \
  model_ckpt=[path-to-checkpoint]

# For example, including overriding
python -m mshab.train_ppo \
  mshab_checkpoints/rl/tidy_house/pick/all/config.yml \
  model_ckpt=mshab_checkpoints/rl/tidy_house/pick/all/policy.pt \
  algo.lr=1e-3  # overrides algo.lr to 1e-3

# From previous run (resumes logging as well)
python -m mshab.train_[algo] \
  [path-to-checkpoint-cfg] \
  resume_logdir=[path-to-exp-dir]
```

Note that resuming training for SAC is less straightforward than other algorithms since it fills a replay buffer with samples collected during online training. However, setting `algo.init_steps=200_000` to partially refill the replay buffer can give decent results. Your mileage may vary.

### Evaluation

Note that BC and DP need a dataset to be downloaded or generated first. Evaluation using the provided checkpoints for the long-horizon tasks (with teleport nav) or individual subtasks can be done with 
```bash
bash scripts/evaluate_sequential_task.sh
```

Please note that evaluating with teleport nav is currently much slower than evaluating individual subtasks since ManiSkill3 beta does not currently support partial PhysX steps, which are needed for spawn rejection.

## Feature Requests, Bugs, Questions etc

If you have any feature requests find any bugs, or have any questions, please open up an issue or contact us! We're happy to incorporate fixes and changes to improve users' experience. We'll continue to provide updates and improvements to MS-HAB (especially since ManiSkill3 is still in Beta).

We hope our environments, baselines, and dataset are useful to the community!
