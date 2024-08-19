# Getting Started with ODRL

In this section, we provide guides on how to create an environment with dynamics shift and run the corresponding off-dynamics RL algorithm with ODRL.

## Supported Tasks

Now ODRL supports the following tasks:

| Task Domain   | Friction | Gravity | Kinematic | Morphology | Map Layout | Offline Datasets |
|---------------|----------|----------|----------|----------|----------|----------|
| **Locomotion**  | ✅     |    ✅   |   ✅    |    ✅    |    ❎     |   ✅ |
| **Navigation**  | ❎     |    ❎   |  ❎      |    ❎   |    ✅   | ✅  |
| **Dexterous Manipulation**| ❎ | ❎ |   ✅   |    ✅    |    ❎     |  ✅  | 

## Experimental Settings

ODRL contains the following experiemental settings: 

* **Online-Online** setting (online source domain and online target domain)
* **Offline-Online** setting (offline source domain and online target domain)
* **Online-Offline** setting (online source domain and offline target domain)
* **Offline-Offline** setting (offline source domain and offline target domain)

## Hello World

ODRL supports `MuJoCo`, `AntMaze`, `Adroit` and `Sawyer` tasks. We provide a function to call environments from these domains, and they share similar way of usage:

```
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.sawyer.call_sawyer_env          import call_sawyer_env
```

All these functions accepts a dictionary as input where `env_name` and `shift_level` should be specified. For example

```
env_config = {
            'env_name': 'ant-friction',
            'shift_level': '0.5',
        }
```

Then one can call `ant-friction-0.5` environment with

```
env = call_mujoco_env(env_config)
```

Similarly, if one wants to run experiments on `pen-broken-joint-easy` environment, then one can use the following codes:

```
env_config = {
            'env_name': 'pen-broken-joint',
            'shift_level': 'easy',
        }

env = call_adroit_env(env_config)
```

## Running Implemented Algorithms

We run all four experimental settings with the `train.py` file, with `mode 0` denotes the **Online-Online** setting, `mode 1` denotes the **Offline-Online** seting, `mode 2` specifies the **Online-Offline** setting, and `mode 3` means the **Offline-Offline** setting. One can switch different setting by specifying the `--mode` flag. The default value is 0, i.e., **Online-Online** setting. We give an example of how to use our benchmark below:
```bash
# online-online
CUDA_VISIBLE_DEVICES=0 python train.py --policy DARC --env hopper-kinematic-legjnt --shift_level easy --seed 1 --mode 0 --dir runs
# offline-online
CUDA_VISIBLE_DEVICES=0 python train.py --policy CQL_SAC --env ant-friction --shift_level 0.5 --srctype medium-replay --seed 1 --mode 1 --dir runs
# online-offline
CUDA_VISIBLE_DEVICES=0 python train.py --policy PAR_BC --env ant-morph-alllegs --shift_level hard --tartype expert --seed 1 --mode 2 --dir runs
# offline-offline
CUDA_VISIBLE_DEVICES=0 python train.py --policy BOSA --env walker2d-kinematic-footjnt --shift_level medium --srctype medium --tartype medium --seed 1 --mode 3 --dir runs
```
We explain some key flags below:

- `--env` specifies the name of the target domain, and the source domain will be automatically prepared
- `--shift_level` specifies the shift level for the task
- `--srctype` specifies the dataset quality of the source domain dataset
- `--tartype` specifies the dataset quality of the target domain dataset
- `--params` specifies the hyperparameter for the underlying algorithm if one wants to change the default hyperparameters, e.g., `--params '{"actor_lr": 0.003}'`

We directly adopt offline source domain datasets from the popular [D4RL](https://github.com/Farama-Foundation/D4RL) library. Please note that different dynamics shift tasks have varied shift levels. We summarize the shift levels for different tasks below.

| Task          | Supported Shift Levels |
|---------------|-----------------|
| **Locomotion friction/gravity**  | 0.1, 0.5, 2.0, 5.0 |
| **Locomotion kinematic/morphology**  | easy, medium, hard |
| **Antmaze small maze**| centerblock, empty, lshape, zshape, reverseu, reversel  | 
| **Antmaze medium/large maze**| 1, 2, 3, 4, 5, 6 | 
| **Dexterous Manipulation**| easy, medium, hard |

