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

