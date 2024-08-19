# Using ODRL with Other Algorithms or Benchmarks

In practice, one may want to use ODRL in their own environments or using their own algorithms without relying on our training scripts. We provide two functions to call environments and datasets from ODRL, `call_odrl_env` and `call_odrl_dataset` in `call_odrl_env.py`.

It is super simple to use them, for example, `call_odrl_env`,

```
call_odrl_env(env_type='mujoco',
                  env_name='halfcheetah-friction',
                  shift_level='0.5'):
```

One needs to specify the env_type (should be one of `mujoco, antmaze, adroit, sawyer`), the environment name (e.g., `pen-broken-joint`), as well as the shift level. It then returns the environment in ODRL.

To call datasets from ODRL, it is recommended to use the `call_odrl_dataset` function,

```
call_odrl_dataset(env_name='halfcheetah-friction',
                      shift_level='0.5',
                      dataset_type='random',
                      )
```

where one needs to specify the environment name, the shift level and the dataset type (should be one of `random, medium, expert`)