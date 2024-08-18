# import all algorithms this benchmark implement

def call_tune_algo(algo_name, config, mode, device):
    if mode == 0:
        algo_name = algo_name.lower()
        assert algo_name == 'sac'
        # online online setting, we support SAC
        from finetune.sac_tune import SAC

        algo_to_call = {
            'sac': SAC,
        }

        algo = algo_to_call[algo_name]
        policy = algo(config, device)
    else:
        raise NotImplementedError

    return policy