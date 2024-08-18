from collections import OrderedDict
import re

import numpy as np

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from sawyer_xyz.v2 import (
    SawyerNutAssemblyEnvV2,
    SawyerBasketballEnvV2,
    SawyerBinPickingEnvV2,
    SawyerBoxCloseEnvV2,
    SawyerButtonPressTopdownEnvV2,
    SawyerButtonPressTopdownWallEnvV2,
    SawyerButtonPressEnvV2,
    SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnvV2,
    SawyerCoffeePullEnvV2,
    SawyerCoffeePushEnvV2,
    SawyerDialTurnEnvV2,
    SawyerNutDisassembleEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnvV2,
    SawyerDoorEnvV2,
    SawyerDrawerCloseEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerFaucetCloseEnvV2,
    SawyerFaucetOpenEnvV2,
    SawyerHammerEnvV2,
    SawyerHandInsertEnvV2,
    SawyerHandlePressSideEnvV2,
    SawyerHandlePressEnvV2,
    SawyerHandlePullSideEnvV2,
    SawyerHandlePullEnvV2,
    SawyerLeverPullEnvV2,
    SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnvV2,
    SawyerPickOutOfHoleEnvV2,
    SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2,
    SawyerPlateSlideBackSideEnvV2,
    SawyerPlateSlideBackEnvV2,
    SawyerPlateSlideSideEnvV2,
    SawyerPlateSlideEnvV2,
    SawyerPushBackEnvV2,
    SawyerPushEnvV2,
    SawyerPushWallEnvV2,
    SawyerReachEnvV2,
    SawyerReachWallEnvV2,
    SawyerShelfPlaceEnvV2,
    SawyerSoccerEnvV2,
    SawyerStickPullEnvV2,
    SawyerStickPushEnvV2,
    SawyerSweepEnvV2,
    SawyerSweepIntoGoalEnvV2,
    SawyerWindowCloseEnvV2,
    SawyerWindowOpenEnvV2,
)

ALL_V2_ENVIRONMENTS = OrderedDict((
    ('sawyer-assembly', SawyerNutAssemblyEnvV2),
    ('sawyer-basketball', SawyerBasketballEnvV2),
    ('sawyer-bin-picking', SawyerBinPickingEnvV2),
    ('sawyer-box-close', SawyerBoxCloseEnvV2),
    ('sawyer-button-press-topdown', SawyerButtonPressTopdownEnvV2),
    ('sawyer-button-press-topdown-wall', SawyerButtonPressTopdownWallEnvV2),
    ('sawyer-button-press', SawyerButtonPressEnvV2),
    ('sawyer-button-press-wall', SawyerButtonPressWallEnvV2),
    ('sawyer-coffee-button', SawyerCoffeeButtonEnvV2),
    ('sawyer-coffee-pull', SawyerCoffeePullEnvV2),
    ('sawyer-coffee-push', SawyerCoffeePushEnvV2),
    ('sawyer-dial-turn', SawyerDialTurnEnvV2),
    ('sawyer-disassemble', SawyerNutDisassembleEnvV2),
    ('sawyer-door-close', SawyerDoorCloseEnvV2),
    ('sawyer-door-lock', SawyerDoorLockEnvV2),
    ('sawyer-door-open', SawyerDoorEnvV2),
    ('sawyer-door-unlock', SawyerDoorUnlockEnvV2),
    ('sawyer-hand-insert', SawyerHandInsertEnvV2),
    ('sawyer-drawer-close', SawyerDrawerCloseEnvV2),
    ('sawyer-drawer-open', SawyerDrawerOpenEnvV2),
    ('sawyer-faucet-open', SawyerFaucetOpenEnvV2),
    ('sawyer-faucet-close', SawyerFaucetCloseEnvV2),
    ('sawyer-hammer', SawyerHammerEnvV2),
    ('sawyer-handle-press-side', SawyerHandlePressSideEnvV2),
    ('sawyer-handle-press', SawyerHandlePressEnvV2),
    ('sawyer-handle-pull-side', SawyerHandlePullSideEnvV2),
    ('sawyer-handle-pull', SawyerHandlePullEnvV2),
    ('sawyer-lever-pull', SawyerLeverPullEnvV2),
    ('sawyer-peg-insert-side', SawyerPegInsertionSideEnvV2),
    ('sawyer-pick-place-wall', SawyerPickPlaceWallEnvV2),
    ('sawyer-pick-out-of-hole', SawyerPickOutOfHoleEnvV2),
    ('sawyer-reach', SawyerReachEnvV2),
    ('sawyer-push-back', SawyerPushBackEnvV2),
    ('sawyer-push', SawyerPushEnvV2),
    ('sawyer-pick-place', SawyerPickPlaceEnvV2),
    ('sawyer-plate-slide', SawyerPlateSlideEnvV2),
    ('sawyer-plate-slide-side', SawyerPlateSlideSideEnvV2),
    ('sawyer-plate-slide-back', SawyerPlateSlideBackEnvV2),
    ('sawyer-plate-slide-back-side', SawyerPlateSlideBackSideEnvV2),
    ('sawyer-peg-insert-side', SawyerPegInsertionSideEnvV2),
    ('sawyer-peg-unplug-side', SawyerPegUnplugSideEnvV2),
    ('sawyer-soccer', SawyerSoccerEnvV2),
    ('sawyer-stick-push', SawyerStickPushEnvV2),
    ('sawyer-stick-pull', SawyerStickPullEnvV2),
    ('sawyer-push-wall', SawyerPushWallEnvV2),
    ('sawyer-push', SawyerPushEnvV2),
    ('sawyer-reach-wall', SawyerReachWallEnvV2),
    ('sawyer-reach', SawyerReachEnvV2),
    ('sawyer-shelf-place', SawyerShelfPlaceEnvV2),
    ('sawyer-sweep-into', SawyerSweepIntoGoalEnvV2),
    ('sawyer-sweep', SawyerSweepEnvV2),
    ('sawyer-window-open', SawyerWindowOpenEnvV2),
    ('sawyer-window-close', SawyerWindowCloseEnvV2),
))

_NUM_METAWORLD_ENVS = len(ALL_V2_ENVIRONMENTS)

def create_hidden_goal_envs():
    hidden_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = True
            env._freeze_rand_vec = False
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d['__init__'] = initialize
        hg_env_name = re.sub("(^|[-])\s*([a-zA-Z])",
                             lambda p: p.group(0).upper(), env_name)
        hg_env_name = hg_env_name.replace("-", "")
        hg_env_key = '{}-goal-hidden'.format(env_name)
        hg_env_name = '{}GoalHidden'.format(hg_env_name)
        HiddenGoalEnvCls = type(hg_env_name, (env_cls, ), d)
        hidden_goal_envs[hg_env_key] = HiddenGoalEnvCls

    return OrderedDict(hidden_goal_envs)


def create_observable_goal_envs():
    observable_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d['__init__'] = initialize
        og_env_name = re.sub("(^|[-])\s*([a-zA-Z])",
                             lambda p: p.group(0).upper(), env_name)
        og_env_name = og_env_name.replace("-", "")

        og_env_key = '{}-goal-observable'.format(env_name)
        og_env_name = '{}GoalObservable'.format(og_env_name)
        ObservableGoalEnvCls = type(og_env_name, (env_cls, ), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = create_hidden_goal_envs()
ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs()
