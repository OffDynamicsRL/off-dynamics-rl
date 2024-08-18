from gym.envs.registration import register
# register metaworld tasks as gym environments

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from env_dict import (ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
                      ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
                      )

__all__ = ['ALL_V2_ENVIRONMENTS_GOAL_HIDDEN',
           'ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE']


register(
    id='sawyer-pick-place-v2', entry_point='sawyer_xyz.v2.sawyer_pick_place_v2:SawyerPickPlaceEnvV2',
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_pick_place_v2.xml'
    }
)

register(
    id='sawyer-pick-place-broken-easy-v2', entry_point='sawyer_xyz.v2.sawyer_pick_place_v2:SawyerPickPlaceEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_pick_place_broken_easy_v2.xml'
    }
)

register(
    id='sawyer-pick-place-morph-gripper-easy-v2', entry_point='sawyer_xyz.v2.sawyer_pick_place_v2:SawyerPickPlaceEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_pick_place_morph_easy_v2.xml'
    }
)

register(
    id='sawyer-pick-place-broken-medium-v2', entry_point='sawyer_xyz.v2.sawyer_pick_place_v2:SawyerPickPlaceEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_pick_place_broken_medium_v2.xml'
    }
)

register(
    id='sawyer-pick-place-morph-gripper-medium-v2', entry_point='sawyer_xyz.v2.sawyer_pick_place_v2:SawyerPickPlaceEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_pick_place_morph_medium_v2.xml'
    }
)

register(
    id='sawyer-pick-place-broken-hard-v2', entry_point='sawyer_xyz.v2.sawyer_pick_place_v2:SawyerPickPlaceEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_pick_place_broken_hard_v2.xml'
    }
)

register(
    id='sawyer-pick-place-morph-gripper-hard-v2', entry_point='sawyer_xyz.v2.sawyer_pick_place_v2:SawyerPickPlaceEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_pick_place_morph_hard_v2.xml'
    }
)


register(
    id='sawyer-box-v2', entry_point='sawyer_xyz.v2.sawyer_box_close_v2:SawyerBoxCloseEnvV2',
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_box.xml'
    }
)

register(
    id='sawyer-box-broken-easy-v2', entry_point='sawyer_xyz.v2.sawyer_box_close_v2:SawyerBoxCloseEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_box_broken_easy_v2.xml'
    }
)

register(
    id='sawyer-box-morph-gripper-easy-v2', entry_point='sawyer_xyz.v2.sawyer_box_close_v2:SawyerBoxCloseEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_box_morph_easy_v2.xml'
    }
)

register(
    id='sawyer-box-broken-medium-v2', entry_point='sawyer_xyz.v2.sawyer_box_close_v2:SawyerBoxCloseEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_box_broken_medium_v2.xml'
    }
)

register(
    id='sawyer-box-morph-gripper-medium-v2', entry_point='sawyer_xyz.v2.sawyer_box_close_v2:SawyerBoxCloseEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_box_morph_medium_v2.xml'
    }
)

register(
    id='sawyer-box-broken-hard-v2', entry_point='sawyer_xyz.v2.sawyer_box_close_v2:SawyerBoxCloseEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_box_broken_hard_v2.xml'
    }
)

register(
    id='sawyer-box-morph-gripper-hard-v2', entry_point='sawyer_xyz.v2.sawyer_box_close_v2:SawyerBoxCloseEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_box_morph_hard_v2.xml'
    }
)


register(
    id='sawyer-button-press-v2', entry_point='sawyer_xyz.v2.sawyer_button_press_v2:SawyerButtonPressEnvV2',
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_button_press.xml'
    }
)

register(
    id='sawyer-button-press-broken-easy-v2', entry_point='sawyer_xyz.v2.sawyer_button_press_v2:SawyerButtonPressEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_button_press_broken_easy_v2.xml'
    }
)

register(
    id='sawyer-button-press-morph-gripper-easy-v2', entry_point='sawyer_xyz.v2.sawyer_button_press_v2:SawyerButtonPressEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_button_press_morph_easy_v2.xml'
    }
)

register(
    id='sawyer-button-press-broken-medium-v2', entry_point='sawyer_xyz.v2.sawyer_button_press_v2:SawyerButtonPressEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_button_press_broken_medium_v2.xml'
    }
)

register(
    id='sawyer-button-press-morph-gripper-medium-v2', entry_point='sawyer_xyz.v2.sawyer_button_press_v2:SawyerButtonPressEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_button_press_morph_medium_v2.xml'
    }
)

register(
    id='sawyer-button-press-broken-hard-v2', entry_point='sawyer_xyz.v2.sawyer_button_press_v2:SawyerButtonPressEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_button_press_broken_hard_v2.xml'
    }
)

register(
    id='sawyer-button-press-morph-gripper-hard-v2', entry_point='sawyer_xyz.v2.sawyer_button_press_v2:SawyerButtonPressEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_button_press_morph_hard_v2.xml'
    }
)

register(
    id='sawyer-hammer-v2', entry_point='sawyer_xyz.v2.sawyer_hammer_v2:SawyerHammerEnvV2',
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_hammer.xml'
    }
)

register(
    id='sawyer-hammer-broken-easy-v2', entry_point='sawyer_xyz.v2.sawyer_hammer_v2:SawyerHammerEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_hammer_broken_easy_v2.xml'
    }
)

register(
    id='sawyer-hammer-morph-gripper-easy-v2', entry_point='sawyer_xyz.v2.sawyer_hammer_v2:SawyerHammerEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_hammer_morph_easy_v2.xml'
    }
)

register(
    id='sawyer-hammer-broken-medium-v2', entry_point='sawyer_xyz.v2.sawyer_hammer_v2:SawyerHammerEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_hammer_broken_medium_v2.xml'
    }
)

register(
    id='sawyer-hammer-morph-gripper-medium-v2', entry_point='sawyer_xyz.v2.sawyer_hammer_v2:SawyerHammerEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_hammer_morph_medium_v2.xml'
    }
)

register(
    id='sawyer-hammer-broken-hard-v2', entry_point='sawyer_xyz.v2.sawyer_hammer_v2:SawyerHammerEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_hammer_broken_hard_v2.xml'
    }
)

register(
    id='sawyer-hammer-morph-gripper-hard-v2', entry_point='sawyer_xyz.v2.sawyer_hammer_v2:SawyerHammerEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_hammer_morph_hard_v2.xml'
    }
)

register(
    id='sawyer-push-v2', entry_point='sawyer_xyz.v2.sawyer_push_v2:SawyerPushEnvV2',
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_push_v2.xml'
    }
)

register(
    id='sawyer-push-broken-easy-v2', entry_point='sawyer_xyz.v2.sawyer_push_v2:SawyerPushEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_push_broken_easy_v2.xml'
    }
)

register(
    id='sawyer-push-morph-gripper-easy-v2', entry_point='sawyer_xyz.v2.sawyer_push_v2:SawyerPushEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_push_morph_easy_v2.xml'
    }
)

register(
    id='sawyer-push-broken-medium-v2', entry_point='sawyer_xyz.v2.sawyer_push_v2:SawyerPushEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_push_broken_medium_v2.xml'
    }
)

register(
    id='sawyer-push-morph-gripper-medium-v2', entry_point='sawyer_xyz.v2.sawyer_push_v2:SawyerPushEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_push_morph_medium_v2.xml'
    }
)

register(
    id='sawyer-push-broken-hard-v2', entry_point='sawyer_xyz.v2.sawyer_push_v2:SawyerPushEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_push_broken_hard_v2.xml'
    }
)

register(
    id='sawyer-push-morph-gripper-hard-v2', entry_point='sawyer_xyz.v2.sawyer_push_v2:SawyerPushEnvV2', 
    kwargs={
        'model_name': '/sawyer_xyz/sawyer_push_morph_hard_v2.xml'
    }
)


