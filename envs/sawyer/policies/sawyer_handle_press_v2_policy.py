import numpy as np

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from policies.action import Action
from policies.policy import Policy, assert_fully_parsed, move


class SawyerHandlePressV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'handle_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = -1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['handle_pos'] + np.array([.0, -.02, .0])

        if np.linalg.norm(pos_curr[:2] - pos_button[:2]) > 0.02:
            return pos_button + np.array([0., 0., 0.2])
        else:
            return pos_button + np.array([.0, .0, -.5])