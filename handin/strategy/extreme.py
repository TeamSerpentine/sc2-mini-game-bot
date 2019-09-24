
# General dependencies
from itertools import cycle

from handin.policies.attack import *
from handin.policies.movement_units import *
from handin.policies.selection import SelectArmyP
from handin.policies.movements_camera import *
from handin.actions.atomic_actions import NoOp

_PLAYER_SELF = features.PlayerRelative.SELF


class ExtremeAggressiveBackUpS(Policy):

    def __init__(self, training_flag, unittracker):
        Policy.__init__(self, unittracker)

        # Selects
        self.SelectArmyP = SelectArmyP(unittracker)

        # Cameras
        self.CamPineapple = CamPineapple(unittracker)
        self.CamCenterFriendly = CamCenterFriendly(unittracker)

        # Movements
        self.LocFlankLeftP = LocFlankLeftP(unittracker)
        self.LocFlankRightP = LocFlankRightP(unittracker)
        self.LocPineappleP = LocPineappleP(unittracker)

        # Attacks
        self.AttackClosest = AttackClosest(unittracker)
        self.AttackLowest = AttackLowest(unittracker)
        self.SpecialAttackedAimed = SpecialAttackedAimed(unittracker)

        self.camera_positions = cycle([CamFriendlySpawn, CamPineapple, CamEnemySpawn])

        self.possible_actions = self.get_possible_actions()
        self.possible_action_distribution = [0.30, 0.35, 0.20, 0.05, 0.05, 0.05]

    def step(self, obs):
        # Perform actions
        if obs.first():
            # As first move, select army, to help the AI.
            if self.SelectArmyP.is_available(obs):
                return self.SelectArmyP.step(obs)
            else:
                return NoOp(1)

        if obs.last():
            return NoOp(1)

        # Rotate the camera to search for units
        if not [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]:
            return next(self.camera_positions)(self.unit_tracker).step(obs)

        for _ in range(10):
            action = np.random.choice(self.possible_actions, p=self.possible_action_distribution)

            if action.is_available(obs):
                return action.step(obs)
        return NoOp(1)

    def get_possible_actions(self):
        return [self.SpecialAttackedAimed,
                self.AttackClosest,
                self.AttackLowest,
                self.LocPineappleP,
                self.SelectArmyP,
                self.CamCenterFriendly]
