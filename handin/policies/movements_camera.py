

import numpy as np
from pysc2.lib import features  # This is probably useful

from handin.policies.abc_policy import Policy
from handin.actions.atomic_actions import MoveCamera


_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class CamPineapple(Policy):
    def step(self, obs):
        self.unit_tracker.set_camera_location([32, 32])
        return MoveCamera(1, 32, 32)


class CamPinFriendlySpawn(Policy):

    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        self.unit_tracker = unittracker
        # Coordinates
        self.x = 32
        self.y = 44 if self.unit_tracker.bottom else 20

    def step(self, obs):
        self.unit_tracker.set_camera_location([self.x, self.y])
        return MoveCamera(1, self.x, self.y)


class CamPinEnemySpawn(Policy):
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        self.unit_tracker = unittracker
        # Coordinates
        self.x = 32
        self.y = 20 if self.unit_tracker.bottom else 44

    def step(self, obs):
        self.unit_tracker.set_camera_location([self.x, self.y])
        return MoveCamera(1, self.x, self.y)


class CamFriendlySpawn(Policy):
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        self.unit_tracker = unittracker
        # Coordinates
        self.x = 32
        self.y = 56 if self.unit_tracker.bottom else 8

    def step(self, obs):
        self.unit_tracker.set_camera_location([self.x, self.y])
        return MoveCamera(1, self.x, self.y)


class CamEnemySpawn(Policy):
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        self.unit_tracker = unittracker
        # Coordinates
        self.x = 32
        self.y = 8 if self.unit_tracker.bottom else 56

    def step(self, obs):
        self.unit_tracker.set_camera_location([self.x, self.y])
        return MoveCamera(1, self.x, self.y)


class CamCenterFriendly(Policy):
    """ Center the camera on friendly units.  """

    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        self.unit_tracker = unittracker
        # Coordinates
        self.camera_pos = [32, 32]


    def find_friendly_center(self, obs):
        units = obs.observation.feature_units
        friendlies = [[unit.x, unit.y] for unit in units if unit.alliance == _PLAYER_SELF]
        friendly_center = np.mean(friendlies, axis=0) if len(friendlies) else friendlies[0]

        new_x = round(self.camera_pos[0] + (friendly_center[0] - 39) / 4.5)
        new_y = round(self.camera_pos[1] + (friendly_center[1] - 39) / 4.5)
        return np.clip([new_x, new_y], a_min=0, a_max=63)

    def is_available(self, obs):
        units = obs.observation.feature_units
        friendlies = [[unit.x, unit.y] for unit in units if unit.alliance == _PLAYER_SELF]

        if not friendlies:
            return False

        camera_pos = self.unit_tracker.camera_pos
        friendly_center = np.mean(friendlies, axis=0) if len(friendlies) else friendlies[0]

        if np.sqrt((np.subtract(camera_pos, friendly_center) ** 2).sum(axis=0)) > 15:
            return True
        return False

    def step(self, obs):
        target = self.find_friendly_center(obs)
        self.unit_tracker.set_camera_location([*target])
        return MoveCamera(1, *target)
