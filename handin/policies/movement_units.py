
import numpy as np

from pysc2.lib import features
from pysc2.agents import base_agent

from handin.policies.abc_policy import Policy
from handin.actions.atomic_actions import *

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class LocEnemySpawnP(base_agent.BaseAgent, Policy):
    """ Move to enemy spawn.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker
        # Coordinates
        self.center = [32, 32]
        self.enemy_base = [32, 8] if self.unit_tracker.bottom else [32, 56]

    def move_enemy_base(self):
        return MoveMinimap(1, *self.enemy_base)

    def is_available(self, obs):
        if 332 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called at every step """
        action = self.move_enemy_base
        return action()


class LocFriendlySpawnP(base_agent.BaseAgent, Policy):
    """ Move to friendly spawn.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker
        # Coordinates
        self.player_start = [32, 56] if self.unit_tracker.bottom else [32, 8]

    def retreat(self):
        return MoveMinimap(1, *self.player_start)

    def is_available(self, obs):
        if 332 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called at every step """
        action = self.retreat
        return action()


class LocPinEnemyP(base_agent.BaseAgent, Policy):
    """ Moves to point between enemy spawn and pineapple.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker
        # Coordinates
        self.pin_enemy = [32, 20] if self.unit_tracker.bottom else [32, 44]

    def move_pin_enemy(self):
        return MoveMinimap(1, *self.pin_enemy)

    def is_available(self, obs):
        if 332 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called at every step """
        action = self.move_pin_enemy
        return action()


class LocPinFriendlyP(base_agent.BaseAgent, Policy):
    """ Moves to point between friendly spawn and pineapple.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker
        # Coordinates
        self.pin_friendly = [32, 44] if self.unit_tracker.bottom else [32, 20]

    def move_pin_friendly(self):
        return MoveMinimap(1, *self.pin_friendly)

    def is_available(self, obs):
        if 332 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called at every step """
        action = self.move_pin_friendly
        return action()


class LocPineappleP(base_agent.BaseAgent, Policy):
    """ Moves to pineapple.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker
        # Coordinates
        self.center = [32, 32]

    def move_center(self):
        return MoveMinimap(1, *self.center)

    def is_available(self, obs):
        if 332 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called at every step """
        action = self.move_center
        return action()


class LocFlankLeftP(base_agent.BaseAgent, Policy):
    """ Move to the left of pineapple.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker
        # Coordinates
        self.flank_left = [24, 32] if self.unit_tracker.bottom else [40, 32]

    def move_flank_left(self):
        return MoveMinimap(1, *self.flank_left)

    def is_available(self, obs):
        if 332 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called at every step """
        action = self.move_flank_left
        return action()


class LocFlankRightP(base_agent.BaseAgent, Policy):
    """ Move to the right of pineapple.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker
        # Coordinates
        self.flank_right = [40, 32] if self.unit_tracker.bottom else [24, 32]

    def move_flank_right(self):
        return MoveMinimap(1, *self.flank_right)

    def is_available(self, obs):
        if 332 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called at every step """
        action = self.move_flank_right
        return action()


class MoveFarthestP(base_agent.BaseAgent, Policy):
    """ Move to the furthest enemy.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker

    @staticmethod
    def _xy_locs(mask):
        """ Mask should be a set of bools from comparison with a feature layer.  """
        y, x = mask.nonzero()
        return list(zip(x, y))

    def is_available(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        friendlies = self._xy_locs(player_relative == _PLAYER_SELF)
        enemy = self._xy_locs(player_relative == _PLAYER_ENEMY)
        if friendlies and enemy and 331 in obs.observation.available_actions:
            return True
        return False

    def move_farthest_enemy(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        friendlies = self._xy_locs(player_relative == _PLAYER_SELF)
        enemy = self._xy_locs(player_relative == _PLAYER_ENEMY)

        friendly = np.mean(friendlies, axis=0).round()
        farthest_enemy = enemy[int(np.argmax((np.subtract(enemy, friendly) ** 2).sum(axis=0)))]
        return MoveScreen(1, *np.clip(farthest_enemy, a_min=0, a_max=63))

    def step(self, obs):
        action = self.move_farthest_enemy
        return action(obs)


class MoveClosestP(base_agent.BaseAgent, Policy):
    """ Move to closest enemy.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker

    @staticmethod
    def _xy_locs(mask):
        """ Mask should be a set of bools from comparison with a feature layer.  """
        y, x = mask.nonzero()
        return list(zip(x, y))

    def is_available(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        friendlies = self._xy_locs(player_relative == _PLAYER_SELF)
        enemy = self._xy_locs(player_relative == _PLAYER_ENEMY)
        if friendlies and enemy and 331 in obs.observation.available_actions:
            return True
        return False

    def move_closest_enemy(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        friendlies = self._xy_locs(player_relative == _PLAYER_SELF)
        enemy = self._xy_locs(player_relative == _PLAYER_ENEMY)

        friendly = np.mean(friendlies, axis=0).round()
        closest_enemy = enemy[int(np.argmin((np.subtract(enemy, friendly) ** 2).sum(axis=0)))]
        return MoveScreen(1, *np.clip(closest_enemy, a_min=0, a_max=63))

    def step(self, obs):
        action = self.move_closest_enemy
        return action(obs)


class MoveMeanEnemyP(base_agent.BaseAgent, Policy):
    """ Move to the center of the enemies.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)
        base_agent.BaseAgent.__init__(self)
        self.unit_tracker = unittracker

    @staticmethod
    def _xy_locs(mask):
        """ Mask should be a set of bools from comparison with a feature layer.  """
        y, x = mask.nonzero()
        return list(zip(x, y))

    def is_available(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        friendlies = self._xy_locs(player_relative == _PLAYER_SELF)
        enemy = self._xy_locs(player_relative == _PLAYER_ENEMY)
        if friendlies and enemy and 331 in obs.observation.available_actions:
            return True
        return False

    def move_mean_enemy(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        enemy = self._xy_locs(player_relative == _PLAYER_ENEMY)

        enemy_center = np.mean(enemy, axis=0).round()
        return MoveScreen(1, *np.clip(enemy_center, a_min=0, a_max=63))

    def step(self, obs):
        action = self.move_mean_enemy
        return action(obs)
