import numpy as np
from pysc2.lib import features

from .abc_policy import Policy
from handin.actions.atomic_actions import AttackScreen, EffectStimQuick
from handin.actions.higher_level import SpecialAttackComplex

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class AttackClosest(Policy):
    """ Attack the closest enemy.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)

        self.enemies = []
        self.friendlies = []

    def get_target(self):
        friendly = np.mean(self.friendlies, axis=0) if len(self.friendlies) > 1 else self.friendlies[0]
        enemy_distance = np.subtract(self.enemies, friendly) ** 2
        enemy_sorted = sorted(enemy_distance, key=lambda x: x[0] + x[1])
        return np.clip(enemy_sorted[0], a_min=0, a_max=63)

    def is_available(self, obs):
        self.enemies = []
        self.friendlies = []
        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_ENEMY:
                self.enemies.append([unit.x, unit.y])
            if (unit.alliance == _PLAYER_SELF) and unit.is_selected:
                self.friendlies.append([unit.x, unit.y])
        if len(self.enemies) == 0 or len(self.friendlies) == 0:
            return False
        return True

    def step(self, obs):
        target = self.get_target()
        return AttackScreen(1, *target)


class AttackFurthest(Policy):
    """ Attack the furthest enemy.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)

        self.enemies = []
        self.friendlies = []

    def get_target(self):
        friendly = np.mean(self.friendlies, axis=0) if len(self.friendlies) > 1  else self.friendlies[0]
        enemy_distance = np.subtract(self.enemies, friendly) ** 2
        enemy_sorted = sorted(enemy_distance, key=lambda x: x[0] + x[1], reverse=True)
        return np.clip(enemy_sorted[0], a_min=0, a_max=63)

    def is_available(self, obs):
        self.enemies = []
        self.friendlies = []
        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_ENEMY:
                self.enemies.append([unit.x, unit.y])
            if (unit.alliance == _PLAYER_SELF) and unit.is_selected:
                self.friendlies.append([unit.x, unit.y])

        if len(self.enemies) == 0 or len(self.friendlies) == 0:
            return False
        return True

    def step(self, obs):
        target = self.get_target()
        return AttackScreen(1, *target)


class AttackLowest(Policy):
    """ Attack the unit with the lowest health + shield.  """
    def __init__(self, unittracker):
        Policy.__init__(self, unittracker)

        self.enemies = []
        self.friendlies = []

    def get_target(self):
        unit = sorted(self.enemies, key=lambda unit: unit.shield + unit.health)[0]
        return np.clip([unit.x, unit.y], a_min=0, a_max=63)

    def is_available(self, obs):
        self.enemies = []
        self.friendlies = []
        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_ENEMY:
                self.enemies.append(unit)
            if (unit.alliance == _PLAYER_SELF) and unit.is_selected:
                self.friendlies.append([unit.x, unit.y])

        if len(self.enemies) == 0 or len(self.friendlies) == 0:
            return False
        return True

    def step(self, obs):
        target = self.get_target()
        return AttackScreen(1, *target)


class SpecialAttackP(Policy):
    """ Fires special attack """
    def special_attack(self):
        return EffectStimQuick(1)

    def is_available(self, obs):
        if 234 in obs.observation.available_actions:
            return True
        return False

    def step(self, obs):
        """ Called every step """
        action = self.special_attack
        return action()


class SpecialAttackedAimed(Policy):
    def special_attack(self, obs):
        enemies = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.alliance == _PLAYER_ENEMY]
        enemies_center = np.mean(enemies, axis=0) if len(enemies) > 1 else enemies[0]
        return SpecialAttackComplex(1, *np.clip(enemies_center, a_min=0, a_max=64))

    def is_available(self, obs):
        if 234 not in obs.observation.available_actions:
            return False

        enemies = []
        friendlies = []
        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_ENEMY:
                enemies.append([unit.x, unit.y])
            if (unit.alliance == _PLAYER_SELF) and unit.is_selected:
                friendlies.append([unit.x, unit.y])

        if not friendlies or not enemies:
            return False

        friendlies_center = np.mean(friendlies, axis=0) if len(friendlies) > 1  else friendlies[0]
        enemies_center = np.mean(enemies, axis=0) if len(enemies) > 1 else enemies[0]

        distance = np.sqrt(np.sum(np.subtract(friendlies_center, enemies_center)**2, axis=0))

        if 12 <= distance <= 20:
            return True

    def step(self, obs):
        """ Called every step """
        action = self.special_attack
        return action(obs)
