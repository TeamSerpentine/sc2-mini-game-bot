

import numpy as np

from collections import deque
from pysc2.lib import features

from handin.policies.abc_policy import Policy
from handin.actions.atomic_actions import *
from handin.actions.higher_level import SplitGroup

_PLAYER_SELF = features.PlayerRelative.SELF


class SelectArmyP(Policy):
    """Select army"""
    def is_available(self, obs):
        return 7 in obs.observation.available_actions

    def step(self, obs):
        return SelectArmy(1)


class SelectArmySplitManyTo1P(Policy):
    """ Select army with split X and 1.  """

    def is_available(self, obs):
        select_army = 7 in obs.observation.available_actions
        if len(self.return_units(obs)) >= 2 and select_army:
            return True
        return False

    def step(self, obs):
        units = self.return_units(obs)
        return self.order_split(units, 1)

    @staticmethod
    def return_units(obs):
        units = []
        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_SELF:
                units.append(unit)
        return units

    @staticmethod
    def order_split(own_units: list, second_group: int):
        """ Executes the command for splitting the army in multiple grousp.

        :param own_units list of all the units given by the feature screenm
        :param second_group the number of units in the second group
        """

        return SplitGroup(1, own_units[:second_group])


class SelectArmySplitManyTo2P(Policy):
    """ Select army with split X and 2.  """

    def is_available(self, obs):
        select_army = 7 in obs.observation.available_actions
        if len(self.return_units(obs)) >= 4 and select_army:
            return True
        return False

    def step(self, obs):
        units = self.return_units(obs)
        return self.order_split(units, 2)

    @staticmethod
    def return_units(obs):
        units = []
        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_SELF:
                units.append(unit)
        return units

    @staticmethod
    def order_split(own_units: list, second_group: int):
        """ Executes the command for splitting the army in multiple grousp.

        :param own_units list of all the units given by the feature screenm
        :param second_group the number of units in the second group
        """

        return SplitGroup(1, own_units[:second_group])


class SelectArmyControlGroup0(Policy):
    """ Select army from control group  """
    def is_available(self, obs):
        units = obs.observation.control_groups[0][1]
        if units:
            return True
        return False

    def step(self, obs):
        return SelectControlGroup(1, 0, "recall")


class SelectArmyControlGroup1(Policy):
    """ Select army from control group  """
    def is_available(self, obs):
        units = obs.observation.control_groups[1][1]
        if units:
            return True
        return False

    def step(self, obs):
        return SelectControlGroup(1, 1, "recall")
