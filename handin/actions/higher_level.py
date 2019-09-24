
import numpy as np

from .atomic_actions import *


class FirstObs(CompositeAction):
    """
    The optimal sequence when starting the game
        - Select your army
        - Move your army to the center
        - Move the camera (more for player than AI)
    """

    def build_actions(self):
        """
        :return: Executable action
        """
        return [actions.FUNCTIONS.select_army("select"),
                actions.FUNCTIONS.Move_screen("now", [32, 32]),
                actions.FUNCTIONS.move_camera([32, 32])]


class SpecialAttackComplex(CompositeAction):
    """
    The optimal sequence when starting the game
        - Select your army
        - Move your army to the center
        - Move the camera (more for player than AI)
    """

    def build_actions(self, x, y):
        """
        :return: Executable action
        """
        return [actions.FUNCTIONS.Move_screen("now", [x, y]),
                actions.FUNCTIONS.Effect_Stim_quick("now")]


class SplitGroup(CompositeAction):
    """ Splits a group in two, where the second is a selected number of units.  """

    def build_actions(self, units: list):
        commands = []

        # Get all units in 1 group
        commands.append(actions.FUNCTIONS.select_army("select"))
        commands.append(actions.FUNCTIONS.select_control_group("append_and_steal", 0))

        # Select the 'new' units and put them in another group
        for unit in units:
            commands.append(actions.FUNCTIONS.select_point("select", np.clip([unit.x, unit.y], a_min=0, a_max=63)))
            commands.append(actions.FUNCTIONS.select_control_group("append_and_steal", 1))
        return commands