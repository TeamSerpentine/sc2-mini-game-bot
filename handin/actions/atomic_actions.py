from .action import CompositeAction
from pysc2.lib import actions

""""This file contains all the atomic actions available during the game"""
# TODO: Do not return an action list but instead add an execute function
# TODO: This way it can be added to the queue and executed there.
# TODO: We should discuss about the standard properties of an action.


class NoOp(CompositeAction):
    """
    Performs no action. Useful for when the AI still has to wait for something to happen.
    """

    def build_actions(self):
        """
        :return: Executable action
        """
        return [actions.FUNCTIONS.no_op()]


class MoveCamera(CompositeAction):
    """
    Tries to move the camera center to a particular location on the minimap.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, x, y):
        """
        :param: x, y coordinates on the minimap
        :return: Executable action
        """
        if x < 0 or x > 64 or y < 0 or y > 64:
            raise ValueError("MoveCamera: Coordinates are not within bounds")

        return [actions.FUNCTIONS.move_camera([x, y])]


class SelectPoint(CompositeAction):
    """
    Tries to select a unit at a particular location on the minimap.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, x, y):
        """
        :param: x, y coordinates on the minimap
        :return: Executable action
        """
        if x < 0 or x > 64 or y < 0 or y > 64:
            raise ValueError("SelectPoint: Coordinates are not within bounds")

        return [actions.FUNCTIONS.select_point("select", [x, y])]


class SelectRect(CompositeAction):
    """
    Tries to select units within a rectangle on the screen.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, x1, y1, x2, y2):
        """
        :param: x1, y1, x2, y2 coordinates on the screen
        :return: Executable action
        """
        if x1 < 0 or x1 > 64 or y1 < 0 or y1 > 64 or x2 < 0 or x2 > 64 or y2 < 0 or y2 > 64:
            raise ValueError("SelectRect: Coordinates are not within bounds")

        return [actions.FUNCTIONS.select_rect("select", [x1, y1], [x2, y2])]


class SelectControlGroup(CompositeAction):
    """
    Tries to select units by control group number.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, group_nr, command="set"):
        """
        :param: group_nr the number of the control group
        :param: command possible options '"recall", "set", "append", "set_and_steal", "append_and_steal"'
        :return: Executable action
        """
        if group_nr < 0:
            raise ValueError("SelectControlGroup: Invalid group number")

        return [actions.FUNCTIONS.select_control_group(command, group_nr)]

class SelectArmy(CompositeAction):
    """
    Tries to select the whole army.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self):
        """
        :return: Executable action
        """
        return [actions.FUNCTIONS.select_army("select")]


class AttackScreen(CompositeAction):
    """
    Tries to attack a unit at a particular location on the screen.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, x, y):
        """
        :param: x, y coordinates on the screen
        :return: Executable action
        """
        if x < 0 or x > 64 or y < 0 or y > 64:  # FIXME: These bounds should be 84 84 right?
            raise ValueError("AttackScreen: Coordinates are not within bounds")

        return [actions.FUNCTIONS.Attack_screen("now", [x, y])]


class AttackMinimap(CompositeAction):
    """
    Tries to attack a unit at a particular location on the minimap.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, x, y):
        """
        :param: x, y coordinates on the minimap
        :return: Executable action
        """
        if x < 0 or x > 64 or y < 0 or y > 64:  # FIXME: These bounds should be 64 64 right?
            raise ValueError("AttackMinimap: Coordinates are not within bounds")

        return [actions.FUNCTIONS.Attack_minimap("now", [x, y])]


class EffectStimQuick(CompositeAction):
    """
    Tries to cast the explosion ability.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self):
        """
        :return: Executable action
        """
        return [actions.FUNCTIONS.Effect_Stim_quick("now")]


class MoveScreen(CompositeAction):
    """
    Tries to move selected units to a particular location on the screen.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, x, y):
        """
        :param: x, y coordinates on the screen
        :return: Executable action
        """
        if x < 0 or x > 84 or y < 0 or y > 84:
            raise ValueError("MoveScreen: Coordinates are not within bounds")

        return [actions.FUNCTIONS.Move_screen("now", [x, y])]


class MoveMinimap(CompositeAction):
    """
    Tries to move selected units to a particular location on the minimap.
    TODO: Check whether this functionality description is true
    """

    def build_actions(self, x, y):
        """
        :param: x, y coordinates on the minimap
        :return: Executable action
        """
        if x < 0 or x > 64 or y < 0 or y > 64:
            raise ValueError("MoveMinimap: Coordinates are not within bounds")

        return [actions.FUNCTIONS.Move_minimap("now", [x, y])]
