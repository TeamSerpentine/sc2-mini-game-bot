
import logging

from pysc2.lib import actions

from abc import abstractmethod, ABC


class Action(ABC):
    """
    An action is a low-level but not (necessarily) atomic action a unit is executing.
    Examples: Walk to pineapple; attack nearest enemy; walk in circle around pineapple
    """

    @abstractmethod
    def step(self, *args):
        """
        :return: List of executable action(s) (may be empty)
        """
        pass


class CompositeAction(ABC):
    """
    An action is a low-level but not (necessarily) atomic action a unit is executing.
    Examples: Walk to pineapple; attack nearest enemy; walk in circle around pineapple
    """

    actions = []        # The list of actions in this composite action class
    action_count = 0    # The number of actions in this composite action
    priority = 0        # The priority of this composite action

    def __init__(self, priority, *args):
        self.actions = self.build_actions(*args)
        self.debug_data = [(action.function.name, *args) for action in self.actions]
        self.action_count = len(self.actions)
        self._priority = priority

    @abstractmethod
    def build_actions(self, *args):
        """
        :param: Any argument related to the PySC2 actions
        :return: List of executable PySC2 actions
        """
        pass

    @staticmethod
    def print_available_actions(obs, output=False):
        result = dict()
        for action in sorted(obs.observation.available_actions):
            result[action] = actions.FUNCTIONS[action]

        if output:
            logging.info("Available actions:")
            for key, value in result.items():
                logging.info(f"  {key}: {value}")
        return result

    def step(self):
        """
        :return: Executable PySC2 action or None if no actions are left.
        """
        # If an action is present, reduce action count. If not, return None
        if len(self.actions) > 0:
            self.action_count -= 1
            return self.actions.pop(0)
        else:
            return None

    def get_action_count(self):
        """
        :return: The number of actions that have not yet been executed in this composite action.
        """
        return self.action_count

    def get_priority(self):
        """
        :return: The assigned priority in the action queue
        """
        return self._priority
