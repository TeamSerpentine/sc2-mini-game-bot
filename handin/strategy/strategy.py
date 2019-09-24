from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    The strategy is the most high level decider. One Strategy is picked in main and is used for the entire game
    The reason it is abstract is that others can implement their own strategy classes now while still using each
        other's policies and such
    A strategy should pick and/or switch the active global_policy
    """

    def __init__(self, unit_tracker):
        self.unit_tracker = unit_tracker

    @abstractmethod
    def step(self, obs):
        """
        :return: List of executable action(s) (may be empty)
        """
        return
