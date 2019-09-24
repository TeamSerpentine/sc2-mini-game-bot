from abc import ABC, abstractmethod


class UnitTracker(ABC):
    """
    The unitTracker is responsible for keeping track of where different agents are.
    It stores additional information about the current game state.
    The reason it is abstract is that now people can make multiple different unit trackers while reusing each other's
        policies and/or actions and such.
    """

    @abstractmethod
    def update(self, obs):
        """
        Called every before the AI makes it decision. This function should update all information about the game.
        """
