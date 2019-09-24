from abc import abstractmethod, ABC


class Policy(ABC):
    """
    A policy is the highest level unit specific decider. It chooses which (composite) action a unit should do
    Examples may be picking up the pineapple until some attacker comes close then switching to shooting him
    Or using the close range attack until at low health, to then walk away and use long range attacks
    """

    def __init__(self, unittracker):
        self.unit_tracker = unittracker
        super().__init__()

    @abstractmethod
    def step(self, obs):
        """
        :return: List of executable action(s) (may be empty)
        """
        pass

    @staticmethod
    def is_available(obs):
        return True