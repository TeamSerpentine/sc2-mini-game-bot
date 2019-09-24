import os
import time

from pysc2.agents import base_agent
from pysc2.lib import actions

from handin.strategy.serpentine import SerpentineS
from handin.unittracker.basic_tracker import BasicTracker

from collections import deque


# change to class name to the name of you Team
class Serpentine(base_agent.BaseAgent):
    """
    Core of the Hivemind bot structure.
    To make your bot, it should not be required to edit this class.
    Adjust the config above to your desired strategy and unit_tracker
        and run the part below to run hivemind
    """

    # ========= CONFIG ========

    # How many times as quick as human time the game will play
    # (At fastest. It may take too long for an action to complete to achieve this)
    # 0 for infinite
    PLAYBACK_SPEED_MULTIPLIER = 2.0

    SELECTED_UNITTRACKER = BasicTracker
    SELECTED_STRATEGY = SerpentineS

    # ======== ====== ========

    HUMAN_TIME_BETWEEN_ACTIONS = 0

    unit_tracker = None
    strategy = None
    action_queue = None
    last_time = -1

    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.TRAINING = False                   # Disables training

    def step(self, obs):
        """
        The function actually called by PySc2.
        Either takes an action from the built up queue or queries the strategy for new actions to do
        :param obs: observation from PySc2
        :return: Action to do (no_op action if none)
        """
        # First step of a map, empty queue and reset all trackers/strategies
        if obs.first():
            self.action_queue = deque()

            side = "top" if self.__class__.__name__.endswith("V2") else "bot"
            self.unit_tracker = self.SELECTED_UNITTRACKER(side=side)
            self.strategy = self.SELECTED_STRATEGY(self.TRAINING, self.unit_tracker)
            self.last_time = 0

        self.unit_tracker.update(obs)

        # Reduce playback speed to desired speed
        if self.PLAYBACK_SPEED_MULTIPLIER > 0:
            cur_time = time.time()
            time_between_actions = self.HUMAN_TIME_BETWEEN_ACTIONS / self.PLAYBACK_SPEED_MULTIPLIER
            time_since_last_action = cur_time - self.last_time
            if time_since_last_action < time_between_actions:
                time.sleep(time_between_actions - time_since_last_action)
            self.last_time = time.time()

        # If there are no actions left to do, get new actions from the strategy
        if len(self.action_queue) <= 0:
            self.action_queue.append(self.strategy.step(obs))

        # If there are actions to be left to do, execute them
        in_order = sorted(self.action_queue)
        self.action_queue = deque()
        self.action_queue.extend(in_order)

        if len(self.action_queue) > 0:
            # Get the first item from the queue
            action_type = self.action_queue[0]
            action = action_type.step()

            if action is None:
                import logging
                logging.info(f"Added a no op {self.action_queue}, {self.action_queue[0].get_action_count()}")
                action = actions.FUNCTIONS.no_op()

            if self.action_queue[0].get_action_count() <= 0:
                self.action_queue.popleft()
            return action
        else:
            # If there is still nothing to do, tell PySc2 that we will not do anything
            return actions.FUNCTIONS.no_op()


class SerpentineV2(Serpentine):
    pass


if __name__ == "__main__":
    from bash import RunBash

    # Get the module name and class name of your bot
    module_name = os.path.basename(__file__)[:-3]
    class_name_bot = "Serpentine"
    class_name_top = "SerpentineV2"

    # Get the script with the right module and bot name
    script = RunBash(module=module_name, bot_name=class_name_bot)
    script.run_bash_with_features()
