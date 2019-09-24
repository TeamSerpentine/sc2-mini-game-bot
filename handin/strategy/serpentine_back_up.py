import logging
import os
import random
import re
import pickle
import copy

from collections import namedtuple, deque

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input

from pysc2.lib import features  # This is probably useful

from main.hivemind.action.higher_level import *
from .strategy import Strategy

from main.hivemind.policy import movements_camera, attack, selection, movement_units

from keras import backend as K
import tensorflow as tf


FUNCTIONS = actions.FUNCTIONS

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class SerpentineS(Strategy):
    """
    The strategy is the most high level decider. One Strategy is picked in main and is used for the entire game
    The reason it is abstract is that others can implement their own strategy classes now while still using each
    other's policies and such
    A strategy should pick and/or switch the active global_policy
    """
    # Settings
    multiple_models = True
    load_from_save = True
    save_model_params = True

    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "network_params")
    SAVE_FILE = "Final6450"

    SAVE_PATH = os.path.join(SAVE_DIR, SAVE_FILE)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    def __init__(self, training, unit_tracker):
        super().__init__(unit_tracker)

        # Training variable
        self.TRAINING = training

        # Name
        self.name = self.__class__.__name__

        # Store loss
        self.loss = float("inf")

        # Strategy init
        self.unit_tracker = unit_tracker
        self.save_model_flag = self.save_model_params
        self.save_path = self.SAVE_PATH

        # Takes care of finding all policies
        modules_to_check = [attack, movement_units, movements_camera, selection]
        policies = self.get_all_policies(modules_to_check=modules_to_check)
        policy_names = namedtuple("policy_names", [*policies])
        policies_items = [policy(unit_tracker) for policy in policies.values()]
        self.policy = policy_names(*policies_items)

        # Workaround for using more than 1 models
        if self.multiple_models:
            self.allow_multiple_models()

        # Load the actual model and model parameters
        if self.load_from_save:
            try:
                self.model, self.model_params = self.load_model(self.SAVE_PATH)
            except ValueError as e:
                logging.warning(f"Catched error during loading: {e}, building new model instead")
                self.load_from_save = False

        # Load model or compile a new model

        if not self.load_from_save:
            self.model_params = dict(
                    input_shape=self.get_input_shape(),
                    nr_policies=len(policies),
                    policies=[*policies],
                    final_act='linear',
                    discount_factor=0.94,
                    epsilon=0.8,
                    epsilon_decay=0.99,  # Reaches 0.1 after ~200 games
                    epsilon_min=0.1,
                    batch_size=16,
                    epochs=2,
                    games_played=0
            )
            self.model = self.build(
                    self.model_params['input_shape'],
                    self.model_params['nr_policies'],
                    self.model_params['final_act'])

        # Training data and predictions
        self._observations = []  # List of all observations
        self._predictions = []  # List of all predicted q_values
        self._picked_actions = []  # List of all selected actions
        self._rewards = []  # List of all retrieved rewards

        # Observation queue
        self.combined_obs = deque(maxlen=self.get_input_shape()[-1])
        for _ in range(self.get_input_shape()[-1]):
            self.combined_obs.append([0] * self.get_input_shape()[0])

        # Camera position
        self.camera_set = False
        self.counter_steps = 0

    @staticmethod
    def get_input_shape():
        """ Is used as input shape of model and reshape dimensions.  """
        return 122, 4

    def step(self, obs):
        """ Given a PySC2 observation, find the action to execute """
        # Start training on the last frame and select all on first frame
        if obs.first():
            if self.policy.SelectArmyP.is_available(obs):
                return self.policy.SelectArmyP.step(obs)
            else:
                return NoOp(1)

        elif obs.last():
            if self.TRAINING:
                self.train()

            # Requires an 'output'
            return NoOp(1)

        # Move the camera, so we can see what happens
        if not self.camera_set and self.counter_steps > 10:
            self.camera_set = True
            return MoveCamera(1, 32, 32)
        self.counter_steps += 1

        # Construct a full observation using the unit tracker and obs data
        processed_obs = np.expand_dims(np.transpose(self.preprocess(obs)), axis=0)

        if processed_obs.shape == (1, 4):
            return NoOp(1)

        self._observations.append(np.squeeze(processed_obs, axis=0))
        q_values = self.model.predict(processed_obs)

        # Appending training data
        self._predictions.append(np.squeeze(q_values, axis=0))

        # Calculating reward for this time step
        delta_score = self.unit_tracker.get_delta_points()
        score = delta_score['ally_pineapple'] + 1.5 * delta_score['ally_kills'] - 0.5 * delta_score[
            'enemy_pineapple'] - 0.75 * delta_score['enemy_kills']

        self._rewards.append(score)

        # Selecting an action (and disable all unavailable actions)
        is_available = np.array([pol.is_available(obs) for pol in self.policy])
        is_unavailable = np.logical_not(is_available)
        adjusted_q_values = copy.deepcopy(q_values)
        adjusted_q_values[np.expand_dims(is_unavailable, axis=0)] = -float('inf')
        action = np.argmax(adjusted_q_values)

        # Random action selection
        if self.TRAINING:
            if self.model_params['epsilon'] >= random.random():
                possible_actions = np.where(is_available)[0]
                action = random.choice(possible_actions)

        # Add selected action to the queue and the list
        self._picked_actions.append(action)
        return self.policy[int(action)].step(obs)

    def preprocess(self, obs):
        """ Given a PySC2 observation, calculate a processed observation
         to be put into the neural net """

        new_line = self.get_model_input(self.unit_tracker)
        self.combined_obs.append(new_line)
        obs = np.array(list(self.combined_obs))
        return obs

    def train(self):
        """ Train the model """
        # Retrieve target q_values by looping back through the game
        target_predictions = self._determine_q_values()

        # Fit observations to target_predictions
        history = self.model.fit(np.array(self._observations), target_predictions, self.model_params['batch_size'], self.model_params['epochs'])
        self.loss = history.history['loss']

        # Clear the history
        self._observations = []
        self._predictions = []
        self._picked_actions = []
        self._rewards = []

        # Calculate new epsilon
        epsilon = self.model_params["epsilon"]
        epsilon_decay = self.model_params["epsilon_decay"]
        epsilon_min = self.model_params["epsilon_min"]
        self.model_params["epsilon"] = max(epsilon * epsilon_decay, epsilon_min)

        # Update number of games played
        if self.model_params.get("games_played", None) is None:
            self.model_params["games_played"] = 0
        self.model_params["games_played"] += 1

        # Store model if not training
        if self.save_model_flag and self.TRAINING:
            self.save_model(self.model, self.model_params, self.save_path)

    def _determine_q_values(self):
        """"Loop backwards through rewards to retrieve corrected predictions"""
        rewards = np.array(self._rewards)               # Rewards in a Numpy array
        target_predicts = np.array(self._predictions)   # Ground truth in a Numpy array
        target_for_action = np.zeros(rewards.shape)     # Predicted q_values for the selected action

        # Last target equals the reward of the last action
        nr_actions = len(rewards)
        target_for_action[nr_actions-1] = rewards[nr_actions-1]

        # Loop through history
        for idx in range(nr_actions - 2, -1, -1):
            selected_action = self._picked_actions[idx]
            target_for_action[idx] = self.model_params['discount_factor'] * target_for_action[idx + 1] + rewards[idx]
            target_predicts[idx, selected_action] = target_for_action[idx]
        return target_predicts

    def get_model_input(self, unit_tracker):
        """  Make all inputs (5,) """
        # Ally pineapples, kills, enemy pineapples, kills, padding
        stats = np.array([*unit_tracker.get_delta_points().values(), 0])

        # Army count
        army_count = np.array([*unit_tracker.army_count, *unit_tracker.unit_count_on_map, 0])

        # Respawn timers
        spawn_allied = unit_tracker.allied_respawn_timers
        spawn_enemy = unit_tracker.enemy_respawn_timers

        spawn_allied = np.array([*spawn_allied, *[0 for _ in range(5 - len(spawn_allied))]])
        spawn_enemy = np.array([*spawn_enemy, *[0 for _ in range(5 - len(spawn_enemy))]])
        alive = np.array(self.unit_tracker.get_alive())

        # Scores and pineapple timer
        scores_timer = np.array([*unit_tracker.scores,
                                 *unit_tracker.delta_score_own,
                                 *unit_tracker.get_pineapple_timer()])

        # Get the average location of all units on the field
        avg_locations = np.array(self.unit_tracker.get_avg_locations())
        all_locations = np.array(self.unit_tracker.get_all_locations())

        unit_info = []
        for player in ["unit_info_own", "unit_info_enemy"]:
            for each in ["x", "y", "is_selected", "hp", "shield"]:
                information = getattr(unit_tracker, player)[each]
                unit_info.append(np.array(information, dtype=np.float32)[:5])

        obs = np.concatenate([stats.flatten(), army_count.flatten(),
                        spawn_allied.flatten(), spawn_enemy.flatten(), alive.flatten(),
                        scores_timer.flatten(), *unit_info,
                        avg_locations.flatten(), all_locations.flatten()]).ravel()

        return list(obs)

    def get_output_shape(self):
        return len(self.model_params['policies'])

    def save_model(self, model, model_params, path):
        """ Save the model and the parameters to files """

        # Do not save when we do not train
        if not self.TRAINING:
            return

        model_path = path + ".h5"
        model_params_path = path + ".pickle"

        if self.model_params['games_played'] % 25 == 0:
            model.save(path + str(self.model_params['games_played']) + ".h5")
            with open(path + str(self.model_params['games_played']) + ".pickle", "wb") as model_params_file:
                pickle.dump(model_params, model_params_file)

        model.save(model_path)
        with open(model_params_path, "wb") as model_params_file:
            pickle.dump(model_params, model_params_file)

        info = "--------------- %s ---------------"
        output = "\n".join([f"  {key}:  {value}" for key, value in model_params.items()])
        logging.info(info % "Saved Model" + f"\n{output}\n")

    @staticmethod
    def get_all_policies(modules_to_check: list):
        """ Returns all the available policies in modules_to_checks.  """
        exclude = []

        include = ["AttackClosest", "AttackLowest",
                   "CamCenterFriendly", "CamPineapple",
                   "LocPineappleP", "LocFlankLeftP", "LocFlankRightP", "LocPinFriendlyP", "LocPinEnemyP",
                   "SpecialAttackP", "SelectArmyP"]

        results = dict()
        for module in modules_to_check:
            for item in dir(module):
                # Check python main file
                if not re.match("[^__]", item):
                    continue

                #  Check if it is connected to a module
                if not hasattr(getattr(module, item), "__module__"):
                    continue

                # Check if it is from the checked module
                if not getattr(module, item).__module__ == module.__name__:
                    continue

                # Check if it should be excluded
                if exclude and item in exclude:
                    continue

                # Check if it should be excluded
                if include and item not in include:
                    continue

                results[item] = getattr(module, item)
        return results

    def allow_multiple_models(self):
        """ Limits GPU use, so multiple models can coincide on 1 GPU. """
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.40
        session = tf.Session(config=config)
        K.set_session(session)

    def build(self, input_shape, nr_policies, final_act="linear"):
        """ Build the entire model """
        inputs = Input(input_shape)
        function = Flatten()(inputs)

        # Dense layer 1
        function = Dense(256)(function)
        function = Activation('relu')(function)
        # function = Dropout(0.5)(function)

        # Dense layer 2
        function = Dense(256)(function)
        function = Activation('relu')(function)
        # function = Dropout(0.5)(function)

        # Dense layer 3
        function = Dense(256)(function)
        function = Activation('relu')(function)

        # Dense layer 4
        function = Dense(256)(function)
        function = Activation('relu')(function)

        # Dense layer 5
        function = Dense(256)(function)
        function = Activation('relu')(function)

        # Dense layer 6
        function = Dense(256)(function)
        function = Activation('relu')(function)

        # Output layer
        function = Dense(nr_policies)(function)
        function = Activation(final_act, name="function_output")(function)

        model = Model(inputs=inputs, outputs=function)
        model.compile(optimizer="adam", loss="mse")
        return model

    def load_model(self, path):
        """ Load a model and its parameters from files """
        model_path = path + ".h5"
        model_params_path = path + ".pickle"

        info = "--------------- %s ---------------"
        logging.warning(info % "Creating Model".ljust(11).rjust(11))
        if os.path.isfile(model_path) and os.path.isfile(model_params_path):
            try:
                with open(model_params_path, "rb") as model_params_file:
                    model_params = pickle.load(model_params_file)

                    model = self.build(
                            model_params['input_shape'],
                            model_params['nr_policies'],
                            model_params['final_act'])

                    model.load_weights(model_path)

                output = "\n".join([f"  {key}:  {value}" for key, value in model_params.items()])
                logging.warning(info % "Loaded model from disk" + f"\n{output}\n")
            except OSError:
                raise ValueError("Unable to load model?")
        else:
            raise ValueError(f"Can't locate: '{path}'")
        return model, model_params



if __name__ == "__main__":
    # Sanity check for model
    unit_tracker = type('unit_tracker', (), dict(bottom=False))
    agent = SerpentineS(training=False, unit_tracker=unit_tracker)

    # Expected output:
    assert agent.model_params['input_shape'] == (122, 4), "Model not loaded correctly"
    assert agent.model_params['nr_policies'] == 11, "Model not loaded correctly"
    assert agent.model_params['games_played'] == 1025, "Model not loaded correctly"
    assert agent.TRAINING is False, "Training wasn't disable correctly"
