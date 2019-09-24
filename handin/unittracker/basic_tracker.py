import json
import numpy as np
import os
import pandas as pd

from copy import deepcopy

from .unit_tracker import UnitTracker


class BasicTracker(UnitTracker):
    """
    The unitTracker is responsible for keeping track of where different agents are.
    It stores additional information about the current game state.
    The reason it is abstract is that now people can make multiple different unit trackers while reusing each other's
        policies and/or actions and such.
    """

    def __init__(self, side='bot'):
        """
        Constructor
        :param: side: The side on which the team is located
        """
        self.debug_mode = False
        self.log_everything = False
        if self.log_everything:
            if not os.path.exists("unit_tracker_log"):
                os.mkdir("unit_tracker_log")
            self.output_log = open(os.path.join('unit_tracker_log', 'output_log.txt'), 'w+')
        if self.debug_mode:
            if not os.path.exists("unit_tracker_log"):
                os.mkdir("unit_tracker_log")
            self.steps = 0
            self.file = open(os.path.join('unit_tracker_log', 'unittracker_log.txt'), 'w+')
            self.tracker_log = open(os.path.join('unit_tracker_log', 'cam_location_log.txt'), 'w+')
            self.tracker_log2 = open(os.path.join('unit_tracker_log', 'tracker_selection.txt'), 'w+')
            self.match_log = open(os.path.join('unit_tracker_log', 'match_log.txt'), 'w+')
            self.result_log = open(os.path.join('unit_tracker_log', 'result_log.txt'), 'w+')

        self.bottom = side == 'bot'

        # Initial conditions for the units
        top_units = [
            {'allied': not self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 7, "is_selected": 0,
             "weapon_cd": 0, "found": False},
            {'allied': not self.bottom, 'hp': 1, 'shield': 1, 'x': 31, 'y': 8, "is_selected": 0,
             "weapon_cd": 0, "found": False},
            {'allied': not self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 8, "is_selected": 0,
             "weapon_cd": 0, "found": False},
            {'allied': not self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 8, "is_selected": 0,
             "weapon_cd": 0, "found": False},
            {'allied': not self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 8, "is_selected": 0,
             "weapon_cd": 0, "found": False},
        ]
        bot_units = [
            {'allied': self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 55, "is_selected": 0, "weapon_cd": 0,
             "found": False},
            {'allied': self.bottom, 'hp': 1, 'shield': 1, 'x': 31, 'y': 56, "is_selected": 0, "weapon_cd": 0,
             "found": False},
            {'allied': self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 56, "is_selected": 0, "weapon_cd": 0,
             "found": False},
            {'allied': self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 56, "is_selected": 0, "weapon_cd": 0,
             "found": False},
            {'allied': self.bottom, 'hp': 1, 'shield': 1, 'x': 32, 'y': 56, "is_selected": 0, "weapon_cd": 0,
             "found": False}
        ]

        team_own = [{'points': 0, 'army_count': 5}]
        team_enemy = [{'points': 0, 'army_count': 5}]

        self.team_info_own = pd.DataFrame(team_own)
        self.team_info_enemy = pd.DataFrame(team_enemy)
        if self.bottom:
            self.last_allied_center = np.array([56, 32])
            self.last_enemy_center = np.array([8, 32])
            self.unit_info_own = pd.DataFrame(bot_units)
            self.unit_info_enemy = pd.DataFrame(top_units)
            self.camera_pos = [32, 56]
        else:
            self.last_allied_center = np.array([8, 32])
            self.last_enemy_center = np.array([56, 32])
            self.unit_info_own = pd.DataFrame(top_units)
            self.unit_info_enemy = pd.DataFrame(bot_units)
            self.camera_pos = [32, 8]

        # Important counters
        self.pineapple_timer = 0                # Respawn timer list. 0 indicates the pineapple is present
        self.army_count = [5, 5]                # Unit count measured by inferring from points [allied, enemy]
        self.enemy_respawn_timers = []          # Respawn timer list. Each entry stands for one dead units timer
        self.allied_respawn_timers = []         # Respawn timer list. Each entry stands for one dead units timer
        self.scores = [0, 0]                    # Scores in this frame [allied, enemy]
        self.delta_score_own = [0, 0]           # Score breakdown for this timestep for our team [kills, pineapple]
        self.delta_score_enemy = [0, 0]         # Score breakdown for this timestep for the enemy [kills, pineapple]
        self.unit_count_on_map = [3, 3]         # Unit count found on the minimap [allied, enemy]

        # Info related to previous frames
        self.last_scores = [0, 0]               # Scores in the previous frame [allied, enemy]
        self.last_unit_count_on_map = [3, 3]    # Unit count found on the minimap [allied, enemy]

        # Base locations
        if self.bottom:
            self.allied_base = [56, 32]
            self.enemy_base = [8, 32]
        else:
            self.allied_base = [8, 32]
            self.enemy_base = [56, 32]

        super().__init__()

    def update(self, obs):
        """
        Called every before the AI makes it decision. This function should update all information about the game.
        """
        # Remove unnecessary info
        obs = dict(deepcopy(obs.observation))
        obs = dict((k, obs[k]) for k in (
            'single_select', 'multi_select', 'feature_screen', 'feature_minimap', 'last_actions',
            'game_loop', 'player', 'control_groups', 'feature_units', 'available_actions'))

        # Convrting to regular numpy arrays (Easier display)
        for key, value in obs.items():
            try:
                obs[key] = np.array(value)
            except ValueError:
                print('Can not convert:', key)

        # Feature Units
        if len(obs['feature_units']) > 0:
            if len(obs['feature_units'][0]) == 26:
                screen_info = pd.DataFrame(obs['feature_units'],
                                           columns=["type", "team", "hp", "shield", "energy", "cargo_space",
                                                    "build_prog", "hp_ratio", "shield_ratio", "energy_ratio",
                                                    "display_type", "owner", "x", "y", "facing", "radius", "cloak",
                                                    "is_selected", "is_blip", "is_powered", "minerals", "vespene",
                                                    "cargo_max", "cur_harvesters", "ideal_harvesters", "weapon_cd"])
            else:
                screen_info = pd.DataFrame(obs['feature_units'],
                                           columns=["type", "team", "hp", "shield", "energy", "cargo_space",
                                                    "build_prog", "hp_ratio", "shield_ratio", "energy_ratio",
                                                    "display_type", "owner", "x", "y", "facing", "radius", "cloak",
                                                    "is_selected", "is_blip", "is_powered", "minerals", "vespene",
                                                    "cargo_max", "cur_harvesters", "ideal_harvesters", "weapon_cd",
                                                    "order_length", "tag"])

            screen_info = screen_info.drop(
                columns=['energy', "cargo_space", "build_prog", "energy_ratio", "display_type", "owner",
                         "facing", "cloak", "is_blip", "is_powered", "minerals", "vespene", "cargo_max",
                         "cur_harvesters", "ideal_harvesters"])
        else:
            screen_info = pd.DataFrame(
                columns=["type", "team", "hp", "shield", "hp_ratio", "shield_ratio", "x", "y", "is_selected",
                         "weapon_cd"])

        # Normalize the shields and hp for the units (0 to 1 range)
        screen_info['hp_ratio'] = screen_info['hp_ratio'].apply(lambda x: float(x) / 255)
        screen_info['shield_ratio'] = screen_info['shield_ratio'].apply(lambda x: float(x) / 255)

        # Minimap info
        obs['map_type'] = obs['feature_minimap'][4, :, :]  # Red = 1, Blue = 2, Neut = 16
        obs['map_team'] = obs['feature_minimap'][5, :, :]  # Red = 1, Blue = 4, Neut = 3
        obs['map_own'] = obs['feature_minimap'][6, :, :]  # Own = 1 (Only selected untis)

        # Update scores
        self._update_score_breakdown(obs, screen_info)
        self.scores = [obs['player'][1], obs['player'][2]]

        # Update timers
        self.allied_respawn_timers = [x - 0.5 for x in self.allied_respawn_timers]
        self.enemy_respawn_timers = [x - 0.5 for x in self.enemy_respawn_timers]
        self.pineapple_timer = max(self.pineapple_timer - 0.5, 0)  # Never decrease below 0

        # Erase respawn timer once they return
        self.allied_respawn_timers = [item for item in self.allied_respawn_timers if item >= 0]
        self.enemy_respawn_timers = [item for item in self.enemy_respawn_timers if item >= 0]
        self.army_count = [obs['player'][8], 5 - len(self.enemy_respawn_timers)]

        # Update counters
        self.unit_count_on_map = [len(np.where(obs['map_team'] == 1)[0]), len(np.where(obs['map_team'] == 4)[0])]

        # Getting the location on the field
        enemy_units_minimap = np.where(obs['map_team'] == 4)
        allied_units_minimap = np.where(obs['map_team'] == 1)

        self.enemy_locations = \
            [*np.array([*enemy_units_minimap[0], *[self.enemy_base[0] for _ in range(5 - len(enemy_units_minimap[0]))]]),
             *np.array([*enemy_units_minimap[1], *[self.enemy_base[1] for _ in range(5 - len(enemy_units_minimap[1]))]])
             ]
        self.enemy_visible = len(enemy_units_minimap[0]) * [1] + (5-len(enemy_units_minimap[0])) * [0]

        self.allied_locations = \
            [*np.array([*allied_units_minimap[0], *[self.allied_base[0] for _ in range(5 - len(allied_units_minimap[0]))]]),
             *np.array([*allied_units_minimap[1], *[self.allied_base[1] for _ in range(5 - len(allied_units_minimap[1]))]])
              ]
        self.allied_visible = len(allied_units_minimap[0]) * [1] + (5 - len(allied_units_minimap[0])) * [0]

        # Getting the average location on the field
        if len(enemy_units_minimap[0]) == 0:
            enemy_units_minimap = self.enemy_base
        if len(allied_units_minimap[0]) == 0:
            allied_units_minimap = self.allied_base

        self.enemy_center = np.array([np.mean(enemy_units_minimap[0]), np.mean(enemy_units_minimap[1])])
        self.allied_center = np.array([np.mean(allied_units_minimap[0]), np.mean(allied_units_minimap[1])])
        self.battle_field_center = (self.enemy_center+self.allied_center)/2


        # Update unit info
        self._update_global_unit_info(obs, screen_info)

        self.obs = obs
        self.screen_info = screen_info

        if self.log_everything:
            self.output_log.write(f"delta points: {json.dumps(self.get_delta_points())}\n"
                                  f"army_count: {self.army_count} "
                                  f"map_army_count: {self.unit_count_on_map}\n"
                                  f"allied_respawn: {self.allied_respawn_timers} "
                                  f"enemy_respawn: {self.enemy_respawn_timers}\n"
                                  f"pineapple_timer: {self.get_pineapple_timer()}\n"
                                  f"scores: {self.scores} "
                                  f"allied_delta: {self.delta_score_own} "
                                  f"enemy_delta: {self.delta_score_enemy}\n"
                                  f"alive: {self.get_alive()}\n "
                                  f"locations: {self.get_all_locations()}\n "
                                  f"battle_center: {self.battle_field_center}"
                                  f"allied_center: {self.allied_center}"
                                  f"enemy_center: {self.enemy_center}\n\n")

        if self.debug_mode:
            self._debug(obs, screen_info)

    def get_all_locations(self):
        return self.allied_locations + self.enemy_locations + self.allied_visible + self.enemy_visible

    def get_avg_locations(self):
        return self.allied_center, self.enemy_center, self.battle_field_center

    def get_alive(self):
        allied = self.army_count[0] * [1] + (5-self.army_count[0]) * [0]
        enemy = self.army_count[1] * [1] + (5 - self.army_count[1]) * [0]
        return allied + enemy

    def get_delta_points(self):
        """
        :return: Dictionary containing why certain points were gained this round
        """
        return {"ally_pineapple": self.delta_score_own[0] / 10,
                "ally_kills": self.delta_score_own[1] / 5,
                "enemy_pineapple": self.delta_score_enemy[0] / 10,
                "enemy_kills": self.delta_score_enemy[1] / 5}

    def get_unit_info(self):
        """
        :return Information about all the units on the map in a Pandas Dataframe
        """
        return {'allied_info': self.unit_info_own,              # Contains bugs upon killing
                'enemy_info': self.unit_info_enemy,             # Contains bugs upon killing

                # Pretty solid - # Unit count measured by inferring from points [allied, enemy]
                'army_count': self.army_count,

                # Rock solid - # Unit count found on the minimap [allied, enemy]
                'map_army_count': self.unit_count_on_map,

                # Pretty solid - Respawn timer list. Each entry stands for one dead units timer
                'allied_respawn': self.allied_respawn_timers,

                # Pretty solid - Respawn timer list. Each entry stands for one dead units timer
                'enemy_respawn': self.enemy_respawn_timers}

    def get_pineapple_timer(self):
        """
        :return: Real time seconds until pineapple respawn
        """
        return 1 if self.pineapple_timer <= 0 else 0, self.pineapple_timer  # Infered from score changes

    def get_score_info(self):
        """
        :return Score information, actual scores, score breakdown for ally and enemy
        """
        return {"scores": self.scores,
                "allied_delta": self.delta_score_own,
                "enemy_delta": self.delta_score_enemy}

    def get_screen_info(self):
        """
        :return Information about both teams in a Pandas Dataframe
        """
        # TODO: Return on screen averages: HP Weapon cooldown selected units
        # TODO: Also return cluster factor and dot count for the both teams
        return None

    def set_camera_location(self, xy):
        """
        :param xy: A list containing the x and y coordinates of the last camera command
        """
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            self.camera_pos = xy
        else:
            raise ValueError("Incorrect xy coordinates")

    def _update_score_breakdown(self, obs, screen_info):
        # Pineapple / unit death detection
        # Check if the score of our team has increased
        self.delta_score_own = [0, 0]
        if obs['player'][1] > self.last_scores[0]:

            # It can only be a pineapple if the difference is at least 10
            if obs['player'][1] - self.last_scores[0] >= 10 and self.pineapple_timer <= 0:

                # Pineapple if it has now disappeared or if the enemy has more than 3 units visible FIXME: Correct value
                if (abs(self.camera_pos[0] - 32) < 5 and abs(self.camera_pos[1] - 32) < 5 and 442 not in
                        screen_info['type']) or len(np.where(obs['map_team'] == 4)) > 3:

                    # Set the pineapple respawn timer,
                    self.delta_score_own[0] = 10  # Add pineapple scores
                    self.pineapple_timer = 10.1

                    # Set the unit respawn timer
                    nr_units_killed = max(((obs['player'][1] - self.last_scores[0] - 10)/5), 0)
                    self.delta_score_own[1] = int(nr_units_killed) * 5
                    self.enemy_respawn_timers += int(nr_units_killed) * [12]

                elif self.last_unit_count_on_map[1] > len(np.where(obs['map_team'] == 4)):
                    nr_units_killed = max(((obs['player'][1] - self.last_scores[0])/5), 0)
                    self.delta_score_own[1] = int(nr_units_killed) * 5
                    self.enemy_respawn_timers += int(nr_units_killed) * [12]
                else:
                    self.delta_score_own[0] = 10  # Add pineapple scores
                    self.pineapple_timer = 10.1

                    # Set the unit respawn timer
                    nr_units_killed = max(((obs['player'][1] - self.last_scores[0] - 10)/5), 0)
                    self.delta_score_own[1] = int(nr_units_killed) * 5
                    self.enemy_respawn_timers += int(nr_units_killed) * [12]
            else:
                nr_units_killed = max(((obs['player'][1] - self.last_scores[0])/5), 0)
                self.delta_score_own[1] = int(nr_units_killed) * 5
                self.enemy_respawn_timers += int(nr_units_killed) * [12]

        # Pineapple / unit death detection
        # Check if the score of the enemy team has increased
        total_point_increase = obs['player'][2] - self.last_scores[1]
        nr_units_killed = max((self.army_count[0] - obs['player'][8]), 0)

        self.delta_score_enemy = [total_point_increase - 5 * int(nr_units_killed), 5 * int(nr_units_killed)]

        # Set respawn timers
        if total_point_increase - 5 * int(nr_units_killed) >= 10:
            self.pineapple_timer = 10.1
        self.allied_respawn_timers += int(nr_units_killed) * [11.5]

        self.last_scores = [obs['player'][1], obs['player'][2]]  # Only used in next frame during update

    def _update_global_unit_info(self, obs, all_screen_info):
        """
        :param obs: Modified observations
        :return:
        """
        screen_info = deepcopy(all_screen_info)
        self._update_for_a_team(obs, screen_info.copy(), 1, self.unit_info_own)

        screen_info = deepcopy(all_screen_info)
        self._update_for_a_team(obs, screen_info.copy(), 4, self.unit_info_enemy)

        if self.debug_mode:
            self.result_log.write(self.unit_info_own.to_string() + '\n\n')
            self.tracker_log2.write(screen_info.to_string() + '\n')

        # TODO: Save on screen info + Dot count enemy ally + cluster factor + army count
        # TODO: Number of allied / enemy units on screen
        print('done')

    def _update_for_a_team(self, obs, screen_info, team_number, global_fields):
        screen_info = screen_info.loc[screen_info['team'] == team_number].copy()

        screen_info['map_x'] = self.camera_pos[0] + (screen_info['x'] - 39) / 4.5
        screen_info['map_y'] = self.camera_pos[1] + (screen_info['y'] - 39) / 4.5

        screen_info['map_x'] = screen_info['map_x'].apply(lambda value: round(value))
        screen_info['map_y'] = screen_info['map_y'].apply(lambda value: round(value))

        # Retrieve minimap coordinates of allied units
        y_mm, x_mm = np.where(obs['map_team'] == team_number)

        if len(y_mm) == 0:
            return

        euc_dist = np.zeros((len(screen_info), len(x_mm)))

        # R
        idx_su = 0
        for row in screen_info.iterrows():
            for idx_mm, (x, y) in enumerate(zip(x_mm, y_mm)):
                euc_dist[idx_su, idx_mm] = ((row[1]['map_x'] - x) ** 2 + (row[1]['map_y'] - y) ** 2) ** 0.5
                if self.debug_mode:
                    self.tracker_log.write(str(idx_su) + ' ' + str(idx_mm) + ': MX:' + str(row[1]['map_x']) + ' X:' +
                                           str(x) + ' MY:' + str(row[1]['map_y']) + ' ' + str(y) + ' Y:' +
                                           str(euc_dist[idx_su, idx_mm]) + '\n')
            idx_su += 1

        # Attach screen units to unit global list
        screen_info['matched_x'] = [-1]*len(screen_info)
        screen_info['matched_y'] = [-1]*len(screen_info)
        for idx_su in range(len(screen_info)):

            # Find minimap point where distance difference is the smallest
            idx_mm = np.where(euc_dist[idx_su, :] == min(euc_dist[idx_su, :]))[0][0]
            x = x_mm[idx_mm]
            y = y_mm[idx_mm]
            screen_info.at[idx_su, 'matched_x'] = x
            screen_info.at[idx_su, 'matched_y'] = y

            if screen_info.at[idx_su, 'is_selected'] == 1:
                print('Stop')

            # Find the best matching unit from the list
            idx_list = 0
            min_loss = 1000
            min_idx = -1
            for row in global_fields.iterrows():
                if not row[1]['found']:
                    euc = ((row[1]['x'] - x) ** 2 + (row[1]['y'] - y) ** 2) ** 0.5
                    stats = ((screen_info.at[idx_su, 'hp_ratio'] - row[1]['hp']) ** 2 + (
                                screen_info.at[idx_su, 'shield_ratio'] - row[1]['shield']) ** 2) ** 0.5
                    loss = euc + stats
                    min_idx = idx_list if loss < min_loss else min_idx
                    min_loss = loss if loss < min_loss else min_loss
                    idx_list += 1

            # Updating info
            global_fields.at[min_idx, 'found'] = True
            global_fields.at[min_idx, 'x'] = x
            global_fields.at[min_idx, 'y'] = y
            global_fields.at[min_idx, 'hp'] = screen_info.at[idx_su, 'hp_ratio']
            global_fields.at[min_idx, 'shield'] = screen_info.at[idx_su, 'shield_ratio']
            global_fields.at[min_idx, 'is_selected'] = screen_info.at[idx_su, 'is_selected']
            global_fields.at[min_idx, 'weapon_cd'] = screen_info.at[idx_su, 'weapon_cd']
            # TODO: Update shield_cd, Respawn timer

        # Attach the other units to the found coordinates
        for idx_un in range(5):
            if not global_fields.at[idx_un, "found"]:
                euc_dist = np.zeros((len(x_mm)))
                for idx_mm, (x, y) in enumerate(zip(x_mm, y_mm)):
                    euc_dist[idx_mm] = ((global_fields.at[idx_un, 'x'] - x) ** 2 + (
                                global_fields.at[idx_un, 'y'] - y) ** 2) ** 0.5
                idx_mm = np.where(euc_dist == min(euc_dist))[0][0]

                global_fields.at[idx_un, 'x'] = x_mm[idx_mm]
                global_fields.at[idx_un, 'y'] = y_mm[idx_mm]

            # Reset for the next run
            global_fields.at[idx_un, 'found'] = False

        # TODO: Deal with dying

    def _debug(self, obs, screen_info):

        x, y = np.where(obs['map_team'] != 0)

        if len(x) > 0:
            self.file.write(str(x) + ";" + str(y) + ";" + str(len(x)) + ';\n')
            self.file.write(str(self.steps) + ": " + screen_info.to_string() + "\n\n")
        else:
            self.file.write(str(self.steps) + ": " + 'No selection\n')

        self.tracker_log.write(str(self.camera_pos))
        self.steps += 1
        # if self.steps > 50:
        #     pyplot.imshow(obs['map_team'])
        #     pyplot.show()
        #     self.file.close()
        #     self.tracker_log.close()
        #     self.tracker_log2.close()
        #     self.match_log.close()
        #     self.result_log.close()
        #     print("Finished Properly")
        #     exit()
