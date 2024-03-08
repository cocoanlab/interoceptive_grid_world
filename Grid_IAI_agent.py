import os
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Q_Agent:
    # Intialise
    def __init__(
        self,
        environment,
        agent_params,
        save_params=None,
        random_action=False,
    ):
        self.environment = environment
        self.random_action = random_action
        self.save_params = save_params

        self.epsilon = agent_params["epsilon"]
        self.epsilon_end = agent_params["epsilon_end"]
        self.alpha = agent_params["alpha"]
        self.gamma = agent_params["gamma"]

        self.food_level = agent_params["max_food"]
        self.max_food = agent_params["max_food"]
        self.min_food = agent_params["min_food"]
        self.set_point_food = agent_params["set_point_food"]

        self.water_level = agent_params["max_water"]
        self.max_water = agent_params["max_water"]
        self.min_water = agent_params["min_water"]
        self.set_point_water = agent_params["set_point_water"]

        self.q_table = dict()  # Store all Q-values in dictionary of dictionaries
        for x in range(environment.height):
            for y in range(environment.width):
                for f in range(agent_params["max_food"] + 1):
                    for w in range(agent_params["max_water"] + 1):
                        self.q_table[(x, y, f, w)] = {
                            "UP": 0,
                            "DOWN": 0,
                            "LEFT": 0,
                            "RIGHT": 0,
                        }  # Populate sub-dictionary with zero values for possible moves

    def get_reward(self, old_state, new_state):
        """Returns the reward for an input position"""

        reward = -((self.set_point_food - new_state[2]) ** 2) - ((self.set_point_water - new_state[3]) ** 2)  # type: ignore
        reward -= -((self.set_point_food - old_state[2]) ** 2) - ((self.set_point_water - old_state[3]) ** 2)  # type: ignore

        if self.food_level == self.min_food:
            # reward *= 10
            reward = -10
        elif self.food_level == self.max_food:
            reward = -10
        elif self.water_level == self.min_water:
            # reward *= 10
            reward = -10
        elif self.water_level == self.max_water:
            reward = -10
        # else:
        #     reward = 0

        reward = 0.01 * reward

        return reward

    def check_state(self):
        if self.food_level == self.min_food:
            return "DEATH"
        elif self.food_level == self.max_food:
            return "DEATH"
        elif self.water_level == self.min_water:
            return "DEATH"
        elif self.water_level == self.max_water:
            return "DEATH"

    def choose_action(self, available_actions, max_action=False, verbose=False):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""

        if (
            not max_action
            and np.random.uniform(0, 1) < self.epsilon
            or self.random_action
        ):
            action = available_actions[np.random.randint(0, len(available_actions))]

            if verbose:
                print("Random Action by epsilon")
        else:
            q_values_of_state = self.q_table[
                tuple(
                    (
                        self.environment.current_location[0],
                        self.environment.current_location[1],
                        self.food_level,
                        self.water_level,
                    )
                )
            ]

            maxValue = max(q_values_of_state.values())
            action = np.random.choice(
                [k for k, v in q_values_of_state.items() if v == maxValue]
            )
            if verbose:
                print("Greedy Action")

        return action

    def learn(self, old_state, reward, new_state, action):
        """Updates the Q-value table using Q-learning"""

        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]

        self.q_table[old_state][action] = current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state - current_q_value)  # type: ignore

        # if self.environment.check_done(self) == "DONE":
        #     for k in self.q_table[new_state].keys():
        #         self.q_table[new_state][k] = reward

    def save_model(self, trial):
        _save = os.path.join(self.save_params["savedir"], self.save_params["tag"])

        if not os.path.exists(_save):
            os.makedirs(_save)

        _q_file = os.path.join(_save, "Trial_{:07d}_q_table.pkl".format(trial))
        with open(_q_file, "wb") as f:
            pickle.dump(self.q_table, f)
