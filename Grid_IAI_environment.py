import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    ## Initialise starting environment
    def __init__(self, env_params):
        # Set information about the gridworld
        self.height = env_params["height"]
        self.width = env_params["width"]
        self.grid = np.zeros((self.height, self.width)) - 1

        # Set locations for the food and the water
        self.food_location = env_params["food_location"]
        self.water_location = env_params["water_location"]

        # Set random start location for the agent
        Choose = True
        while Choose:
            self.current_location = (
                np.random.randint(0, self.height),
                np.random.randint(0, self.width),
            )

            if self.current_location == self.food_location:
                continue
            elif self.current_location == self.water_location:
                continue
            else:
                Choose = False

        # Set available actions
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    ## Put methods here:
    def get_available_actions(self):
        """Returns possible actions"""
        return self.actions

    def agent_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros((self.height, self.width))
        grid[self.current_location[0], self.current_location[1]] = 1

        return grid

    def print_locations(self, alpha=0.2, draw_text=True):
        grid = self.agent_on_map()

        cmap = plt.cm.get_cmap("Greens", 10)
        norm = plt.Normalize(np.min(grid), np.max(grid))
        rgba = cmap(norm(grid))

        for i in range(rgba.shape[0]):
            for j in range(rgba.shape[1]):
                rgba[i, j][-1] = alpha

        rgba[self.food_location] = 1.0, 0.5, 0.1, alpha
        rgba[self.water_location] = 0.0, 0.5, 0.8, alpha

        fig, ax = plt.subplots()
        im = ax.imshow(rgba)

        if draw_text:
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] == 1:
                        text = ax.text(
                            j, i, "Agent", ha="center", va="center", color="k"
                        )

            text = ax.text(
                self.food_location[0],
                self.food_location[1],
                "Food",
                ha="center",
                va="center",
                color="k",
            )
            text = ax.text(
                self.water_location[0],
                self.water_location[1],
                "Water",
                ha="center",
                va="center",
                color="k",
            )
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # plt.axis("off")
        plt.show()

    def make_step(self, action, agent):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.current_location

        if not agent.food_level == agent.min_food:
            agent.food_level -= 1

            if agent.food_level <= agent.min_food:
                agent.food_level = agent.min_food

        if not agent.water_level == agent.min_water:
            # agent.water_level -= 2
            agent.water_level -= 1

            if agent.water_level <= agent.min_water:
                agent.water_level = agent.min_water

        # UP
        if action == "UP":
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                self.current_location = last_location

            else:
                self.current_location = (
                    self.current_location[0] - 1,
                    self.current_location[1],
                )

        # DOWN
        elif action == "DOWN":
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.height - 1:
                self.current_location = last_location
            else:
                self.current_location = (
                    self.current_location[0] + 1,
                    self.current_location[1],
                )

        # LEFT
        elif action == "LEFT":
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0:
                self.current_location = last_location
            else:
                self.current_location = (
                    self.current_location[0],
                    self.current_location[1] - 1,
                )

        # RIGHT
        elif action == "RIGHT":
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.width - 1:
                self.current_location = last_location
            else:
                self.current_location = (
                    self.current_location[0],
                    self.current_location[1] + 1,
                )

        if self.check_state() == "EAT_FOOD":
            agent.food_level += int(0.5 * agent.max_food)

            if agent.food_level >= agent.max_food:
                agent.food_level = agent.max_food

        elif self.check_state() == "EAT_WATER":
            agent.water_level += int(0.5 * agent.max_water)

            if agent.water_level >= agent.max_water:
                agent.water_level = agent.max_water

    def check_state(self):
        """Check if the agent is in the food or water location, if so return 'EAT'"""
        if self.current_location == self.food_location:
            return "EAT_FOOD"

        elif self.current_location == self.water_location:
            return "EAT_WATER"

    def check_done(self, agent):
        if (
            agent.check_state()
            == "DEATH"
            # or self.check_state() == "EAT_FOOD"
            # or self.check_state() == "EAT_WATER"
        ):
            return "DONE"
        else:
            return None

    def reset(self, agent, env_params):
        # if (
        #     agent.check_state() == "DEATH"
        # ):
        # If game is in terminal state, game over and start next trial
        #     self.__init__(env_params)
        #     # agent.food_level = agent.set_point_food
        #     # agent.water_level = agent.set_point_water
        #     agent.food_level = np.random.randint(
        #         low=agent.min_food + 5, high=agent.max_food
        #     )
        #     agent.water_level = np.random.randint(
        #         low=agent.min_water + 5, high=agent.max_water
        #     )

        # elif self.check_state() == "EAT_FOOD":
        #     self.__init__(env_params)
        #     # agent.food_level = agent.set_point_food
        #     agent.food_level = np.random.randint(
        #         low=agent.min_food + 5, high=agent.max_food
        #     )

        # elif self.check_state() == "EAT_WATER":
        #     self.__init__(env_params)
        #     # agent.water_level = agent.set_point_water
        #     agent.water_level = np.random.randint(
        #         low=agent.min_water + 5, high=agent.max_water
        #     )

        self.__init__(env_params)
        agent.food_level = np.random.randint(low=agent.min_food, high=agent.max_food)
        agent.water_level = np.random.randint(low=agent.min_water, high=agent.max_water)
        # agent.food_level = agent.max_food
        # agent.water_level = agent.max_water

        game_over = True
        return game_over
