# %%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Grid_IAI_environment import GridWorld
from Grid_IAI_agent import Q_Agent


def train(
    environment,
    agent,
    env_params,
    episodes=500,
    max_steps_per_episode=1000,
    exploration_episode_ratio=0.1,
):
    """The train function runs iterations and updates Q-values if desired."""
    reward_per_episode = np.zeros(episodes)  # Initialise performance log
    step_per_episode = np.zeros(episodes)
    epsilon_per_episode = np.zeros(episodes)

    exploration_episodes = int(episodes * exploration_episode_ratio)

    for episode in tqdm(range(episodes)):  # Run episodes
        cumulative_reward = 0  # Initialise values of each game
        step = 0
        game_over = False

        agent.epsilon -= agent.epsilon / exploration_episodes
        if agent.epsilon < agent.epsilon_end:
            agent.epsilon = agent.epsilon_end

        while (
            step < max_steps_per_episode and game_over != True
        ):  # Run until max steps or until game is finished
            old_state = tuple(
                np.hstack(
                    [environment.current_location, agent.food_level, agent.water_level]
                )
            )

            action = agent.choose_action(environment.actions)
            environment.make_step(action, agent)
            new_state = tuple(
                np.hstack(
                    [environment.current_location, agent.food_level, agent.water_level]
                )
            )

            reward = agent.get_reward(old_state, new_state)

            agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward
            step += 1

            if not agent.save_params == None:
                if (episode % 100000) == 0:
                    agent.save_model(episode)

            if environment.check_done(agent) == "DONE":
                game_over = environment.reset(agent, env_params)
                break

        # Append reward for current episode to performance log
        reward_per_episode[episode] = cumulative_reward
        step_per_episode[episode] = step
        epsilon_per_episode[episode] = agent.epsilon

    return (
        reward_per_episode,
        step_per_episode,
        epsilon_per_episode,
    )  # Return performance log


def play(
    environment,
    agent,
    env_params,
):
    action = agent.choose_action(environment.actions, max_action=True, verbose=True)
    environment.make_step(action, agent)
    reward = agent.get_reward()

    if environment.check_done(agent) == "DONE":
        game_over = environment.reset(agent, env_params)

    return reward, action


# def print_values(agent, normalized=True, save=False, savedir=None, title=None):
#     grid = np.zeros((agent.environment.height, agent.environment.width))

#     food_level = agent.food_level
#     water_level = agent.water_level

#     for i in range(agent.environment.height):
#         for j in range(agent.environment.width):
#             q = agent.q_table[tuple([i, j, food_level, water_level])]
#             q_list = list(q.values())

#             grid[i, j] = np.max(q_list)

#     cmap = plt.cm.get_cmap("viridis", 10)
#     norm = plt.Normalize(np.min(grid), np.max(grid))
#     rgba = cmap(norm(grid))

#     fig, ax = plt.subplots()

#     if not title == None:
#         fig.suptitle(title)
#     im = ax.imshow(rgba)

#     for i in range(grid.shape[0]):
#         for j in range(grid.shape[1]):
#             if normalized:
#                 text = ax.text(
#                     i,
#                     j,
#                     round(norm(grid)[i, j], 2),
#                     ha="center",
#                     va="center",
#                     color="k",
#                 )
#             else:
#                 text = ax.text(
#                     i, j, round(grid[i, j], 2), ha="center", va="center", color="k",
#                 )

#     # plt.show()
#     if save:
#         if savedir == None:
#             raise ("Save plot of Q values need savedir")
#         else:
#             plt.savefig(savedir, dpi=300)
#             plt.close()


if __name__ == "__main__":
    # agent_params = {
    #     "max_food": 30,
    #     "min_food": 0,
    #     "max_water": 30,
    #     "min_water": 0,
    #     "set_point_food": 30,
    #     "set_point_water": 30,
    #     "gamma": 0.9,
    #     "epsilon": 1.0,
    #     "epsilon_end": 0.2,
    #     "alpha": 0.1,
    # }

    # env_params = {
    #     "height": 20,
    #     "width": 20,
    #     "food_location": (0, 0),
    #     "water_location": (19, 19),
    # }

    # agent_params = {
    #     "max_food": 20,
    #     "min_food": 0,
    #     "max_water": 20,
    #     "min_water": 0,
    #     "set_point_food": 10,
    #     "set_point_water": 10,
    #     "gamma": 0.9,
    #     "epsilon": 1.0,
    #     "epsilon_end": 0.2,
    #     "alpha": 0.1,
    # }

    # env_params = {
    #     "height": 5,
    #     "width": 5,
    #     "food_location": (0, 0),
    #     "water_location": (4, 4),
    # }

    agent_params = {
        "max_food": 15,
        "min_food": 0,
        "max_water": 15,
        "min_water": 0,
        "set_point_food": 15,
        "set_point_water": 15,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_end": 0.2,
        "alpha": 0.1,
    }

    env_params = {
        "height": 5,
        "width": 5,
        "food_location": (0, 0),
        "water_location": (4, 4),
    }

    # agent_params = {
    #     "max_food": 10,
    #     "min_food": 0,
    #     "max_water": 10,
    #     "min_water": 0,
    #     "set_point_food": 10,
    #     "set_point_water": 10,
    #     "gamma": 0.9,
    #     "epsilon": 1.0,
    #     "epsilon_end": 0.2,
    #     "alpha": 0.1,
    # }

    # env_params = {
    #     "height": 5,
    #     "width": 5,
    #     "food_location": (0, 0),
    #     "water_location": (4, 4),
    # }

    tag = "IAI_G{}{}_I{}{}_gamma{}_eps{}_{}".format(
        env_params["height"],
        env_params["width"],
        agent_params["max_food"],
        agent_params["max_water"],
        agent_params["gamma"],
        agent_params["epsilon"],
        "fixedWaterDecay",
    )

    save_params = {
        "savedir": "/Users/sungwoo320/Documents/cocoan_github/interoceptive_grid_world/save",
        "imgdir": "/Users/sungwoo320/Documents/cocoan_github/interoceptive_grid_world/img",
        "tag": tag,
    }

    environment = GridWorld(env_params)

    # save_params = None
    agentQ = Q_Agent(environment, agent_params, save_params, random_action=False)

    # Note the learn=True argument!
    # reward_per_episode, step_per_episode, epsilon_per_episode = train(
    #     environment, agentQ, env_params, max_steps_per_episode=100, episodes=500000
    # )

    reward_per_episode, step_per_episode, epsilon_per_episode = train(
        environment, agentQ, env_params, max_steps_per_episode=100, episodes=5000000
    )
