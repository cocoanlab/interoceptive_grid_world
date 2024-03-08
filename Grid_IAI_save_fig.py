from genericpath import exists
import os, glob, re
import pickle
import numpy as np
import pandas as pd
from Grid_IAI_environment import GridWorld
from Grid_IAI_agent import Q_Agent

# from Grid_IAI_Q_learning import print_values
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


def print_values(
    agent,
    normalized=True,
    save=False,
    savedir=None,
    title=None,
    draw_text=True,
    alpha=0.7,
):
    grid = np.zeros((agent.environment.height, agent.environment.width))

    food_level = agent.food_level
    water_level = agent.water_level

    for i in range(agent.environment.height):
        for j in range(agent.environment.width):
            q = agent.q_table[tuple([i, j, food_level, water_level])]
            q_list = list(q.values())

            grid[i, j] = np.max(q_list)

    cmap = plt.cm.get_cmap("viridis", 10)
    norm = plt.Normalize(np.min(grid), np.max(grid))
    rgba = cmap(norm(grid))

    for i in range(rgba.shape[0]):
        for j in range(rgba.shape[1]):
            rgba[i, j][-1] = alpha

    rgba[agent.environment.food_location] = 242 / 255, 241 / 255, 239 / 255, alpha
    rgba[agent.environment.water_location] = 242 / 255, 241 / 255, 239 / 255, alpha

    fig, ax = plt.subplots()

    if not title == None:
        fig.suptitle(title)
    im = ax.imshow(rgba)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i, j) == agent.environment.food_location:
                if draw_text:
                    text = ax.text(
                        agent.environment.food_location[0],
                        agent.environment.food_location[1],
                        "Food",
                        ha="center",
                        va="center",
                        color="k",
                    )

            elif (i, j) == agent.environment.water_location:
                if draw_text:
                    text = ax.text(
                        agent.environment.water_location[0],
                        agent.environment.water_location[1],
                        "Water",
                        ha="center",
                        va="center",
                        color="k",
                    )

            else:
                if normalized:
                    if draw_text:
                        text = ax.text(
                            i,
                            j,
                            round(norm(grid)[i, j], 2),
                            ha="center",
                            va="center",
                            color="k",
                        )
                else:
                    if draw_text:
                        text = ax.text(
                            i,
                            j,
                            round(grid[i, j], 2),
                            ha="center",
                            va="center",
                            color="k",
                        )

    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # plt.show()
    if save:
        if savedir == None:
            raise ("Save plot of Q values need savedir")
        else:
            plt.savefig(savedir, dpi=300)
            plt.close()


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
    #     "epsilon_end": 0.1,
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
        "savedir": "/media/nas01/projects/Interoceptive-AI/interoceptive-ai/_analysis/Grid_IAI/save",
        "imgdir": "/media/nas01/projects/Interoceptive-AI/interoceptive-ai/_analysis/Grid_IAI/img",
        "tag": tag,
    }

    savedir = os.path.join(save_params["savedir"], tag)
    imgdir = os.path.join(save_params["imgdir"], tag)

    if not os.path.exists(imgdir):
        os.makedirs(imgdir, exist_ok=True)

    save_params = None
    environment = GridWorld(env_params)
    agentQ = Q_Agent(environment, agent_params, save_params, random_action=False)

    # Load files
    Q_files = glob.glob(os.path.join(savedir, "*.pkl"))
    Q_files = np.sort(Q_files)
    print(len(Q_files), Q_files[:5])

    Trials = []
    for i in range(len(Q_files)):
        p = re.compile("Trial_\d*")
        m = p.findall(Q_files[i])

        p = re.compile("\d*")
        m = np.array(p.findall(m[0]))

        index = m == ""
        trial_num = int(m[~index][0])
        Trials.append("Trial_{:07d}".format(trial_num))

    Trials = np.sort(np.array(Trials))
    print(Trials[:5])

    # Saving Figs
    environment.current_location = (10, 10)

    for i, trial in enumerate(Trials):
        with open(Q_files[i], "rb") as f:
            Q_model = pickle.load(f)

        agentQ.q_table = Q_model

        # agentQ.food_level = int(0.4 * agentQ.max_food)
        # agentQ.water_level = int(0.7 * agentQ.max_water)

        agentQ.food_level = int(0.4 * agentQ.max_food)
        agentQ.water_level = int(0.9 * agentQ.max_water)

        title = "{}, Food Level: {}, Water Level: {}".format(
            trial, agentQ.food_level, agentQ.water_level
        )

        _save = os.path.join(imgdir, "External_Q_value_V1_{}".format(trial))
        if not os.path.isfile(_save + ".png"):
            print_values(
                agentQ,
                normalized=True,
                title=title,
                save=True,
                savedir=_save,
            )

            print("Save: {}".format(_save))

        # agentQ.food_level = int(0.7 * agentQ.max_food)
        # agentQ.water_level = int(0.4 * agentQ.max_water)

        agentQ.food_level = int(0.9 * agentQ.max_food)
        agentQ.water_level = int(0.4 * agentQ.max_water)

        title = "{}, Food Level: {}, Water Level: {}".format(
            trial, agentQ.food_level, agentQ.water_level
        )

        _save = os.path.join(imgdir, "External_Q_value_V2_{}".format(trial))
        if not os.path.isfile(_save + ".png"):
            print_values(
                agentQ,
                normalized=True,
                title=title,
                save=True,
                savedir=_save,
            )

            print("Save: {}".format(_save))

        # # Internal Q Grid
        # savedir = os.path.join(imgdir, "Internal_Q_value_Grid_{}".format(trial))
        # internal_q = np.zeros((agentQ.max_food, agentQ.max_water))

        # if not os.path.isfile(savedir + ".png"):
        #     for f in range(agentQ.max_food):
        #         for w in range(agentQ.max_water):
        #             q_list = []
        #             for i in range(environment.height):
        #                 for j in range(environment.width):
        #                     key = (i, j, f, w)
        #                     if key in agentQ.q_table.keys():
        #                         q = list(agentQ.q_table[key].values())
        #                         max_q = np.max(q)
        #                         q_list.append(max_q)

        #             internal_q[f, w] = np.mean(q_list)

        #     f_q = np.mean(internal_q, axis=1)
        #     w_q = np.mean(internal_q, axis=0)

        #     cmap = plt.cm.get_cmap("viridis", 10)
        #     norm = plt.Normalize(np.min(internal_q), np.max(internal_q))
        #     rgba = cmap(norm(internal_q))

        #     fig, ax = plt.subplots(figsize=(12, 12))

        #     title = "{}, Internal Q Grid".format(trial)
        #     fig.suptitle(title)

        #     im = ax.imshow(rgba)

        #     for i in range(internal_q.shape[0]):
        #         for j in range(internal_q.shape[1]):
        #             text = ax.text(
        #                 i,
        #                 j,
        #                 round(internal_q[i, j], 1),
        #                 ha="center",
        #                 va="center",
        #                 color="k",
        #             )

        #     fig.tight_layout()
        #     plt.savefig(savedir, dpi=300)
        #     plt.close()

        #     print("Save: {}".format(savedir))

        #     # Internal Q Food
        #     savedir = os.path.join(imgdir, "Internal_Q_value_Food_{}".format(trial))
        #     if not os.path.isfile(savedir + ".png"):
        #         cmap = plt.cm.get_cmap("viridis", 10)
        #         norm = plt.Normalize(np.min(f_q), np.max(f_q))
        #         rgba = cmap(np.expand_dims(norm(f_q), axis=-1).T)

        #         fig, ax = plt.subplots(figsize=(7, 7))

        #         title = "{}, Internal Q Food".format(trial)
        #         fig.suptitle(title)

        #         im = ax.imshow(rgba)

        #         for i in range(f_q.shape[0]):
        #             text = ax.text(
        #                 i, 0, round(f_q[i], 1), ha="center", va="center", color="k"
        #             )

        #         fig.tight_layout()
        #         plt.savefig(savedir, dpi=300)
        #         plt.close()

        #         print("Save: {}".format(savedir))

        #     # Internal Q Water
        #     savedir = os.path.join(imgdir, "Internal_Q_value_Water_{}".format(trial))
        #     if not os.path.isfile(savedir + ".png"):
        #         cmap = plt.cm.get_cmap("viridis", 10)
        #         norm = plt.Normalize(np.min(w_q), np.max(w_q))
        #         rgba = cmap(np.expand_dims(norm(w_q), axis=-1).T)

        #         fig, ax = plt.subplots(figsize=(7, 7))

        #         title = "{}, Internal Q Water".format(trial)
        #         fig.suptitle(title)

        #         im = ax.imshow(rgba)

        #         for i in range(w_q.shape[0]):
        #             text = ax.text(
        #                 i, 0, round(w_q[i], 1), ha="center", va="center", color="k"
        #             )

        #         fig.tight_layout()
        #         plt.savefig(savedir, dpi=300)
        #         plt.close()

        #         print("Save: {}".format(savedir))
