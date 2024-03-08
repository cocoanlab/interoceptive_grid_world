#%%
import os, glob, re
import pickle
import numpy as np
import pandas as pd
from Grid_IAI_environment import GridWorld
from Grid_IAI_agent import Q_Agent
from Grid_IAI_Q_learning import GridWorld, Q_Agent, play, train
from Grid_IAI_save_fig import print_values
import matplotlib.pyplot as plt

# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")

np.random.seed(1223)

# ## Run Random Agent

# In[2]:


# agent_params = {
#     "max_food": 15,
#     "min_food": 0,
#     "max_water": 15,
#     "min_water": 0,
#     "set_point_food": 15,
#     "set_point_water": 15,
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

# save_params = None

# environment = GridWorld(env_params)
# agentQ = Q_Agent(environment, agent_params, save_params, random_action=True)

# # Note the learn=True argument!
# reward_per_episode, step_per_episode, epsilon_per_episode = train(
#     environment, agentQ, env_params, episodes=10000
# )
# # Simple learning curve
# print(np.mean(reward_per_episode[-100:]))
# plt.plot(reward_per_episode[-100:])
# plt.show()
# plt.plot(step_per_episode[-100:])
# plt.show()
# plt.plot(epsilon_per_episode)
# plt.show()


# ## Q-Agent

# In[3]:
agent_params = {
    "max_food": 20,
    "min_food": 0,
    "max_water": 20,
    "min_water": 0,
    "set_point_food": 10,
    "set_point_water": 10,
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

save_params = None

environment = GridWorld(env_params)
agentQ = Q_Agent(environment, agent_params, save_params, random_action=False)


# In[4]:
# Note the learn=True argument!
reward_per_episode, step_per_episode, epsilon_per_episode = train(environment, agentQ, env_params, episodes=100000)
# Simple learning curve
print(np.mean(reward_per_episode[-100:]))
plt.plot(reward_per_episode[-100:])
plt.show()
plt.plot(step_per_episode[-100:])
plt.show()
plt.plot(epsilon_per_episode)
plt.show()


# In[5]:
print("Food Level: {}, Water Level: {}".format(agentQ.food_level, agentQ.water_level))
print("Current position of the agent =", environment.current_location)
environment.print_locations()
print_values(normalized=True, agent=agentQ)


# In[6]:
# reward, action = play(
#     environment, agentQ, env_params
# )

reward = environment.make_step("LEFT", agentQ)
if environment.check_done(agentQ) == "DONE":
    game_over = environment.reset(agentQ, env_params)

print("Food Level: {}, Water Level: {}".format(agentQ.food_level, agentQ.water_level))
print("Current position of the agent =", environment.current_location)
environment.print_locations()
print_values(normalized=True, agent=agentQ)


# In[7]:

environment.current_location = (2, 2)
print(environment.current_location)
environment.print_locations(alpha=0.9, draw_text=True)

agentQ.food_level = 5
agentQ.water_level = 10

title = "Food Level: {}, Water Level: {}".format(agentQ.food_level, agentQ.water_level)
print_values(normalized=True, agent=agentQ, title=None, draw_text=True, alpha=0.8)

agentQ.food_level = 10
agentQ.water_level = 5
title = "Food Level: {}, Water Level: {}".format(agentQ.food_level, agentQ.water_level)
print_values(normalized=True, agent=agentQ, title=None, draw_text=True, alpha=0.8)


# In[8]:
internal_q = np.zeros((agentQ.max_food, agentQ.max_water))

for f in range(agentQ.max_food):
    for w in range(agentQ.max_water):

        q_list = []
        for i in range(environment.height):
            for j in range(environment.width):
                key = (i, j, f, w)
                if key in agentQ.q_table.keys():
                    q = list(agentQ.q_table[key].values())
                    max_q = np.max(q)
                    q_list.append(max_q)

        internal_q[f, w] = np.mean(q_list)
print(internal_q.shape)


# In[9]:
cmap = plt.cm.get_cmap("viridis", 10)
norm = plt.Normalize(np.min(internal_q), np.max(internal_q))
rgba = cmap(norm(internal_q))

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(rgba)

for i in range(internal_q.shape[0]):
    for j in range(internal_q.shape[1]):
        text = ax.text(
            i, j, round(norm(internal_q)[i, j], 1), ha="center", va="center", color="k"
        )


# %%
## Plotting Internal state values
f_q = np.mean(internal_q, axis=1)
w_q = np.mean(internal_q, axis=0)

## Food
cmap = plt.cm.get_cmap("viridis", 10)
norm = plt.Normalize(np.min(f_q), np.max(f_q))
rgba = cmap(np.expand_dims(norm(f_q), axis=-1).T)

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(rgba)

for i in range(f_q.shape[0]):
    text = ax.text(i, 0, round(norm(f_q)[i], 1), ha="center", va="center", color="k")
title = "Internal Q Food"
# fig.suptitle(title)

## Water
cmap = plt.cm.get_cmap("viridis", 10)
norm = plt.Normalize(np.min(w_q), np.max(w_q))
rgba = cmap(np.expand_dims(norm(w_q), axis=-1).T)

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(rgba)

for i in range(w_q.shape[0]):
    text = ax.text(i, 0, round(norm(w_q)[i], 1), ha="center", va="center", color="k")
title = "Internal Q Water"
# fig.suptitle(title)


# %%
