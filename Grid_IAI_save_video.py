import os, glob, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


agent_params = {
    "max_food": 30,
    "min_food": 0,
    "max_water": 30,
    "min_water": 0,
    "set_point_food": 30,
    "set_point_water": 30,
    "gamma": 0.9,
    "epsilon": 0.1,
    "alpha": 0.1,
}

env_params = {
    "height": 20,
    "width": 20,
    "food_location": (0, 0),
    "water_location": (19, 19),
}

tag = "IAI_G{}{}_I{}{}_gamma{}_eps{}".format(
    env_params["height"],
    env_params["width"],
    agent_params["max_food"],
    agent_params["max_water"],
    agent_params["gamma"],
    agent_params["epsilon"],
)

save_params = {
    "videodir": "/media/das_junhyeok/interoceptive-ai/_analysis/state_value_dynamic_vis/video",
    "imgdir": "/media/das_junhyeok/interoceptive-ai/_analysis/state_value_dynamic_vis/img",
    "tag": tag,
}

imgdir = os.path.join(save_params["imgdir"], tag)
videodir = os.path.join(save_params["videodir"], tag)

if not os.path.exists(videodir):
    os.mkdir(videodir)


# Internal Q Grid
_list = glob.glob(os.path.join(imgdir, "Internal*Grid*.png"))
_list = np.sort(np.array(_list))

img_array = []
for filename in _list:
    img = cv2.imread(filename)
    try:
        height, width, layers = img.shape
    except:
        continue
    size = (width, height)
    img_array.append(img)

_save = os.path.join(videodir, "Internal_Q_Grid.mp4")
out = cv2.VideoWriter(_save, cv2.VideoWriter_fourcc(*"MP4V"), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Done: {}".format(_save))


# Internal Q Food
_list = glob.glob(os.path.join(imgdir, "Internal*Food*.png"))
_list = np.sort(np.array(_list))

img_array = []
for filename in _list:
    img = cv2.imread(filename)
    try:
        height, width, layers = img.shape
    except:
        continue
    size = (width, height)
    img_array.append(img)

_save = os.path.join(videodir, "Internal_Q_Food.mp4")
out = cv2.VideoWriter(_save, cv2.VideoWriter_fourcc(*"MP4V"), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Done: {}".format(_save))

# Internal Q Water
_list = glob.glob(os.path.join(imgdir, "Internal*Water*.png"))
_list = np.sort(np.array(_list))

img_array = []
for filename in _list:
    img = cv2.imread(filename)
    try:
        height, width, layers = img.shape
    except:
        continue
    size = (width, height)
    img_array.append(img)

_save = os.path.join(videodir, "Internal_Q_Water.mp4")
out = cv2.VideoWriter(_save, cv2.VideoWriter_fourcc(*"MP4V"), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Done: {}".format(_save))


# External V1
_list = glob.glob(os.path.join(imgdir, "External*V1*.png"))
_list = np.sort(np.array(_list))

img_array = []
for filename in _list:
    img = cv2.imread(filename)
    try:
        height, width, layers = img.shape
    except:
        continue
    size = (width, height)
    img_array.append(img)

_save = os.path.join(videodir, "External_Q_V1.mp4")
out = cv2.VideoWriter(_save, cv2.VideoWriter_fourcc(*"MP4V"), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Done: {}".format(_save))

# External V2
_list = glob.glob(os.path.join(imgdir, "External*V2*.png"))
_list = np.sort(np.array(_list))

img_array = []
for filename in _list:
    img = cv2.imread(filename)
    try:
        height, width, layers = img.shape
    except:
        continue
    size = (width, height)
    img_array.append(img)

_save = os.path.join(videodir, "External_Q_V2.mp4")
out = cv2.VideoWriter(_save, cv2.VideoWriter_fourcc(*"MP4V"), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Done: {}".format(_save))
