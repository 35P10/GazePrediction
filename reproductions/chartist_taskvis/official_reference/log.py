import os
import numpy as np
import random
# from stable_baselines3.common.env_util import make_atari_env
import json
from PIL import Image
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from visualization_utils import plot_img, plot_scanpath, plot_heatmap
from model.search_env import SearchEnv
from model.map_env import MapEnv
from model.read_env import ReadEnv
from model.title_env import TitleEnv
from model.llm_policy import LLMPolicy
import csv
from tqdm import tqdm
import pandas as pd
from copy import copy

IMAGE_SIZE = 320 # 320x320

scanpath_xs = []
scanpath_ys = []

folder = "images"
meta_json = 'meta.json'
meta_infos = json.load(open(meta_json))

human_data_meta_infos = [meta_infos[0], meta_infos[5], meta_infos[8],
                     meta_infos[10], meta_infos[15], meta_infos[18],
                     meta_infos[20], meta_infos[24], meta_infos[29],
                     meta_infos[30], meta_infos[35], meta_infos[38]]

image_sizes = {
    "economist_daily_chart_35": [1000, 973],
    "economist_daily_chart_85": [1000, 980],
    "economist_daily_chart_165": [1000, 919],
    "economist_daily_chart_392": [1000, 798]
}

for meta_info in tqdm(human_data_meta_infos):
    print(meta_info['question'])
    meta = meta_info

    search_env = SearchEnv(meta_info, meta_json = meta_json, folder=folder)
    map_env = MapEnv(meta_info, meta_json = meta_json, folder=folder)
    read_env = ReadEnv(meta_info, meta_json = meta_json, folder=folder)
    title_env = TitleEnv(meta_info, meta_json = meta_json, folder=folder)

    search_model = PPO.load("model/vision_search.pt", device="auto")
    map_model = PPO.load("model/vision_map.pt", device="auto")
    read_model = PPO.load("model/vision_read.pt", device="auto")
    title_model = PPO.load("model/vision_title.pt", device="auto")
    model = LLMPolicy(search_model=search_model, map_model=map_model, read_model=read_model, title_model=title_model)

    ORIGIN_WIDTH = image_sizes[meta_info['image']][0]
    ORIGIN_HEIGHT = image_sizes[meta_info['image']][1]

    with open('generated_scanpaths/%s_%s.csv'%(meta_info['image'], meta_info['task']), mode='w') as scanpath_file:
        scanpath_writer = csv.writer(scanpath_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        scanpath_writer.writerow(["user", "index", "x", "y"])

        for i in tqdm(range(20)):
            model.setup(meta, meta_json = meta_json, folder = folder)

            # llm based high-level control
            step = 0
            done = False
            while not done:
                step += 1
                done = model.action()
            scanpath_xs = model.scanpath_xs
            scanpath_ys = model.scanpath_ys

            # write scanpath to csv, one row one fixation
            for j in range(len(scanpath_xs)):
                x = int(scanpath_xs[j] * (ORIGIN_WIDTH / IMAGE_SIZE))
                y = int(scanpath_ys[j] * (ORIGIN_HEIGHT / IMAGE_SIZE))
                scanpath_writer.writerow([i, j, x, y])

images = ['economist_daily_chart_35', 'economist_daily_chart_85', 'economist_daily_chart_165', 'economist_daily_chart_392'] # 'vis586'
tasks = ['filter', 'retrieve_value', 'find_extreme'] # 'rv'

for image in images:
    print(image)
    for task in tasks:
        print(task)
        df = pd.read_csv('./generated_scanpaths/%s_%s.csv'%(image, task))
        groups = df.groupby('user')
        for user, group in tqdm(groups):
            print(user)
            image_file = "%s.png"%image
            chart = Image.open('./labels/'+image_file)
            fig = plot_img(chart)
            x = copy(df['x'].values)
            y = copy(df['y'].values)
            # how to sample a sublist from a list
            num = 20
            indices = sorted(random.sample(range(len(x)), num))
            x_subset = [x[i] for i in indices]
            y_subset = [y[i] for i in indices]

            duration = [1] * len(x)
            plot_scanpath(x_subset,y_subset,duration)
            plot_heatmap(x_subset,y_subset,duration)
            # make a folder and save the images
            plt.savefig('./generated_scanpaths/images/'+image+'_'+task+'_'+str(user)+'.png')
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()