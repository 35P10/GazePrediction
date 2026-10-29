import numpy as np
import json
import random
import gymnasium as gym
from gymnasium import Env
from gym import spaces
from abc import ABC
import torchvision.transforms as transforms
from PIL import Image
import pygame

IMAGE_SIZE = 320 # 320x320

class ChartEnv(Env, ABC):
    def __init__(self, meta = None, meta_json = 'meta.json', folder = 'images', ACTION = 5):
        self.ACTION = ACTION
        self.meta = meta
        self.meta_json = json.load(open(meta_json))
        self.folder = folder
        self.chart_width = IMAGE_SIZE
        self.chart_height = IMAGE_SIZE
        self.img_url = "%s/chart_%s.png"%(self.folder, self.meta['id'])
        image = Image.open(self.img_url)
        self.image = transforms.Resize((self.chart_width, self.chart_height))(image)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.chart_width, self.chart_height, 3), dtype=np.uint8)

        self.question = self.meta['question']

        self.x = 0 # inital x position
        self.y = 0 # inital y position
        self.action_space = spaces.MultiDiscrete([ACTION, ACTION, 2]) # an action is 60x60 pixels
        self.gaze_size = int(self.chart_width / ACTION)
        self.gaze_area = (0, 0, self.gaze_size, self.gaze_size)

        #### Render #####
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.chart_width, self.chart_height))
        self.clock = pygame.time.Clock()

    def observation(self):
        return self.image

    def step(self, action):
        self.x = action[0]
        self.y = action[1]
        self.gaze_area = (self.x * self.gaze_size, self.y * self.gaze_size, (self.x + 1) * self.gaze_size, (self.y + 1) * self.gaze_size)
        self.done = action[2] == 1
        obs = self.observation()
        r = self.reward()
        return obs, r, self.done, {}
    
    def get_text_in_gaze(self):
        text_positions = self.meta['text_positions']
        text_in_gaze_area = ""
        for text_position in text_positions:
            x_int_tl = max(self.gaze_area[0], text_position['position'][0])
            y_int_tl = max(self.gaze_area[1], text_position['position'][1])
            x_int_br = min(self.gaze_area[2], text_position['position'][2])
            y_int_br = min(self.gaze_area[3], text_position['position'][3])
            if x_int_br > x_int_tl and y_int_br > y_int_tl:
                text_in_gaze_area += text_position["text"] + ";"

        return text_in_gaze_area
    
    def reward(self):
        pass

    def reset(self, meta = None):
        if meta is not None:
            self.meta = meta
        else:
            self.meta = random.choice(self.meta_json)
        self.img_url = "%s/chart_%s.png"%(self.folder, self.meta['id'])
        self.x = -1 # inital x position at no where
        self.y = -1 # inital y position at no where
        self.pre_x = 2
        self.pre_y = 2
        self.done = 0
        image = Image.open(self.img_url)
        self.image = transforms.Resize((self.chart_width, self.chart_height))(image)
        self.question = self.meta['question']
        self.gaze_area = (0, 0, self.gaze_size, self.gaze_size)

        obs = self.observation()
        return obs
    
    def render(self):
        pass