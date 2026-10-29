import numpy as np
from gym import spaces
from PIL import Image
import pygame
from model.chart_env import ChartEnv
import torch
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer

ACTION = 20
VISION_SIZE = 16

class ReadEnv(ChartEnv):
    def __init__(self, meta = None, meta_json = 'meta.json', ACTION = ACTION, folder = "images"):
        super().__init__(meta, meta_json, folder, ACTION)
        self.vision_width = VISION_SIZE
        self.vision_height = VISION_SIZE
        self.folder = folder

        # self.observation_space = spaces.Box(low=0, high=255, shape=(6, self.vision_width, self.vision_height), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "fixation": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.uint8),
            "visited": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.uint8),
            "reference": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.uint8), # reference_map_point
            "peripheral": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.float32),
            "saliency": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.float32),
            "task": spaces.Box(low=-3, high=3, shape=(768,), dtype=np.float32), # embeded text
        })
        self.action_space = spaces.MultiDiscrete([ACTION, ACTION]) # ACTIONxACTION places to look

        self.num_success = 0
        self.num_fixations = 0
        self.revisits = 0
        self.steps = 0
        self.dist = 0
        self.init_gaze_position = None

        self.sentenceTransformer = SentenceTransformer("stsb-roberta-base-v2")

        self.init()

        #### Render #####
        self.window = pygame.display.set_mode((self.chart_width, self.chart_height + self.vision_height * 2))
        self.clock = pygame.time.Clock()

    def observation(self):

        obs = {
            "fixation": self.fixation_map,
            "visited": self.visited_map,
            "reference": self.reference_map,
            "peripheral": self.peripheral,
            "saliency": self.saliency,
            "task": self.task_obs, # embeded text
        }

        return obs

    def step(self, action):
        self.steps += 1
        self.num_fixations += 1
        self.pre_x = self.x
        self.pre_y = self.y
        if self.visited_map[self.x][self.y] == 1: self.revisits += 1
        if self.pre_x != -1 or self.pre_y != -1: self.visited_map[self.x][self.y] = 1 # mark the visited position
        self.x = action[0]
        self.y = action[1]
        self.movement_dist = np.sqrt((self.x - self.pre_x)**2 + (self.y - self.pre_y)**2)
        self.dist += self.movement_dist
        self.fixation_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        self.fixation_map[self.x][self.y] = 1
        self.gaze_area = (self.y * self.gaze_size, self.x * self.gaze_size, (self.y + 1) * self.gaze_size, (self.x + 1) * self.gaze_size) # Note: the reason to flip x and y here is because the matrix is rotated
        self.done = self.steps >= 5 # limited to 5 fixations # action[2] == 1 or 
        obs = self.observation()
        r = self.reward()
        info = {"function": "read_axis", "position": (action[0], action[1]), "text": self.get_text_in_gaze()}
        return obs, r, self.done, info
    
    def reward(self):
        r = 0
        if self.done:
            if np.any(self.read_map * self.visited_map > 0):
                r += 2
                self.num_success += 1
            else:
                r += 0.01 * np.any(self.axis_map * self.visited_map > 0) # shaping the reward when it look at the axis 

            r -= 0.1 * self.revisits 
            r -= self.dist * 0.0005
        else:
            # spatial penalty
            r -= self.movement_dist * 0.0005

        return r

    def reset(self, meta = None, fixation = None, visited = None, reference = None):
        super().reset(meta=meta)
        self.revisits = 0
        self.steps = 0
        self.dist = 0
        self.init()
        if fixation is not None:
            self.fixation_map = fixation
            self.x = np.where(fixation == 1)[0][0]
            self.y = np.where(fixation == 1)[1][0]
            self.init_gaze_position = (self.x, self.y)
        else:
            self.fixation_map[self.init_gaze_position] = 1
        if visited is not None:
            self.visited_map = visited
        if reference is not None:
            self.reference_map = reference

        return self.observation()
    
    def init(self):

        self.search_img_url = "%s/chart_%s_search.png"%(self.folder, self.meta['id'])
        try:
            self.search_map = np.array(Image.open(self.search_img_url))/255 # we set search as the reference for now
        except:
            self.search_map = np.zeros([ACTION,ACTION],dtype=np.uint8)

        self.axis_img_url = "%s/chart_%s_axis.png"%(self.folder, self.meta['id'])
        try:
            self.axis_map = np.array(Image.open(self.axis_img_url))/255
        except:
            self.axis_map = np.zeros([ACTION,ACTION],dtype=np.uint8)

        self.read_img_url = "%s/chart_%s_read.png"%(self.folder, self.meta['id'])
        try:
            self.read_map = np.array(Image.open(self.read_img_url))/255
        except:
            self.read_map = np.zeros([ACTION,ACTION],dtype=np.uint8)

        self.target_mark_map_url = "%s/chart_%s_map_mark.png"%(self.folder, self.meta['id'])
        try:
            self.target_mark_map = np.array(Image.open(self.target_mark_map_url))/255
            indices = np.argwhere(self.target_mark_map > 0) # Find the indices of all `1` elements in the matrix
            chosen_index = indices[np.random.choice(len(indices))] # Randomly choose one of these indices
            self.reference_map = np.zeros_like(self.target_mark_map)
            self.reference_map[tuple(chosen_index)] = 1
            self.init_gaze_position = tuple(chosen_index)
        except:
            self.target_mark_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
            self.reference_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
            self.init_gaze_position = (0,0)

        self.peripheral_img_url = "%s/chart_%s_p.png"%(self.folder, self.meta['id'])
        peripheral = Image.open(self.peripheral_img_url)
        self.peripheral = np.array(peripheral.convert('L'))/255

        self.saliency_img_url = "%s/chart_%s_saliency.png"%(self.folder, self.meta['id'])
        saliency = Image.open(self.saliency_img_url)
        self.saliency = np.array(saliency.convert('L'))/255

        # vision + memory
        self.fixation_map = np.zeros([ACTION,ACTION],dtype=np.uint8) # current fixation
        self.visited_map = np.zeros([ACTION,ACTION],dtype=np.uint8) # visited positions

        self.task_obs = self.sentenceTransformer.encode(self.question)

    def render(self):
        render_fps = 1 # 500ms per timestep

        pygame.display.set_caption(self.question)

        mode = self.image.mode
        size = self.image.size
        data = self.image.tobytes()
        image = pygame.image.fromstring(data, size, mode)
        self.window.blit(image, (0,0))

        left, top, width, height = self.gaze_area[0], self.gaze_area[1], self.gaze_area[2] - self.gaze_area[0], self.gaze_area[3] - self.gaze_area[1]
        pygame.draw.rect(self.window,'red',(pygame.Rect(left, top, width, height)), 2)

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(render_fps)