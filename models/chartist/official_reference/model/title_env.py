import numpy as np
from gym import spaces
from PIL import Image
import pygame
from model.chart_env import ChartEnv
import torch

ACTION = 20
VISION_SIZE = 16

class TitleEnv(ChartEnv):
    def __init__(self, meta = None, meta_json = 'meta.json', ACTION = ACTION, folder = "images"):
        super().__init__(meta, meta_json, folder, ACTION)
        self.vision_width = VISION_SIZE
        self.vision_height = VISION_SIZE
        self.folder = folder

        # self.observation_space = spaces.Box(low=0, high=255, shape=(6, self.vision_width, self.vision_height), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "fixation": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.uint8),
            "visited": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.uint8),
            "peripheral": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.float32),
        })
        self.action_space = spaces.MultiDiscrete([ACTION, ACTION, 2]) # ACTIONxACTION places to look

        self.num_success = 0
        self.num_fixations = 0
        self.steps = 0
        self.dist = 0
        self.init_gaze_position = None

        self.init()

        #### Render #####
        self.window = pygame.display.set_mode((self.chart_width, self.chart_height + self.vision_height * 2))
        self.clock = pygame.time.Clock()

    def observation(self):

        obs = {
            "fixation": self.fixation_map,
            "visited": self.visited_map,
            "peripheral": self.peripheral,
        }

        return obs

    def step(self, action):
        self.steps += 1
        self.num_fixations += 1
        self.pre_x = self.x
        self.pre_y = self.y
        if self.pre_x != -1 or self.pre_y != -1: self.visited_map[self.x][self.y] = 1 # mark the visited position
        self.x = action[0]
        self.y = action[1]
        self.movement_dist = np.sqrt((self.x - self.pre_x)**2 + (self.y - self.pre_y)**2)
        self.dist += self.movement_dist
        self.fixation_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        self.fixation_map[self.x][self.y] = 1
        self.gaze_area = (self.y * self.gaze_size, self.x * self.gaze_size, (self.y + 1) * self.gaze_size, (self.x + 1) * self.gaze_size) # Note: the reason to flip x and y here is because the matrix is rotated
        self.done = action[2] == 1 or self.steps >= 5 # limited to 5 fixations
        obs = self.observation()
        r = self.reward()
        info = {"function": "gather_info", "position": (action[0], action[1]), "text": self.get_text_in_gaze()}
        return obs, r, self.done, info
    
    def reward(self):
        r = 0
        if self.done:
            condition_on_title = self.title_map * self.visited_map > 0
            condition_on_axis = self.axis_map * self.visited_map > 0
            if np.sum(condition_on_title) > 3:
                r += 2
            else:
                r += np.sum(condition_on_title)
            r += np.sum(condition_on_axis)*0.5
            if np.any(condition_on_title):
                self.num_success += 1

        else:
            # spatial penalty
            r -= self.movement_dist * 0.0005
            if self.axis_map[self.x][self.y] > 0:
                r += 0.02 # shaping the reward when it look at the axis 

        return r

    def reset(self, meta = None, fixation = None, visited = None):
        super().reset(meta=meta)
        self.steps = 0
        self.dist = 0
        self.init()
        if fixation is not None:
            self.fixation_map = fixation
            self.x = np.where(fixation == 1)[0][0]
            self.y = np.where(fixation == 1)[1][0]
            self.init_gaze_position = (self.x, self.y)
        else:
            self.init_gaze_position = (np.random.randint(ACTION), np.random.randint(ACTION))
            self.x = self.init_gaze_position[0]
            self.y = self.init_gaze_position[1]
            self.fixation_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
            self.fixation_map[self.x][self.y] = 1
        if visited is not None:
            self.visited_map = visited

        return self.observation()
    
    def init(self):
        # chart
        self.title_img_url = "%s/chart_%s_title.png"%(self.folder, self.meta['id'])
        try:
            title_img = Image.open(self.title_img_url)
            self.title_map = np.array(title_img.convert('L'))/255
        except:
            self.title_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        self.axis_img_url = "%s/chart_%s_axis.png"%(self.folder, self.meta['id'])
        self.axis_map = np.array(Image.open(self.axis_img_url).convert('L'))/255
        self.peripheral_img_url = "%s/chart_%s_p.png"%(self.folder, self.meta['id'])
        peripheral = Image.open(self.peripheral_img_url)
        self.peripheral = np.array(peripheral.convert('L'))/255
        # vision + memory
        self.fixation_map = np.zeros([ACTION,ACTION],dtype=np.uint8) # current fixation
        self.visited_map = np.zeros([ACTION,ACTION],dtype=np.uint8) # visited positions

    def render(self):
        render_fps = 1 # 500ms per timestep

        pygame.display.set_caption(self.question)

        mode = self.image.mode
        size = self.image.size
        data = self.image.tobytes()
        image = pygame.image.fromstring(data, size, mode)
        self.window.blit(image, (0,0))

        left, top, width, height = self.gaze_area[0], self.gaze_area[1], self.gaze_area[2] - self.gaze_area[0], self.gaze_area[3] - self.gaze_area[1]
        if self.done:
            pygame.draw.rect(self.window,'green',(pygame.Rect(left, top, width, height)), 2)
        else:
            pygame.draw.rect(self.window,'red',(pygame.Rect(left, top, width, height)), 2)

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(render_fps)