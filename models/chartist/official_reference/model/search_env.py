import numpy as np
from gym import spaces
from PIL import Image
import pygame
from model.chart_env import ChartEnv
from sentence_transformers import SentenceTransformer

ACTION = 20
VISION_SIZE = 16

class SearchEnv(ChartEnv):
    def __init__(self, meta = None, meta_json = 'meta.json', ACTION = ACTION, folder = "images"):
        super().__init__(meta, meta_json, folder, ACTION)
        self.vision_width = VISION_SIZE
        self.vision_height = VISION_SIZE
        self.folder = folder

        # self.observation_space = spaces.Box(low=0, high=255, shape=(6, self.vision_width, self.vision_height), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "fixation": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.uint8),
            "visited": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.uint8),
            "semantic_map": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.float32),
            "peripheral": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.float32),
            "saliency": spaces.Box(low=0, high=1, shape=(ACTION, ACTION), dtype=np.float32),
            "task": spaces.Box(low=-3, high=3, shape=(768,), dtype=np.float32), # embeded text
        })
        self.action_space = spaces.MultiDiscrete([ACTION, ACTION]) # ACTIONxACTION places to look # , 2

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
            "semantic_map": self.semantic_map * self.visited_map,
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
        info = {"function": "search_text", "position": (action[0], action[1]), "text": self.get_text_in_gaze()}
        # print(self.get_text_in_gaze())
        return obs, r, self.done, info
    
    def reward(self):
        r = 0
        if self.done:
            if np.any(self.search_map * self.visited_map * (1 - self.title_map) > 0):
                r += 1
                self.num_success += 1
            else:
                r += np.max(self.semantic_map * self.visited_map * (1 - self.title_map))

            r -= 0.1 * self.revisits # penalize revisits in one operation
        else:
            # spatial penalty
            r -= self.movement_dist * 0.0005
            if self.axis_map[self.x][self.y] > 0:
                r += 0.1 # shaping the reward when it look at the axis 
            if self.semantic_map[self.x][self.y] > self.semantic_map[self.pre_x][self.pre_x]:
                r += 0.1

        return r

    def reset(self, meta = None, fixation = None, visited = None):
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
        # self.task_img_url = "%s/chart_%s_task.png"%(self.folder, self.meta['id'])
        # self.task_image = Image.open(self.task_img_url)
        # self.task_image = transforms.Resize((self.chart_width, self.chart_height))(self.task_image)
        
        self.title_img_url = "%s/chart_%s_title.png"%(self.folder, self.meta['id'])
        try:
            title_img = Image.open(self.title_img_url)
            self.title_map = np.array(title_img.convert('L'))/255
        except:
            self.title_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        self.search_img_url = "%s/chart_%s_search.png"%(self.folder, self.meta['id'])
        try:
            self.search_map = np.array(Image.open(self.search_img_url))/255
        except:
            self.search_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        self.axis_img_url = "%s/chart_%s_axis.png"%(self.folder, self.meta['id'])
        try:
            self.axis_map = np.array(Image.open(self.axis_img_url))/255
        except:
            self.axis_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        self.mark_img_url = "%s/chart_%s_mark.png"%(self.folder, self.meta['id'])
        try:
            self.mark_map = np.array(Image.open(self.mark_img_url))/255
        except:
            self.mark_map = np.zeros([ACTION,ACTION],dtype=np.uint8)

        self.semantic_img_url = "%s/chart_%s_semantic.png"%(self.folder, self.meta['id'])
        semantic_img = Image.open(self.semantic_img_url)
        self.semantic_map = np.array(semantic_img.convert('L'))/255

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
        # if self.done:
        #     pygame.draw.rect(self.window,'green',(pygame.Rect(left, top, width, height)), 2)
        # else:
        #     pygame.draw.rect(self.window,'red',(pygame.Rect(left, top, width, height)), 2)
        pygame.draw.rect(self.window,'red',(pygame.Rect(left, top, width, height)), 2)
        
        # left, top, width, height = self.target[0], self.target[1], self.target[2] - self.target[0], self.target[3] - self.target[1]
        # pygame.draw.rect(self.window,'green',(pygame.Rect(left, top, width, height)), 2)

        # left, top, width, height = self.reference[0], self.reference[1], self.reference[2] - self.reference[0], self.reference[3] - self.reference[1]
        # pygame.draw.rect(self.window,'yellow',(pygame.Rect(left, top, width, height)), 2)

        # peripheral_surface = pygame.surfarray.make_surface(np.stack([self.peripheral*255]*3, axis=-1))
        # foveal_surface = pygame.surfarray.make_surface(np.stack([self.foveal]*3, axis=-1))
        # pos_surface = pygame.surfarray.make_surface(np.stack([self.pos_img]*3, axis=-1))
        # text_map_surface = pygame.surfarray.make_surface(np.stack([self.text_map]*3, axis=-1))
        # mark_map_surface = pygame.surfarray.make_surface(np.stack([self.mark_map]*3, axis=-1))
        # semantic_surface = pygame.surfarray.make_surface(np.stack([self.semantic_map]*3, axis=-1))
        # visited = pygame.surfarray.make_surface(np.stack([self.visited_map*255]*3, axis=-1))

        # self.window.blit(peripheral_surface, (0, self.chart_height))
        # self.window.blit(visited, (self.vision_width, self.chart_height))
        # self.window.blit(foveal_surface, (self.vision_width, self.chart_height))
        # self.window.blit(pos_surface, (self.vision_width * 2, self.chart_height))
        # self.window.blit(mark_map_surface, (0, self.chart_height + self.vision_height))
        # self.window.blit(text_map_surface, (self.vision_width, self.chart_height + self.vision_height))
        # self.window.blit(semantic_surface, (self.vision_width, self.chart_height + self.vision_height))
        # self.window.blit(history_surface, (self.vision_width * 2, self.chart_height + self.vision_height))

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(render_fps)