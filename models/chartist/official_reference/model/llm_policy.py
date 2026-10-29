import os
import re
import ast
import httpx
import json
import time
from openai import OpenAI
from model.search_env import SearchEnv
from model.map_env import MapEnv
from model.read_env import ReadEnv
from model.title_env import TitleEnv
import random
import numpy as np
from stable_baselines3 import PPO

ACTION = 20

class LLMPolicy:
    def __init__(self, meta = None, search_model = None, map_model = None, read_model = None, title_model = None, MEM_LIMIT = 30, MAX_STEPS = 5):
        os.environ['AALTO_OPENAI_API_KEY'] = '8e936d90c1cf497ab28f5fc301981d64'
        assert (
            "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
        ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
        self.client = OpenAI(
            base_url="https://aalto-openai-apigw.azure-api.net",
            api_key=False, # API key not used, and rather set below
            default_headers = {
                "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
            },
            http_client=httpx.Client(
                event_hooks={ "request": [self.update_base_url] }
            ),
        )
        if search_model is not None:
            self.search_model = search_model
        else:
            self.search_model = PPO.load("model/vision_search.pt", device="auto")
        if map_model is not None:
            self.map_model = map_model
        else:
            self.map_model = PPO.load("model/vision_map.pt", device="auto")
        if read_model is not None:
            self.read_model = read_model
        else:
            self.read_model = PPO.load("model/vision_read.pt", device="auto")
        if title_model is not None:
            self.title_model = title_model
        else:
            self.title_model = PPO.load("model/vision_title.pt", device="auto")
        
        self.steps = 0
        self.max_steps = MAX_STEPS
        self.gaze_history = []
        self.action_history = []
        self.memory = [] # vision memory
        self.memory_in_text = ""
        self.memory_limit = MEM_LIMIT
        self.scanpath_xs = []
        self.scanpath_ys = []
        self.answer_to_the_task = {}
        self.visited_map = np.zeros([ACTION,ACTION],dtype=np.uint8) # visited positions

    def update_base_url(self, request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            request.url = request.url.copy_with(path="/v1/chat/gpt4-8k") # chat/gpt4-8k /chat

    def setup(self, meta = None, meta_json = 'meta.json', folder = "images"):
        self.meta = meta
        self.task = meta["question"]
        self.chart = "a scatterplot" # for example, a vertical bar chart, a line chart
        self.steps = 0
        self.gaze_history = []
        self.action_history = []
        self.memory = []
        self.memory_in_text = "no information"
        self.answer_to_the_task = {}
        self.scanpath_xs = []
        self.scanpath_ys = []
        self.visited_map = np.zeros([ACTION,ACTION],dtype=np.uint8) # visited positions
        self.search_env = SearchEnv(meta, meta_json = meta_json, folder = folder)
        self.map_env = MapEnv(meta, meta_json = meta_json, folder = folder)
        self.read_env = ReadEnv(meta, meta_json = meta_json, folder = folder)
        self.title_env = TitleEnv(meta, meta_json = meta_json, folder = folder)
        self.message = [
            {
                "role": "system", 
                "content": "You are looking at %s, and your task is \"%s\". \
                    Your goal is to read the chart and solve the task as less actions as possible. \
                    You can move your gaze to gather information from the chart. \
                    After each step, you get information from the chart and it will also update your memory. \
                    The memory is a list of the information you have gathered from the chart with position (x, y), action, and corresponding text. \
                    The position means where is the gaze on is from (0, 0) to (19, 19). It's from left-top to right-bottom from the chart image. \
                    You can only solve the task by using the information in your memory. \
                    Initially, you have no idea about any information in the chart, so the memory will be an empty list.\
                    There are five actions in total, you can use any of them in any order you want: \
                    1. explore(): explore information in the chart if you have no idea about it. \
                    2. search_text(): visually searching for an item label or a value label related to the task. This is highly focus on task itself. \
                    3. look_at_data_point(ref_pos): visually searching for a bar of the data point given a reference label. This approach is used to look at data points in the chart. \
                    4. read_axis(ref_pos): visually searching to read the associated value or text label of a given mark. This is used to read value of a related bar. \
                    5. answer(): stop gaze movement, and answer the task question. \
                    You can use these actions to solve the task. It is okay to reuse each of the action. \
                    If the action is answer(), it means you can solve the question with the information you have."%(self.chart, self.task)
            },
            {
                "role": "user", 
                "content": "Your current memory is empty. Please decide your next action just based on the memory. \
                    Ensure the action must be a json format with function name and its parameter. The parameter is a coordinate (x,y). If there is no parameter, you can give an empty string. \
                    For example: {\"function\": \"look_at_data_point\", \"parameter\": \"(19, 19)\"}"
            }
        ]

    def update_memory(self, info):
        if len(self.memory) >= self.memory_limit:
            t = len(self.memory)
            x = []
            for i in range(t):
                x.append(0.1 * t)
                t -= 1
            def softmax(x):
                exp_x = np.exp(x)
                return exp_x / np.sum(exp_x, axis=0)
            softmax_probs = softmax(x)
            # print("Softmax probabilities:", softmax_probs)
            idx = np.random.choice(len(self.memory), p=softmax_probs)
            # print("Forget the memory at index %d"%(idx))
            self.memory.pop(idx)

        self.memory.append(info)
        # print("Memory: %s"%(self.memory))
        self.visited_map[info["position"][0]][info["position"][1]] = 1

    def memory_summary(self):
        message = [
            {
                "role": "system", 
                "content": "You are looking at %s, and your task is \"%s\". \
                    Your goal is to read the chart and solve the task as less actions as possible. \
                    You can move your gaze to gather information from the chart. \
                    After each step, you get information from the chart and it will also update your memory. \
                    The memory is a list of the information in patches you have gathered from the chart with position (x, y), action, and corresponding text. \
                    The position means where is the gaze on is from (0, 0) to (19, 19). It's from left-top to right-bottom from the chart image. \
                    You can only solve the task by using the information in your memory. "%(self.chart, self.task)
            },
            {
                "role": "user", 
                "content": "Your current memory in order is %s. Please summarize the memory into one paragraph."%(self.memory)
            }
        ]
        # completion = self.client.chat.completions.create(
        #     model="no_effect", # the model variable must be set, but has no effect, model selection done with URL
        #     messages=message,
        # )
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.message
        )
        self.memory_in_text = completion.choices[0].message.content
        # print("now in memory: %s"%(self.memory_in_text))

    def action(self):
        self.steps += 1
        if self.steps > self.max_steps:
            self.answer()
            return True
        self.message[-1] = {
            "role": "user", 
            "content": "Your current memory including action and information in order is %s. In summary, the information in your memory: %s. You action history in tenmporal order is %s. Please decide your next action just based on the memory. \
                Your goal is to read the chart and solve the task as less actions as possible. \
                Ensure the action must be a json format can be directly parsed with function name and its parameter. \
                The parameter is a coordinate (x,y). If there is no parameter, you can give an empty string. \
                Please also give a reason for your action. \
                Please ensure look_at_data_point and read_axis actions have a parameter. \
                For example: {\"function\": \"look_at_data_point\", \"parameter\": \"(19, 19)\", \"reason\": \"i know the position of the referenced label, need to position the mark\" \
                Please only provide given function! and follow the format of the example! If some functions a seldomly occur in your memory, you are encouraged to use it!\}"%(self.memory, self.memory_in_text, self.action_history) 
                #  ,   
        }
        # print("====================================")
        max_num_requests = 5
        for attempt in range(max_num_requests):
            completion = self.client.chat.completions.create(
                model="gpt-4o", # mini
                messages=self.message
            )
            message_content = completion.choices[0].message.content
            try:
                output = json.loads(message_content)
                self.action_history.append("I did %s(%s) with the reason: %s."%(output["function"], output["parameter"], output['reason']))
                # time.sleep(1)
                if output["function"] == "explore":
                    self.explore()
                    self.memory_summary()
                    return False
                elif output["function"] == "search_text":
                    self.search_text()
                    self.memory_summary()
                    return False
                elif output["function"] == "look_at_data_point":
                    position = [ast.literal_eval(i) for i in re.findall(r'\(.*?\)', output["parameter"])]
                    self.look_at_data_point(position[0])
                    self.memory_summary()
                    return False
                elif output["function"] == "read_axis":
                    position = [ast.literal_eval(i) for i in re.findall(r'\(.*?\)', output["parameter"])]
                    self.read_axis(position[0])
                    self.memory_summary()
                    return False
            except:
                output = {"function": "error", "parameter": ""}
                self.message[-1]["content"] += "Please only provide a valid json format can be directly parsed without any additional text!"
                if attempt == 5:
                    print("Error: Invalid function: %s"%(output))
                    return True
    
    def explore(self):
        obs = self.title_env.reset(self.meta)
        done = False
        while not done:
            action, _states = self.title_model.predict(obs, deterministic=True)
            obs, reward, done, info = self.title_env.step(action)
            x = random.uniform(action[1]*self.title_env.gaze_size, (action[1]+1)*self.title_env.gaze_size)
            y = random.uniform(action[0]*self.title_env.gaze_size, (action[0]+1)*self.title_env.gaze_size)
            self.scanpath_xs.append(x)
            self.scanpath_ys.append(y)
            self.gaze_history.append(action)
            self.update_memory(info)

    def search_text(self):
        obs = self.search_env.reset(self.meta, visited=self.visited_map)
        done = False
        while not done:
            action, _states = self.search_model.predict(obs, deterministic=True)
            obs, reward, done, info = self.search_env.step(action)
            x = random.uniform(action[1]*self.search_env.gaze_size, (action[1]+1)*self.search_env.gaze_size)
            y = random.uniform(action[0]*self.search_env.gaze_size, (action[0]+1)*self.search_env.gaze_size)
            self.scanpath_xs.append(x)
            self.scanpath_ys.append(y)
            self.gaze_history.append(action)
            self.update_memory(info)
        

    def look_at_data_point(self, ref_pos):
        reference_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        reference_map[ref_pos[0]][ref_pos[1]] = 1
        obs = self.map_env.reset(self.meta, visited=self.visited_map, reference=reference_map)
        done = False
        while not done:
            obs = self.map_env.observation()
            action, _states = self.map_model.predict(obs)
            obs, r, done, info = self.map_env.step(action)
            x = random.uniform(action[1]*self.map_env.gaze_size, (action[1]+1)*self.map_env.gaze_size)
            y = random.uniform(action[0]*self.map_env.gaze_size, (action[0]+1)*self.map_env.gaze_size)
            self.scanpath_xs.append(x)
            self.scanpath_ys.append(y)
            self.gaze_history.append(action)
            self.update_memory(info)

    def read_axis(self, ref_pos):
        reference_map = np.zeros([ACTION,ACTION],dtype=np.uint8)
        reference_map[ref_pos[0]][ref_pos[1]] = 1
        self.read_env.reset(self.meta, visited=self.visited_map, reference=reference_map)
        done = False
        while not done:
            obs = self.read_env.observation()
            action, _states = self.read_model.predict(obs)
            obs, r, done, info = self.read_env.step(action)
            x = random.uniform(action[1]*self.read_env.gaze_size, (action[1]+1)*self.read_env.gaze_size)
            y = random.uniform(action[0]*self.read_env.gaze_size, (action[0]+1)*self.read_env.gaze_size)
            self.scanpath_xs.append(x)
            self.scanpath_ys.append(y)
            self.gaze_history.append(action)
            self.update_memory(info)

    def answer(self):
        self.message[-1] = {
            "role": "user", 
            "content": "Your current memory is %s. \
                Please answer the question: %s. The answer should follow json format, follows \{\"answer\":\"xxx\", \"reason\":\"xxx\"\}. \
                The value of \"answer\" should be a value or a text read from chart, if it's a value between two numbers, you can randomly pick one from them.\
                The value of \"reason\" is to explain how you get the answer."%(self.memory, self.task) # You action history is %s. self.action_history, 
        }
        max_num_requests = 5
        for attempt in range(max_num_requests):
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.message
            )
            message_content = completion.choices[0].message.content
            try:
                output = json.loads(message_content)
                self.answer_to_the_task = output
                break
            except:
                output = {"function": "error", "parameter": ""}
                self.message[-1]["content"] += "Please only provide a valid json format without any additional text!"
                time.sleep(1)