import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from model.search_env import SearchEnv
from model.map_env import MapEnv
from model.read_env import ReadEnv
from model.title_env import TitleEnv
from model.search_model import Search
from model.map_model import Map
from model.read_model import Read
from model.title_model import Title

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--search", action="store_true", default=False, help="train search model")
parser.add_argument("--map", action="store_true", default=False, help="train map model")
parser.add_argument("--read", action="store_true", default=False, help="train read model")
parser.add_argument("--title", action="store_true", default=False, help="train title model")
parser.add_argument("--total-timesteps", type=int, default=2000000, help="total timesteps for training the agent")

args = parser.parse_args()

total_timesteps=args.total_timesteps

folder = "images_real"
meta_json = 'meta_real.json'
meta_infos = json.load(open(meta_json))
meta_info = meta_infos[0]

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

if args.map:
    env = MapEnv(meta_info, meta_json = meta_json, folder=folder)
    env = Monitor(env, log_dir)
    policy_kwargs = dict(
        features_extractor_class=Map,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model_path = 'model/vision_map_real.pt'
elif args.search:
    env = SearchEnv(meta_info, meta_json = meta_json, folder=folder)
    env = Monitor(env, log_dir)
    policy_kwargs = dict(
        features_extractor_class=Search,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model_path = 'model/vision_search_real.pt'
elif args.read:
    env = ReadEnv(meta_info, meta_json = meta_json, folder=folder)
    env = Monitor(env, log_dir)
    policy_kwargs = dict(
        features_extractor_class=Read,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model_path = 'model/vision_read_real.pt'
elif args.title:
    env = TitleEnv(meta_info, meta_json = meta_json, folder=folder)
    env = Monitor(env, log_dir)
    policy_kwargs = dict(
        features_extractor_class=Title,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model_path = 'model/vision_title_real.pt'
else:
    raise ValueError("Please specify a model to train")

# model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="runs")
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="runs")

model.learn(
    total_timesteps=total_timesteps, 
    progress_bar=True
)

model.save(model_path)