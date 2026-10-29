import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Read(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(Read, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 4
        self.cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=3, padding=1),  # (20x20x4) -> (20x20x32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (20x20x32) -> (10x10x32)

            # Second convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # (10x10x32) -> (10x10x64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (10x10x64) -> (5x5x64)

            # Flatten the tensor
            nn.Flatten(),  # (5x5x64) -> (1600)
        )
        n_flatten_image = 1600
        self.vis_transform = nn.Sequential(nn.Linear(n_flatten_image, features_dim), nn.ReLU())

        text_embedding_dim = 768
        self.task_transform = nn.Sequential(nn.Linear(text_embedding_dim, features_dim), nn.ReLU())

        self.joint_embedding = nn.Sequential(nn.Linear(features_dim * 2, features_dim), nn.ReLU())

    # def forward(self, observations: th.Tensor) -> th.Tensor:
    def forward(self, obs) -> th.Tensor:
        fixation = obs['fixation']
        visited = obs['visited']
        reference_map = obs['reference']
        peripheral = obs['peripheral']
        vis_feature= th.stack((fixation, visited, reference_map, peripheral), axis=1) # self.mark_map, self.text_map, 
        vis_feature = self.vis_transform(self.cnn(vis_feature))

        task = obs['task']
        task_feature = self.task_transform(task)

        joint_embedding = self.joint_embedding(th.cat([vis_feature, task_feature], dim=1))

        return joint_embedding