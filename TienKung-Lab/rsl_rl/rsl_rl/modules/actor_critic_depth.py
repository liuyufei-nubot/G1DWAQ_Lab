# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal


class DepthOnlyFCBackbone58x87(nn.Module):
    """CNN backbone for processing depth or RGB images."""
    def __init__(self, output_dim, output_activation=None, in_channels=1, input_height=64, input_width=64):
        super().__init__()

        self.in_channels = in_channels
        self.output_dim = output_dim
        activation = nn.ELU()
        
        # Calculate output size after conv layers
        # For 64x64 input: after conv(8,4)->15x15, pool(2,2)->7x7, conv(3,1)->5x5
        self.image_compression = nn.Sequential(
            # [in_channels, H, W]
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            dummy_output = self.image_compression(dummy_input)
            flatten_size = dummy_output.shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 128),
            activation,
            nn.Linear(128, output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        # images: [batch, channels, H, W] or [batch, H, W] for single channel
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dim
        images_compressed = self.image_compression(images)
        latent = self.fc(images_compressed)
        latent = self.output_activation(latent)
        return latent


class RGBEncoder(nn.Module):
    """Simple RGB encoder without temporal stacking."""
    def __init__(self, output_dim=128, input_height=64, input_width=64) -> None:
        super().__init__()
        activation = nn.ELU()
        
        self.backbone = nn.Sequential(
            # [3, H, W]
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_height, input_width)
            dummy_output = self.backbone(dummy_input)
            flatten_size = dummy_output.shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 128),
            activation,
            nn.Linear(128, output_dim),
            activation
        )
        self.output_dim = output_dim
        
    def forward(self, rgb_image):
        # rgb_image: [batch, H, W, 3] -> need to permute to [batch, 3, H, W]
        if rgb_image.dim() == 4 and rgb_image.shape[-1] == 3:
            rgb_image = rgb_image.permute(0, 3, 1, 2)  # [B, H, W, 3] -> [B, 3, H, W]
        features = self.backbone(rgb_image)
        latent = self.fc(features)
        return latent

class ActorCriticDepth(nn.Module):
    """Actor-Critic network with RGB image encoder."""
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        his_encoder_dims=[1024, 512, 128],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        his_latent_dim = 64 + 3,
                        history_dim = 570,
                        rgb_latent_dim = 128,
                        rgb_height = 64,
                        rgb_width = 64,
                        use_rgb = True,
                        activation='elu',
                        init_noise_std=1.0,
                        max_grad_norm=10.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticDepth.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDepth, self).__init__()
        activation = get_activation(activation)

        self.his_latent_dim = his_latent_dim
        self.rgb_latent_dim = rgb_latent_dim
        self.use_rgb = use_rgb
        self.max_grad_norm = max_grad_norm        

        # RGB encoder (only if using RGB)
        if self.use_rgb:
            self.rgb_encoder = RGBEncoder(output_dim=rgb_latent_dim, input_height=rgb_height, input_width=rgb_width)
            mlp_input_dim_a = num_actor_obs + his_latent_dim + rgb_latent_dim
        else:
            self.rgb_encoder = None
            mlp_input_dim_a = num_actor_obs + his_latent_dim
        
        mlp_input_dim_c = num_critic_obs + his_latent_dim
        
        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, his_encoder_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(his_encoder_dims)):
            if l == len(his_encoder_dims) - 1:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_latent_dim))
            else:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_encoder_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, history, rgb_image=None, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        if rgb_image is not None:
            rgb_feature = self.rgb_encoder(rgb_image)
            actor_input = torch.cat((observations, his_feature, rgb_feature), dim=-1)
        else:
            actor_input = torch.cat((observations, his_feature), dim=-1)
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, rgb_image=None, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        if rgb_image is not None:
            rgb_feature = self.rgb_encoder(rgb_image)
            actor_input = torch.cat((observations, his_feature, rgb_feature), dim=-1)
        else:
            actor_input = torch.cat((observations, his_feature), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def evaluate(self, critic_observations, history, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        actor_input = torch.cat((critic_observations, his_feature), dim=-1)
        value = self.critic(actor_input)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
