from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic_DWAQ(nn.Module):
    """Actor-Critic with DWAQ (Deep Variational Autoencoder for Walking) context encoder.
    
    The context encoder (β-VAE) infers velocity and latent state from observation history.
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cenet_in_dim: int,
        cenet_out_dim: int,
        obs_dim: int,
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim

        self.activation = get_activation(activation)
        actor_input_dim = num_actor_obs
        critic_input_dim = num_critic_obs

        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim,128),
            self.activation,
            nn.Linear(128,64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_logvar_latent = nn.Linear(64,cenet_out_dim-3)
        self.encode_mean_vel = nn.Linear(64,3)
        self.encode_logvar_vel = nn.Linear(64,3)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim,64),
            self.activation,
            nn.Linear(64,128),
            self.activation,
            nn.Linear(128,self.obs_dim)
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def reparameterise(self, mean, logvar):
        """Reparameterization trick for VAE with numerical stability.
        
        Clamps logvar to prevent exp() overflow which can cause NaN.
        logvar in [-10, 10] corresponds to std in [~0.007, ~148].
        """
        # Clamp logvar to prevent numerical instability
        # exp(10) ≈ 22026, exp(-10) ≈ 4.5e-5, both are safe
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        var = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(var)
        code = mean + var * code_temp
        return code
    
    def cenet_forward(self, obs_history: torch.Tensor):
        """Forward pass through the context encoder (β-VAE).
        
        Args:
            obs_history: Flattened observation history [batch, history_len * obs_dim]
            
        Returns:
            code: Concatenated latent code [vel(3) + latent(16)] for actor
            code_vel: Predicted velocity (reparameterized)
            decode: Reconstructed observation
            mean_vel: Velocity encoder mean
            logvar_vel: Velocity encoder log variance
            mean_latent: Latent encoder mean  
            logvar_latent: Latent encoder log variance
        """
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        code_latent = self.reparameterise(mean_latent,logvar_latent)
        code_vel = self.reparameterise(mean_vel,logvar_vel)
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        return code,code_vel,decode,mean_vel,logvar_vel,mean_latent,logvar_latent

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
        # sanitize mean: replace NaN/Inf with numeric values
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)

        # prepare std: ensure positive and same shape as mean
        std_param = self.std
        # clamp learned std to reasonable range to avoid numerical issues
        std_param = torch.clamp(std_param, min=1e-6, max=1e3)
        std = std_param.unsqueeze(0).expand_as(mean)

        # final safety: if any non-finite remain, replace and log
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            # print diagnostics for debugging
            print('Warning: non-finite values in distribution params',
                  'mean_any_nan', torch.isnan(mean).any().item(),
                  'mean_any_inf', torch.isinf(mean).any().item(),
                  'std_any_nan', torch.isnan(std).any().item(),
                  'std_any_inf', torch.isinf(std).any().item())
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
            std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-3))

        self.distribution = Normal(mean, std)

    def act(self, observations, obs_history, **kwargs):
        """Compute actions from observations and history.
        
        Args:
            observations: Current actor observations
            obs_history: Observation history for context encoder
            
        Returns:
            Sampled actions from policy distribution
        """
        code, _, _, _, _, _, _ = self.cenet_forward(obs_history)
        observations = torch.cat((code, observations), dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        """Compute deterministic actions for inference.
        
        Args:
            observations: Current actor observations
            obs_history: Observation history for context encoder
            
        Returns:
            Mean actions (deterministic)
        """
        code, _, _, _, _, _, _ = self.cenet_forward(obs_history)
        observations = torch.cat((code, observations), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate critic value for given observations."""
        value = self.critic(critic_observations)
        return value


def get_activation(act_name: str) -> nn.Module | None:
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
