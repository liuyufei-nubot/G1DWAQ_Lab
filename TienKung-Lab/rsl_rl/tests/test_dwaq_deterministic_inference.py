# Copyright (c) 2026, The G1 DWAQ Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause License.

"""Regression tests for deterministic DWAQ inference."""

import torch

from rsl_rl.modules import ActorCritic_DWAQ


def test_dwaq_inference_uses_encoder_means():
    torch.manual_seed(42)
    policy = ActorCritic_DWAQ(
        num_actor_obs=119,
        num_critic_obs=311,
        num_actions=29,
        cenet_in_dim=500,
        cenet_out_dim=19,
        obs_dim=100,
    ).eval()
    observations = torch.randn(8, 100)
    observation_history = torch.randn(8, 500)

    with torch.inference_mode():
        first_actions = policy.act_inference(observations, observation_history)
        second_actions = policy.act_inference(observations, observation_history)

        encoded = policy.encoder(observation_history)
        expected_code = torch.cat(
            (policy.encode_mean_vel(encoded), policy.encode_mean_latent(encoded)), dim=-1
        )
        expected_actions = policy.actor(torch.cat((expected_code, observations), dim=-1))

    torch.testing.assert_close(first_actions, second_actions, rtol=0, atol=0)
    torch.testing.assert_close(first_actions, expected_actions, rtol=0, atol=0)
