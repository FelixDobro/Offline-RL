import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from config import *
from models.Q_net import QNet
from scripts.utils import ReplayBuffer, e_greedy_action, make_cart_pole_env

if __name__ == '__main__':

    model = QNet()
    if MODEL_VERSION != 0:
        model_props = torch.load(f"{MODEL_DIR}")
        state_dict = model_props["model_state_dict"]
        model.load_state_dict(state_dict)
    model.eval()

    vector_env = AsyncVectorEnv(
        [make_cart_pole_env for _ in range(NUM_ENVS)],
        shared_memory=False
    )

    current_obs,_ = vector_env.reset()
    buffer = ReplayBuffer(NUM_SAMPLES)

    while buffer.size < NUM_SAMPLES:
        obs_list, terminated_list, rewards_list, next_obs_list, action_list = [], [], [], [], []
        for i in range(SAMPLE_GEN):
            actions = e_greedy_action(current_obs, model)
            next_obs, reward, terminated, truncated, _ = vector_env.step(actions)

            obs_list.append(current_obs)
            terminated_list.append(terminated)
            rewards_list.append(reward)
            next_obs_list.append(next_obs)
            action_list.append(actions)

            current_obs = next_obs

        buffer.add(obs_list, rewards_list, terminated_list, next_obs_list, action_list)

    data = {
        "observations": buffer.obs,
        "actions": buffer.actions,
        "rewards": buffer.rewards,
        "next_observations": buffer.next_obs,
        "terminals": buffer.terminated
    }
    with open(f"{DATA_DIR}.pkl", "wb") as f:
        pickle.dump(data, f)