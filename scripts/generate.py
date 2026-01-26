import pickle
import sys
from pathlib import Path

# Resolve project root for absolute imports across the repository structure
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from config import *
from models.Q_net import QNet
from scripts.utils import ReplayBuffer, e_greedy_action, make_cart_pole_env

if __name__ == '__main__':
    print(DEVICE)

    # Policy initialization: Load specific model checkpoint if defined in configuration
    model = QNet().to(DEVICE)
    if MODEL_VERSION != 0:
        model_props = torch.load(f"{MODEL_DIR}")
        state_dict = model_props["model_state_dict"]
        model.load_state_dict(state_dict)
    model.eval()

    # Parallelized environment orchestration for high-throughput data acquisition
    vector_env = AsyncVectorEnv(
        [make_cart_pole_env for _ in range(NUM_ENVS)],
        shared_memory=False
    )

    current_obs, _ = vector_env.reset()

    # Initialize replay memory for static dataset generation
    buffer = ReplayBuffer(NUM_SAMPLES)

    # Main sampling loop: Populating the buffer until the required sample count is met
    while buffer.size < NUM_SAMPLES:
        obs_list, terminated_list, rewards_list, next_obs_list, action_list = [], [], [], [], []

        # Batch generation phase: Execute vectorized steps to collect transitions
        for i in range(SAMPLE_GEN):
            actions = e_greedy_action(current_obs, model, eps=EPSILON)
            next_obs, reward, terminated, truncated, _ = vector_env.step(actions)

            # Accumulate step-wise data for bulk buffer insertion
            obs_list.append(current_obs)
            terminated_list.append(terminated)
            rewards_list.append(reward)
            next_obs_list.append(next_obs)
            action_list.append(actions)

            current_obs = next_obs

        # Update replay memory with collected trajectory segments
        buffer.add(obs_list, rewards_list, terminated_list, next_obs_list, action_list)

    # Serialization: Exporting the buffer as a structured dataset for offline RL training
    data = {
        "observations": buffer.obs,
        "actions": buffer.actions,
        "rewards": buffer.rewards,
        "next_observations": buffer.next_obs,
        "terminals": buffer.terminated
    }

    with open(f"{DATA_DIR}.pkl", "wb") as f:
        pickle.dump(data, f)