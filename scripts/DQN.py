import copy
import os
import sys
from pathlib import Path

# Project-level path configuration to allow absolute imports from parent directories
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.utils import make_cart_pole_env, ReplayBuffer, e_greedy_action, soft_update
import torch.nn.utils
from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from models.Q_net import QNet
from config import *
import numpy as np
import torch.nn.functional as F

if __name__ == '__main__':
    print(DEVICE)

    # Telemetry and logging initialization
    tb = SummaryWriter(LOG_DIR, flush_secs=30)

    # Model orchestration: Learning network and target network for Bellman stability
    learning_model = QNet().to(DEVICE)
    optimizer = torch.optim.Adam(learning_model.parameters(), lr=LEARNING_RATE)
    target_model = copy.deepcopy(QNet()).to(DEVICE)
    target_model.eval()

    # Parallelized environment setup for vectorized data collection
    vector_env = AsyncVectorEnv(
        [make_cart_pole_env for _ in range(NUM_ENVS)],
        shared_memory=False
    )

    buffer = ReplayBuffer(BUFFER_SIZE)
    ITERATIONS = 0
    running_scores = np.zeros(NUM_ENVS)
    current_obs, _ = vector_env.reset()

    # Main Training Loop
    while True:
        ITERATIONS += 1
        learning_model.eval()

        obs_list, terminated_list, rewards_list, next_obs_list, action_list = [], [], [], [], []

        # Data Acquisition: Interaction phase with the vectorized environments
        for i in range(SAMPLE_GEN):
            actions = e_greedy_action(current_obs, learning_model)
            next_obs, reward, terminated, truncated, _ = vector_env.step(actions)

            # Score tracking and episode logging logic
            running_scores += reward
            dones = terminated | truncated

            if np.any(dones):
                avg_score = running_scores[dones].mean()
                tb.add_scalar('Episode_Return', avg_score, ITERATIONS)
                running_scores[dones] = 0

            obs_list.append(current_obs)
            terminated_list.append(terminated)
            rewards_list.append(reward)
            next_obs_list.append(next_obs)
            current_obs = next_obs
            action_list.append(actions)

        # Off-policy storage: Committing gathered transitions to the replay buffer
        buffer.add(obs_list, rewards_list, terminated_list, next_obs_list, action_list)

        learning_model.train()

        # Optimization Phase: Minimizing the Temporal Difference (TD) Error
        for i in range(NUM_UPDATES):
            obs, reward, done, next_obs, actions = buffer.sample(BATCH_SIZE)

            # Compute current Q-values and filter by selected actions
            qvals = learning_model(obs)
            chosen_q = torch.gather(qvals, dim=-1, index=actions)

            # Compute Q-Targets using the target network (Double DQN logic)
            with torch.no_grad():
                target_qvals = target_model(next_obs)
                max_q, _ = torch.max(target_qvals, dim=-1, keepdim=True)
                targets = reward + GAMMA * (1 - done) * max_q

            # Gradient descent step
            loss = F.mse_loss(chosen_q, targets)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients in deep Q-networks
            torch.nn.utils.clip_grad_norm_(learning_model.parameters(), 0.1)
            optimizer.step()

            # Moving average update of target network parameters
            soft_update(target_model, learning_model, TAU)

        # Persistence: Periodic checkpointing of model and optimizer states
        if ITERATIONS % SAVE_EVERY == 0:
            torch.save(
                {
                    "model_state_dict": learning_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": ITERATIONS,
                },
                os.path.join(CHECKPOINTS_DIR, f"model{(ITERATIONS // SAVE_EVERY)}.pt"),
            )
            print("saved")