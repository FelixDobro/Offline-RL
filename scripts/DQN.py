import copy
import os
import sys
from pathlib import Path
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

    tb = SummaryWriter(LOG_DIR, flush_secs=30)

    learning_model = QNet().to(DEVICE)
    optimizer = torch.optim.Adam(learning_model.parameters(), lr=LEARNING_RATE)
    target_model = copy.deepcopy(QNet()).to(DEVICE)
    target_model.eval()

    vector_env = AsyncVectorEnv(
        [make_cart_pole_env for _ in range(NUM_ENVS)],
        shared_memory=False
    )

    buffer = ReplayBuffer(BUFFER_SIZE)
    ITERATIONS = 0
    running_scores = np.zeros(NUM_ENVS)
    current_obs, _ = vector_env.reset()
    while True:
        ITERATIONS += 1
        learning_model.eval()

        obs_list, terminated_list, rewards_list, next_obs_list, action_list = [], [], [], [], []

        for i in range(SAMPLE_GEN):
            actions = e_greedy_action(current_obs, learning_model)
            next_obs, reward, terminated, truncated, _ = vector_env.step(actions)
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

        buffer.add(obs_list, rewards_list, terminated_list,next_obs_list, action_list)

        learning_model.train()

        for i in range(NUM_UPDATES):
            obs, reward, done, next_obs, actions = buffer.sample(BATCH_SIZE)

            qvals = learning_model(obs)
            chosen_q = torch.gather(qvals, dim=-1, index=actions)

            with torch.no_grad():
                target_qvals = target_model(next_obs)
                max_q, _ = torch.max(target_qvals, dim=-1, keepdim=True)

                targets = reward + GAMMA * (1-done) * max_q

            loss = F.mse_loss(chosen_q, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learning_model.parameters(), 0.1)
            optimizer.step()
            soft_update(target_model, learning_model, TAU)


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
