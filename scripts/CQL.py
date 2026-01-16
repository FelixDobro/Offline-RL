import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gymnasium.vector import AsyncVectorEnv
from scripts.utils import make_cart_pole_env, soft_update
import copy
import os
from torch.utils.data import Dataset, DataLoader
import pickle
from scripts.evaluate import evaluate_model
from torch.utils.tensorboard import SummaryWriter
from config import *
from models.Q_net import QNet
import torch.nn.functional as F

class OfflineDataset(Dataset):
    def __init__(self, data):
        self.obs = data["observations"]
        self.actions = data["actions"]
        self.rewards = data["rewards"] * REWARD_SCALE
        self.terminated = data["terminals"]
        self.next_obs = data["next_observations"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):

        return self.obs[idx], self.rewards[idx], self.terminated[idx], self.next_obs[idx], self.actions[idx]

if __name__ == "__main__":
    print(DEVICE)
    tb = SummaryWriter(LOG_DIR, flush_secs=30)

    model = QNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    target_model = copy.deepcopy(model).to(DEVICE)
    target_model.eval()

    with open(f"{DATA_DIR}.pkl", mode='rb') as f:
        data = pickle.load(f)

    dataset = OfflineDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    vector_env = AsyncVectorEnv(
        [make_cart_pole_env for _ in range(NUM_ENVS)],
        shared_memory=False
    )

    num_batches = 0
    for i in range(NUM_EPOCHS):
        epoch_loss_dqn = 0
        epoch_loss_CQL = 0
        epoch_loss_total = 0
        model.train()
        for batch in dataloader:

            obs, reward, done, next_obs, actions = batch
            obs = obs.to(DEVICE)
            actions = actions.to(DEVICE)
            reward = reward.to(DEVICE)
            done = done.to(DEVICE)
            next_obs = next_obs.to(DEVICE)
            qvals = model(obs)

            chosen_q = torch.gather(qvals, dim=-1, index=actions)

            with torch.no_grad():
                target_qvals = target_model(next_obs)
                max_q, _ = torch.max(target_qvals, dim=-1, keepdim=True)
                targets = reward + GAMMA * (1 - done) * max_q

            loss_dqn = F.mse_loss(chosen_q, targets)
            logsumexp_q = torch.logsumexp(qvals, dim=1).unsqueeze(1)
            cql_loss = (logsumexp_q - chosen_q).mean()

            total_loss = loss_dqn + ALPHA * cql_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            soft_update(target_model, model, tau=TAU)

            '''if num_batches % CHECK_MODEL == 0:
                model.eval()
                avg_return, _ = evaluate_model(vector_env, model)
                tb.add_scalar("Batch/AVG Return", avg_return, num_batches)
                tb.add_scalar("Batch/Mean Q-Values", qvals.mean().item(), num_batches)
                model.train()
'''

            if num_batches % SAVE_EVERY == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": i,
                    },
                    os.path.join(CHECKPOINTS_DIR, f"model{(num_batches // SAVE_EVERY)}.pt"),
                )
                print("saved")


            epoch_loss_dqn += loss_dqn.item()
            epoch_loss_CQL += cql_loss.item()
            epoch_loss_total += total_loss.item()
            tb.add_scalar("Batch/total loss", total_loss.item(), num_batches)
            tb.add_scalar("Batch/CQL loss", cql_loss.item(), num_batches)
            tb.add_scalar("Batch/DQN loss", total_loss.item(), num_batches)
            tb.add_scalar("Batch/Qvals mean", qvals.mean().item(), num_batches)
            num_batches += 1


        tb.add_scalar("Epoch/Total Loss", epoch_loss_total, i)
        tb.add_scalar("Epoch/CQL Loss", epoch_loss_CQL, i)
        tb.add_scalar("Epoch/DQN Loss", epoch_loss_dqn, i)
        model.eval()
        avg_return, _ = evaluate_model(vector_env, model)
        tb.add_scalar("Epoch/AVG Return", avg_return, i)