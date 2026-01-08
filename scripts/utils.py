import numpy as np
from config import *
import gymnasium as gym

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity

        self.obs = np.zeros((self.capacity, OBS_DIM), dtype=np.float32)
        self.terminated = np.zeros((self.capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 1), dtype=np.int64)
        self.position = 0
        self.size = 0

    def add(self, obs, reward, terminated, next_obs, actions):

        obs = np.array(obs).reshape(-1, OBS_DIM)
        next_obs = np.array(next_obs).reshape(-1, OBS_DIM)
        actions = np.array(actions).reshape(-1, 1)
        reward = np.array(reward).reshape(-1, 1)
        terminated = np.array(terminated).reshape(-1, 1)

        batch_size = obs.shape[0]

        if self.position + batch_size > self.capacity:

            first_chunk = self.capacity - self.position
            second_chunk = batch_size - first_chunk

            self.obs[self.position:] = obs[:first_chunk]
            self.actions[self.position:] = actions[:first_chunk]
            self.rewards[self.position:] = reward[:first_chunk]
            self.terminated[self.position:] = terminated[:first_chunk]
            self.next_obs[self.position:] = next_obs[:first_chunk]

            self.obs[:second_chunk] = obs[first_chunk:]
            self.actions[:second_chunk] = actions[first_chunk:]
            self.rewards[:second_chunk] = reward[first_chunk:]
            self.terminated[:second_chunk] = terminated[first_chunk:]
            self.next_obs[:second_chunk] = next_obs[first_chunk:]

            self.position = second_chunk
        else:

            idx_end = self.position + batch_size
            self.obs[self.position:idx_end] = obs
            self.actions[self.position:idx_end] = actions
            self.rewards[self.position:idx_end] = reward
            self.terminated[self.position:idx_end] = terminated
            self.next_obs[self.position:idx_end] = next_obs
            self.position = idx_end

        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self.size, size=batch_size)

        obs = torch.tensor(self.obs[indices], device=DEVICE)
        rewards = torch.tensor(self.rewards[indices], device=DEVICE)
        terminated = torch.tensor(self.terminated[indices], device=DEVICE)
        next_obs = torch.tensor(self.next_obs[indices], device=DEVICE)
        actions = torch.tensor(self.actions[indices], device=DEVICE)

        return obs, rewards, terminated, next_obs, actions


def make_cart_pole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env

@torch.no_grad()
def soft_update(target_model, model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

@torch.no_grad()
def e_greedy_action(obs, model, eps=0.05):
    if np.random.random() < eps:
        return np.random.choice(a=[0,1], size=obs.shape[0])
    obs_tensor = torch.tensor(obs, device=DEVICE)
    q_values = model(obs_tensor)
    actions = torch.argmax(q_values, dim=-1)
    return actions.detach().cpu().numpy()

