import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import make_cart_pole_env, e_greedy_action
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from config import *
from models.Q_net import QNet


def evaluate_model(vector_env, model):

    current_obs, _ = vector_env.reset()

    returns = []
    running_scores = np.zeros(NUM_ENVS)

    for i in range(NUMBER_OF_EVAL_STEPS):
        for i in range(SAMPLE_GEN):
            actions = e_greedy_action(current_obs, model, eps=-1)
            next_obs, reward, terminated, truncated, _ = vector_env.step(actions)
            running_scores += reward
            dones = terminated | truncated

            if np.any(dones):
                scores = running_scores[dones]
                for score in scores:
                    returns.append(score)
                running_scores[dones] = 0

            current_obs = next_obs

    avg_return = np.mean(returns)

    print(f"Number of returns {len(returns)}")
    print(f"Average Return {avg_return}")

    return avg_return, len(returns)

if __name__ == "__main__":
    print(DEVICE)

    vector_env = AsyncVectorEnv(
        [make_cart_pole_env for _ in range(NUM_ENVS)],
        shared_memory=False
    )
    model = QNet().to(DEVICE)
    if MODEL_VERSION:
        model_props = torch.load(f"{MODEL_DIR}")
        state_dict = model_props["model_state_dict"]
        model.load_state_dict(state_dict)
    model.eval()

    evaluate_model(vector_env, model)

