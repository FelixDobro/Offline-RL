import sys
from pathlib import Path

# Module path configuration for cross-directory imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.utils import make_cart_pole_env, e_greedy_action
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from config import *
from models.Q_net import QNet


def evaluate_model(vector_env, model):
    """
    Executes a standardized evaluation pass over a specified number of episodes.
    Calculates the mean return across vectorized environments to assess policy performance.
    """
    current_obs, _ = vector_env.reset()

    returns = []
    running_scores = np.zeros(NUM_ENVS)
    finished = 0

    # Stochastic evaluation loop until EVAL_EPISODES threshold is reached
    while finished < EVAL_EPISODES:
        actions = e_greedy_action(current_obs, model, eps=EPSILON)
        next_obs, reward, terminated, truncated, _ = vector_env.step(actions)

        running_scores += reward
        dones = terminated | truncated

        # Process completions for vectorized environments
        if np.any(dones):
            scores = running_scores[dones]
            finished += len(scores)
            for score in scores:
                returns.append(score)

            # Reset trackers for environments that reached a terminal state
            running_scores[dones] = 0

        current_obs = next_obs

    avg_return = np.mean(returns)

    print(f"Number of returns {len(returns)}")
    print(f"Average Return {avg_return}")

    return avg_return, len(returns)


if __name__ == "__main__":
    print(DEVICE)

    # Initialize parallel environments for faster sampling during evaluation
    vector_env = AsyncVectorEnv(
        [make_cart_pole_env for _ in range(NUM_ENVS)],
        shared_memory=False
    )

    model = QNet().to(DEVICE)

    # State restoration: Load weights if a specific model version is defined in config
    if MODEL_VERSION:
        model_props = torch.load(f"{MODEL_DIR}")
        state_dict = model_props["model_state_dict"]
        model.load_state_dict(state_dict)

    model.eval()

    # Execute evaluation pipeline
    evaluate_model(vector_env, model)