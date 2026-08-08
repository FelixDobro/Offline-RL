# Offline Reinforcement Learning on CartPole


This repository yields the Code for a university project in Deep Learning at the technical university of Vienna. The objective is to learn the Cartpole environment using Offline Reinforcement Learning methods.

**Deep Q-Networks (DQN)** and **Conservative Q-Learning (CQL)** to solve the CartPole environment entirely offline. 

In Offline Reinforcement Learning, the agent learns exclusively from a static dataset without interacting with the environment. This introduces significant challenges, primarily **distributional shift (Out-of-Distribution states)** and **overestimation bias** of action-values.

## 🧠 The Approach: Conservative Q-Learning (CQL)

Standard DQN updates suffer in offline settings because the max operator in the Bellman equation overestimates values for states not present in the dataset:
$$Target = r + \gamma * \max q' * (1-d)$$ 
$$Loss_{DQN} = MSE(q, Target)$$

To mitigate this, I implemented **Conservative Q-Learning (CQL)**[cite: 12]. CQL penalizes large Q-values and reinforces the action values actually present in the dataset, pushing down overestimations[cite: 12]:

$$Loss_{CQL} = \text{mean}(LSE(q) - q[a])$$
$$Loss_{Total} = \alpha * Loss_{CQL} + Loss_{DQN}$$

The hyperparameter $\alpha$ controls how conservative the network acts[cite: 12].

## 📊 Environment & Datasets

*   **State Space:** 4 dimensions `[position, speed, pole-angle, pole-speed]`[cite: 12].
*   **Actions:** `0` (Push left), `1` (Push right)[cite: 12].
*   **Reward:** `+1` for every surviving step[cite: 12].

I generated three static datasets (100,000 transitions each) using different Online-DQN policies to test the offline algorithms[cite: 12]:
1.  **Random Dataset** (Average Return $\approx$ 20)[cite: 12]
2.  **Mid Dataset** (Average Return $\approx$ 250)[cite: 12]
3.  **Perfect Dataset** (Average Return = 500)[cite: 12]

## 🚀 Results & Evaluation

The evaluation (average return over 20+ episodes) shows how CQL successfully trains offline policies across varying data qualities[cite: 12]. 

### 1. Training on "Random" Data (The Power of CQL)
Even when trained on a dataset of an untrained, random policy (Return: 20), the Offline-CQL agent managed to extrapolate and achieve a significantly higher return[cite: 12].
*   **Online Source Policy:** 20 Return[cite: 12]
*   **Offline CQL Policy:** 122 Return[cite: 12]

*(Insert Random GIF from presentation here)*

### 2. Training on "Mid" Data
*   **Online Source Policy:** 227 Return[cite: 12]
*   **Offline CQL Policy:** 226 Return[cite: 12]

*(Insert Mid GIF from presentation here)*

### 3. Training on "Perfect" Data
*   **Online Source Policy:** 500 Return[cite: 12]
*   **Offline CQL Policy:** 500 Return[cite: 12]

*(Insert Perfect GIF from presentation here)*

## 📈 Training Metrics (TensorBoard)

The logs demonstrate how adjusting the $\alpha$ parameter influences the CQL Loss, DQN Loss, and the mean Q-values over time to prevent overestimation[cite: 12].

*(Insert your best TensorBoard screenshot here, e.g., the Q-values mean graph showing the stabilization[cite: 12])*