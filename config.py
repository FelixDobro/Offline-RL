from pathlib import Path
import torch

# ==========================================
# 1. System & Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Hardware acceleration (uses CUDA if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
# Directory structure
LOG_DIR = PROJECT_ROOT / "logs/offline/mid_500k"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/offline/mid"
DATA_DIR = PROJECT_ROOT / "data/mid_500k"

# Create directories automatically if they don't exist
Path.mkdir(CHECKPOINTS_DIR, exist_ok=True, parents=True)



# ==========================================
# 2. Environment
# ==========================================
NUM_ENVS = 12       # Number of parallel environments (AsyncVectorEnv)
OBS_DIM = 4         # Input vector size (CartPos, CartVel, PoleAngle, PoleVel)
REWARD_SCALE = 1  # Scales rewards down (crucial for gradient stability!)


# ==========================================
# 3. Model & Optimization (Hyperparameters)
# ==========================================
MODEL_VERSION = 11   # ID for saving/loading 20=perfect(500) 11=mid(240), 0=random(10)
LEARNING_RATE = 1e-4
GAMMA = 0.98        # Discount Factor: Importance of future rewards (0=short-sighted, 1=far-sighted)
TAU = 0.005         # Soft Update: How fast the target net follows the main net
EPSILON = 0.05      # Exploration: Probability of random actions (static here)


# ==========================================
# 4. Training Loop & Memory
# ==========================================
BUFFER_SIZE = 100_000   # Replay Buffer size (memory capacity)
BATCH_SIZE = 256        # Number of samples taken from buffer per update

# Balance between data collection and learning (Critical for stability!)
SAMPLE_GEN = 32         # Steps collected per loop
NUM_UPDATES = 4       # Gradient steps (training updates) per loop


# ==========================================
# 5. Logging & Saving
# ==========================================
LOG_INTERVAL = 25       # How often to write to Tensorboard (in iterations)
SAVE_EVERY = 500
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


# ==========================================
# 6. Offline Data / Legacy / Eval
# ==========================================
# Parameters for separate data generation or evaluation
NUM_SAMPLES = 500_000
NUM_WORKERS = 4
NUM_EPOCHS = 1000
ALPHA = 0.01

NUMBER_OF_EVAL_STEPS = 500