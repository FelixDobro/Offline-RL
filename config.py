from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = PROJECT_ROOT / "logs/offline/mid"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints/offline/mid"
DATA_DIR = PROJECT_ROOT / "data/mid"


Path.mkdir(CHECKPOINTS_DIR, exist_ok=True)


## 25 = perfect, 0= random, 4=mid, 1 = bad

MODEL_VERSION = 4
MODEL_DIR = CHECKPOINTS_DIR / f"model{MODEL_VERSION}.pt"


LOG_INTERVAL = 25
SAVE_EVERY = 500
TARGET_UPDATE_INTERVAL = 400

LEARNING_RATE = 1e-4
NUM_ENVS = 12
OBS_DIM = 4

EPSILON = 0.05
GAMMA = 0.99
SAMPLE_GEN = 12
NUM_UPDATES = 5
BATCH_SIZE = 256
BUFFER_SIZE = 100000


# Data generation

NUM_SAMPLES = 100000
NUM_WORKERS = 4
NUM_EPOCHS = 1000
ALPHA = 0.5
TAU = 0.005
CHECK_MODEL = 500
NUMBER_OF_EVAL_STEPS = 500