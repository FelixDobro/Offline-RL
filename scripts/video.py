import sys
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import numpy as np

# Pfade setzen wie im Original
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.utils import e_greedy_action
from config import *
from models.Q_net import QNet


def record_agent_run(model):
    # Ordner für die Videos erstellen
    video_dir = Path("videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Dateiname basierend auf Version und Seed
    file_prefix = f"cartpole_v{MODEL_VERSION}_seed_{SEED}"

    # Environment erstellen
    # WICHTIG: render_mode="rgb_array" ist zwingend für Videoaufnahmen
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Wrapper für die Videoaufnahme
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=file_prefix,
        episode_trigger=lambda x: True,  # Jede Episode aufnehmen (wir machen eh nur eine)
        disable_logger=False,
        fps=FPS
    )

    print(f"Starte Aufnahme für Modell {MODEL_VERSION} mit Seed {SEED}...")

    # Hier wird der FIXE SEED gesetzt
    current_obs, _ = env.reset(seed=SEED)

    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        # Dimension anpassen: Das Modell erwartet (Batch_Size, Obs),
        # wir haben aber nur (Obs). Daher expand_dims auf (1, Obs).
        obs_input = np.expand_dims(current_obs, axis=0)

        # Aktion wählen (eps=-1 für rein gieriges Verhalten/Inference)
        actions = e_greedy_action(obs_input, model, eps=EPSILON)

        # Action aus dem Batch extrahieren (da Batch Size 1)
        action = actions[0]

        next_obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        current_obs = next_obs

    env.close()

    print(f"Episode beendet. Score: {total_reward}")
    print(f"Video gespeichert unter: {video_dir}/{file_prefix}-episode-0.mp4")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Modell initialisieren
    model = QNet().to(DEVICE)

    # Checkpoint laden
    if MODEL_VERSION:
        # Pfadlogik aus deinem Skript übernommen
        checkpoint_path = f"{MODEL_DIR}"
        print(f"Lade Modell von: {checkpoint_path}")

        if torch.cuda.is_available():
            model_props = torch.load(checkpoint_path)
        else:
            model_props = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        state_dict = model_props["model_state_dict"]
        model.load_state_dict(state_dict)
    else:
        print("Warnung: Keine MODEL_VERSION definiert, nutze untrainiertes Modell.")

    model.eval()

    # Aufnahme starten
    record_agent_run(model)