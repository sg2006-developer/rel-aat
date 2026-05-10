import argparse
import csv
import json
import os
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import yaml

from environment import IrrigationEnv
from agent import QLearningAgent


def moving_average(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode="valid")


# Pre-defined training configurations (kept for backward-compat with main.py)
POLICY_CONFIGS = {
    "policy_v1": {
        "label":         "V1 — Conservative (fast decay, fewer episodes)",
        "n_episodes":    800,
        "alpha":         0.1,
        "gamma":         0.95,
        "epsilon":       1.0,
        "epsilon_min":   0.05,
        "epsilon_decay": 0.995,
        "save_path":     "policy_v1.pkl",
        "plot_path":     "training_v1.png",
        "log_csv":       "results_v1.csv",
        "log_json":      "log.json",
        "run_id":        "v1",
    },
    "policy_v2_explored": {
        "label":         "V2 — Exploratory (slow decay, more episodes)",
        "n_episodes":    1500,
        "alpha":         0.08,
        "gamma":         0.97,
        "epsilon":       1.0,
        "epsilon_min":   0.02,
        "epsilon_decay": 0.998,
        "save_path":     "policy_v2_explored.pkl",
        "plot_path":     "training_v2_explored.png",
        "log_csv":       "results_v2.csv",
        "log_json":      "log.json",
        "run_id":        "v2",
    },
}


# ------------------------------------------------------------------ #
#  Experiment logging                                                  #
# ------------------------------------------------------------------ #

def _log_results_csv(path, row: dict):
    """Append one row to a CSV experiment log, creating headers if needed."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Experiment log updated: {path}")


def _log_results_json(path, entry: dict):
    """Append/update a JSON log file with a new experiment entry."""
    data = []
    if os.path.isfile(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON log updated: {path}")


# ------------------------------------------------------------------ #
#  Core training loop                                                  #
# ------------------------------------------------------------------ #

def train(n_episodes=1000, show_plot=False, save_path="q_table.pkl",
          plot_path="training_progress.png", label="Training",
          log_csv="results.csv", log_json="log.json", run_id=None,
          **agent_kwargs):

    env   = IrrigationEnv()
    alpha         = agent_kwargs.get("alpha", 0.1)
    gamma         = agent_kwargs.get("gamma", 0.95)
    epsilon       = agent_kwargs.get("epsilon", 1.0)
    epsilon_min   = agent_kwargs.get("epsilon_min", 0.05)
    epsilon_decay = agent_kwargs.get("epsilon_decay", 0.995)

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    episode_rewards = []
    episode_water   = []
    episode_health  = []

    print("=" * 55)
    print(f"   {label}")
    print(f"   Run ID : {run_id or 'unset'}")
    print("=" * 55)

    for ep in range(1, n_episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        while not done:
            action               = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state        = next_state
            total_reward += reward

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_water.append(env.total_water)
        episode_health.append(env.crop_health)

        if ep % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            avg_w = np.mean(episode_water[-100:])
            print(
                f"  Episode {ep:5d} | "
                f"Avg Reward: {avg_r:8.2f} | "
                f"Avg Water: {avg_w:6.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("=" * 55)
    print("  Training complete.")
    agent.save(save_path)
    agent.save("q_table.pkl")   # keep generic pointer to latest policy

    # ---- Compute summary stats ----
    avg_reward = round(float(np.mean(episode_rewards)), 4)
    avg_water  = round(float(np.mean(episode_water)), 4)
    avg_health = round(float(np.mean(episode_health)), 4)

    # ---- Write CSV log ----
    csv_row = {
        "run_id":        run_id or str(uuid.uuid4())[:8],
        "episodes":      n_episodes,
        "avg_reward":    avg_reward,
        "avg_water_used": avg_water,
        "avg_crop_health": avg_health,
        "epsilon":       round(epsilon, 4),
        "epsilon_min":   round(epsilon_min, 4),
        "epsilon_decay": round(epsilon_decay, 4),
        "alpha":         round(alpha, 4),
        "gamma":         round(gamma, 4),
        "policy_file":   save_path,
    }
    _log_results_csv(log_csv, csv_row)

    # ---- Write JSON log ----
    json_entry = {**csv_row, "label": label}
    _log_results_json(log_json, json_entry)

    # ---- Plot ----
    _plot_training(episode_rewards, episode_water, episode_health, n_episodes,
                   title=label, save_path=plot_path)
    if show_plot:
        plt.show()

    return agent, episode_rewards, episode_water, episode_health


# ------------------------------------------------------------------ #
#  Train all pre-defined policies                                      #
# ------------------------------------------------------------------ #

def train_all_policies(show_plot=False):
    """Train every config in POLICY_CONFIGS and save each to its own file."""
    results = {}
    for key, cfg in POLICY_CONFIGS.items():
        print(f"\n{'#' * 55}")
        print(f"  Starting: {cfg['label']}")
        print(f"{'#' * 55}")
        agent, rewards, water, health = train(
            n_episodes    = cfg["n_episodes"],
            show_plot     = show_plot,
            save_path     = cfg["save_path"],
            plot_path     = cfg["plot_path"],
            label         = cfg["label"],
            log_csv       = cfg.get("log_csv", "results.csv"),
            log_json      = cfg.get("log_json", "log.json"),
            run_id        = cfg.get("run_id"),
            alpha         = cfg["alpha"],
            gamma         = cfg["gamma"],
            epsilon       = cfg["epsilon"],
            epsilon_min   = cfg["epsilon_min"],
            epsilon_decay = cfg["epsilon_decay"],
        )
        results[key] = {"agent": agent, "rewards": rewards, "water": water, "health": health}
    print(f"\n  Saved policies: {', '.join(cfg['save_path'] for cfg in POLICY_CONFIGS.values())}")
    return results


# ------------------------------------------------------------------ #
#  YAML-based training (for reproducibility)                          #
# ------------------------------------------------------------------ #

def train_from_config(config_path: str, show_plot=False):
    """Load a YAML config and run training. Core MLOps reproducibility entry-point."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"\n  Loaded config: {config_path}")
    label = f"{cfg.get('policy', 'experiment')} — from {os.path.basename(config_path)}"

    agent, rewards, water, health = train(
        n_episodes    = cfg.get("n_episodes", 1000),
        show_plot     = show_plot,
        save_path     = cfg.get("save_path", "q_table.pkl"),
        plot_path     = cfg.get("plot_path", "training_progress.png"),
        label         = label,
        log_csv       = cfg.get("log_csv", "results.csv"),
        log_json      = cfg.get("log_json", "log.json"),
        run_id        = cfg.get("run_id"),
        alpha         = cfg.get("alpha", 0.1),
        gamma         = cfg.get("gamma", 0.95),
        epsilon       = cfg.get("epsilon", 1.0),
        epsilon_min   = cfg.get("epsilon_min", 0.05),
        epsilon_decay = cfg.get("epsilon_decay", 0.995),
    )
    return agent, rewards, water, health


# ------------------------------------------------------------------ #
#  Plot helper                                                         #
# ------------------------------------------------------------------ #

def _plot_training(rewards, water, health, n_episodes,
                   title="Training Progress", save_path="training_progress.png"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"Q-Learning Training Progress\n{title}", fontsize=13, fontweight="bold")

    window = 50
    eps    = list(range(1, n_episodes + 1))

    # --- Reward ---
    axes[0].plot(eps, rewards, alpha=0.3, color="steelblue", linewidth=0.8, label="Per episode")
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        axes[0].plot(range(window, n_episodes + 1), ma, color="steelblue", linewidth=2, label=f"{window}-ep avg")
    axes[0].set_title("Total Reward per Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Water used ---
    axes[1].plot(eps, water, alpha=0.3, color="darkorange", linewidth=0.8)
    if len(water) >= window:
        axes[1].plot(range(window, n_episodes + 1), moving_average(water, window),
                     color="darkorange", linewidth=2)
    axes[1].set_title("Water Used per Episode (lower = more efficient)")
    axes[1].set_ylabel("Water Units")
    axes[1].grid(True, alpha=0.3)

    # --- Crop health ---
    axes[2].plot(eps, health, alpha=0.3, color="seagreen", linewidth=0.8)
    if len(health) >= window:
        axes[2].plot(range(window, n_episodes + 1), moving_average(health, window),
                     color="seagreen", linewidth=2)
    axes[2].set_title("Crop Health at End of Episode")
    axes[2].set_ylabel("Health (0–100)")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"  Plot saved: {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  CLI entry-point                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning agent for Smart Irrigation Controller")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file, e.g. qlearning_v1.yaml"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all pre-defined policy configurations"
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display training plots interactively (default: save only)"
    )
    args = parser.parse_args()

    if args.config:
        train_from_config(args.config, show_plot=args.show_plot)
    elif args.all:
        train_all_policies(show_plot=args.show_plot)
    else:
        # Default: train all policies
        train_all_policies(show_plot=args.show_plot)
