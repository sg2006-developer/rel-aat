import numpy as np
import matplotlib.pyplot as plt
from environment import IrrigationEnv
from agent import QLearningAgent


def moving_average(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode="valid")


# Pre-defined training configurations
POLICY_CONFIGS = {
    "policy_v1": {
        "label":         "V1 — Conservative (fast decay, fewer episodes)",
        "n_episodes":    800,
        "alpha":         0.1,
        "gamma":         0.95,
        "epsilon":       1.0,
        "epsilon_min":   0.05,
        "epsilon_decay": 0.995,   # epsilon reaches ~0.05 around ep 600
        "save_path":     "policy_v1.pkl",
        "plot_path":     "training_v1.png",
    },
    "policy_v2_explored": {
        "label":         "V2 — Exploratory (slow decay, more episodes)",
        "n_episodes":    1500,
        "alpha":         0.08,    # smaller lr -> more stable convergence
        "gamma":         0.97,    # values future rewards more
        "epsilon":       1.0,
        "epsilon_min":   0.02,    # explore even longer before converging
        "epsilon_decay": 0.998,   # slower decay -> richer exploration
        "save_path":     "policy_v2_explored.pkl",
        "plot_path":     "training_v2_explored.png",
    },
}


def train(n_episodes=1000, show_plot=True, save_path="q_table.pkl",
          plot_path="training_progress.png", label="Training", **agent_kwargs):
    env   = IrrigationEnv()
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=agent_kwargs.get("alpha", 0.1),
        gamma=agent_kwargs.get("gamma", 0.95),
        epsilon=agent_kwargs.get("epsilon", 1.0),
        epsilon_min=agent_kwargs.get("epsilon_min", 0.05),
        epsilon_decay=agent_kwargs.get("epsilon_decay", 0.995),
    )

    episode_rewards   = []
    episode_water     = []
    episode_health    = []

    print("=" * 55)
    print(f"   {label}")
    print("=" * 55)

    for ep in range(1, n_episodes + 1):
        state      = env.reset()
        total_reward = 0.0
        done       = False

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
    # Also keep the generic q_table.pkl pointing to the latest trained policy
    agent.save("q_table.pkl")

    if show_plot:
        _plot_training(episode_rewards, episode_water, episode_health, n_episodes,
                       title=label, save_path=plot_path)

    return agent, episode_rewards, episode_water, episode_health


def train_all_policies(show_plot=True):
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
            alpha         = cfg["alpha"],
            gamma         = cfg["gamma"],
            epsilon       = cfg["epsilon"],
            epsilon_min   = cfg["epsilon_min"],
            epsilon_decay = cfg["epsilon_decay"],
        )
        results[key] = {"agent": agent, "rewards": rewards, "water": water, "health": health}
    print(f"\n  Saved policies: {', '.join(cfg['save_path'] for cfg in POLICY_CONFIGS.values())}")
    return results


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
    plt.show()


if __name__ == "__main__":
    train_all_policies(show_plot=True)
