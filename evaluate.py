import numpy as np
import matplotlib.pyplot as plt
import mlflow
from environment import IrrigationEnv, ACTIONS, MOISTURE_LEVELS, WEATHER_CONDITIONS, TIME_SLOTS
from agent import QLearningAgent


# ------------------------------------------------------------------ #
#  Baseline agent — fixed schedule, no sensing                        #
# ------------------------------------------------------------------ #

class FixedScheduleAgent:
    """
    Waters every morning at low level regardless of conditions.
    Represents a traditional fixed irrigation schedule.
    """

    def __init__(self, env):
        self.env = env

    def choose_action(self, state):
        # Always water_low in the morning (time_idx == 0), else no water
        time_idx = state % len(TIME_SLOTS)
        return 1 if time_idx == 0 else 0   # 1=water_low, 0=no_water


# ------------------------------------------------------------------ #
#  Evaluation runner                                                   #
# ------------------------------------------------------------------ #

def run_episode(env, policy_fn, verbose=False):
    state = env.reset()
    done  = False
    total_reward  = 0.0
    water_log     = []
    moisture_log  = []
    action_log    = []
    reward_log    = []

    while not done:
        action = policy_fn(state)
        next_state, reward, done = env.step(action)

        water_log.append(env.total_water)
        moisture_log.append(env.moisture_idx)
        action_log.append(action)
        reward_log.append(reward)

        total_reward += reward
        state = next_state

        if verbose:
            print(
                f"  Day {env.step_count:3d} | {env.get_state_label():<55} "
                f"| Action: {ACTIONS[action]:<12} | Reward: {reward:6.2f}"
            )

    return {
        "total_reward": total_reward,
        "total_water":  env.total_water,
        "crop_health":  env.crop_health,
        "water_log":    water_log,
        "moisture_log": moisture_log,
        "action_log":   action_log,
        "reward_log":   reward_log,
    }


def evaluate(n_eval_episodes=50, q_table_path="q_table.pkl", show_plot=True, label=None):
    env = IrrigationEnv()

    # Load trained agent
    rl_agent = QLearningAgent(env.n_states, env.n_actions)
    rl_agent.load(q_table_path)

    baseline = FixedScheduleAgent(env)

    rl_rewards, rl_water, rl_health       = [], [], []
    base_rewards, base_water, base_health = [], [], []

    policy_label = label or q_table_path
    print("\n" + "=" * 55)
    print(f"   Evaluation: {policy_label}")
    print("   RL Agent  vs  Fixed Schedule Baseline")
    print("=" * 55)

    for ep in range(n_eval_episodes):
        # RL agent
        r = run_episode(env, rl_agent.choose_action_greedy)
        rl_rewards.append(r["total_reward"])
        rl_water.append(r["total_water"])
        rl_health.append(r["crop_health"])

        # Baseline
        r = run_episode(env, baseline.choose_action)
        base_rewards.append(r["total_reward"])
        base_water.append(r["total_water"])
        base_health.append(r["crop_health"])

    # --- Summary table ---
    print(f"\n  {'Metric':<28} {'RL Agent':>12} {'Baseline':>12}")
    print("  " + "-" * 54)
    print(f"  {'Avg Total Reward':<28} {np.mean(rl_rewards):>12.2f} {np.mean(base_rewards):>12.2f}")
    print(f"  {'Avg Water Used':<28} {np.mean(rl_water):>12.1f} {np.mean(base_water):>12.1f}")
    print(f"  {'Avg Crop Health':<28} {np.mean(rl_health):>12.2f} {np.mean(base_health):>12.2f}")
    water_saved = (1 - np.mean(rl_water) / max(np.mean(base_water), 1)) * 100
    print(f"\n  Water saved by RL agent : {water_saved:.1f}%")
    print("=" * 55)

    # --- Detailed single episode trace ---
    print("\n  --- RL Agent: sample 10-step trace ---")
    run_episode(env, rl_agent.choose_action_greedy, verbose=False)   # reset env
    state = env.reset()
    done  = False
    step  = 0
    while not done and step < 10:
        action = rl_agent.choose_action_greedy(state)
        next_state, reward, done = env.step(action)
        print(
            f"  Step {step+1:2d} | {env.get_state_label():<55} "
            f"| Action: {ACTIONS[action]:<12} | Reward: {reward:6.2f}"
        )
        state = next_state
        step += 1

    plot_path = "evaluation_comparison.png"
    if show_plot:
        _plot_comparison(rl_rewards, rl_water, rl_health,
                         base_rewards, base_water, base_health,
                         n_eval_episodes)

    # ------------------------------------------------------------------ #
    #  MLflow — log evaluation results                                    #
    # ------------------------------------------------------------------ #
    mlflow.set_experiment("SmartIrrigation-QLearning")
    with mlflow.start_run(run_name=f"eval-{policy_label}"):
        mlflow.log_params({
            "n_eval_episodes": n_eval_episodes,
            "q_table_path":    q_table_path,
            "policy_label":    policy_label,
        })
        mlflow.log_metrics({
            "eval_rl_avg_reward":       round(float(np.mean(rl_rewards)), 4),
            "eval_rl_avg_water":        round(float(np.mean(rl_water)),   4),
            "eval_rl_avg_health":       round(float(np.mean(rl_health)),  4),
            "eval_base_avg_reward":     round(float(np.mean(base_rewards)), 4),
            "eval_base_avg_water":      round(float(np.mean(base_water)),   4),
            "eval_base_avg_health":     round(float(np.mean(base_health)),  4),
            "eval_water_saved_pct":     round(water_saved, 2),
        })
        if show_plot and __import__("os").path.isfile(plot_path):
            mlflow.log_artifact(plot_path)
        print(f"  [MLflow] Evaluation metrics logged.")

    return {
        "rl":       {"rewards": rl_rewards, "water": rl_water, "health": rl_health},
        "baseline": {"rewards": base_rewards, "water": base_water, "health": base_health},
    }


def _plot_comparison(rl_r, rl_w, rl_h, b_r, b_w, b_h, n):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("RL Agent vs Fixed Schedule Baseline\n(Smart Irrigation Controller)",
                 fontsize=13, fontweight="bold")

    episodes = list(range(1, n + 1))
    colors   = {"rl": "steelblue", "base": "tomato"}

    def bar_pair(ax, rl_vals, base_vals, title, ylabel):
        avg_rl   = np.mean(rl_vals)
        avg_base = np.mean(base_vals)
        bars = ax.bar(["RL Agent", "Fixed Schedule"], [avg_rl, avg_base],
                      color=[colors["rl"], colors["base"]], width=0.45, edgecolor="black", linewidth=0.7)
        for bar, val in zip(bars, [avg_rl, avg_base]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    bar_pair(axes[0], rl_r, b_r, "Avg Total Reward",     "Reward")
    bar_pair(axes[1], rl_w, b_w, "Avg Water Consumed\n(lower is better)", "Water Units")
    bar_pair(axes[2], rl_h, b_h, "Avg Crop Health",      "Health (0–100)")

    plt.tight_layout()
    plt.savefig("evaluation_comparison.png", dpi=120)
    print("\n  Plot saved: evaluation_comparison.png")
    plt.show()


if __name__ == "__main__":
    evaluate(n_eval_episodes=50)
