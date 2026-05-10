"""
Smart Irrigation Controller — Reinforcement Learning (Q-Learning)
SDG 2: Zero Hunger | Efficient crop watering via intelligent decision-making

Usage:
  python main.py                   # train both policies then evaluate best
  python main.py --train           # train both policies (v1 + v2_explored)
  python main.py --evaluate        # evaluate latest policy vs baseline
  python main.py --compare         # evaluate & compare both saved policies
  python main.py --demo            # step-by-step demo of trained agent
  python main.py --demo --v2       # demo using the exploratory policy
"""

import sys
import os
import numpy as np
from environment import IrrigationEnv, ACTIONS, MOISTURE_LEVELS, WEATHER_CONDITIONS, TIME_SLOTS
from agent import QLearningAgent


BANNER = """
╔══════════════════════════════════════════════════════╗
║     Smart Irrigation Controller — RL Simulation      ║
║     Algorithm : Q-Learning (Tabular)                 ║
║     SDG Goal  : 2 — Zero Hunger                      ║
╚══════════════════════════════════════════════════════╝
"""


def demo(q_table_path="q_table.pkl", n_days=30):
    """Show the trained agent making decisions day by day."""
    if not os.path.exists(q_table_path):
        print("  No trained model found. Run with --train first.")
        return

    env   = IrrigationEnv()
    agent = QLearningAgent(env.n_states, env.n_actions)
    agent.load(q_table_path)

    state = env.reset()
    print("\n  Day-by-Day Agent Decisions (first 30 days of simulation)")
    print("  " + "-" * 70)
    print(f"  {'Day':<5} {'Moisture':<12} {'Weather':<9} {'Time':<11} {'Action':<14} {'Reward':>7}")
    print("  " + "-" * 70)

    for day in range(1, n_days + 1):
        action = agent.choose_action_greedy(state)
        next_state, reward, done = env.step(action)

        print(
            f"  {day:<5} "
            f"{MOISTURE_LEVELS[env.moisture_idx]:<12} "
            f"{WEATHER_CONDITIONS[env.weather_idx]:<9} "
            f"{TIME_SLOTS[env.time_idx]:<11} "
            f"{ACTIONS[action]:<14} "
            f"{reward:>7.2f}"
        )
        state = next_state
        if done:
            break

    print("  " + "-" * 70)
    print(f"  Total water used : {env.total_water}")
    print(f"  Final crop health: {env.crop_health:.1f}/100")


def compare_policies():
    """Evaluate both saved policies side by side and print a summary table."""
    import matplotlib.pyplot as plt
    from evaluate import evaluate, FixedScheduleAgent, run_episode
    from train import POLICY_CONFIGS

    policies = {
        "V1 Conservative":  "policy_v1.pkl",
        "V2 Exploratory":   "policy_v2_explored.pkl",
    }

    print("\n" + "=" * 65)
    print("   Policy Comparison: V1 Conservative vs V2 Exploratory")
    print("=" * 65)

    summary = {}
    for name, path in policies.items():
        if not os.path.exists(path):
            print(f"  Skipping {name} — {path} not found. Run --train first.")
            continue
        result = evaluate(n_eval_episodes=50, q_table_path=path,
                          show_plot=False, label=name)
        summary[name] = result["rl"]

    if len(summary) < 2:
        return

    print(f"\n  {'Metric':<28}", end="")
    for name in summary:
        print(f"  {name:>20}", end="")
    print()
    print("  " + "-" * (28 + 22 * len(summary)))

    for metric, key in [("Avg Total Reward", "rewards"), ("Avg Water Used", "water"), ("Avg Crop Health", "health")]:
        print(f"  {metric:<28}", end="")
        for name, data in summary.items():
            print(f"  {np.mean(data[key]):>20.2f}", end="")
        print()

    print("=" * 65)

    # Side-by-side bar chart
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Policy Comparison: V1 vs V2\nSmart Irrigation Controller",
                 fontsize=13, fontweight="bold")
    metrics = [("Avg Total Reward", "rewards"), ("Avg Water Used\n(lower = better)", "water"), ("Avg Crop Health", "health")]
    colors  = ["steelblue", "darkorange"]

    for ax, (title, key) in zip(axes, metrics):
        names  = list(summary.keys())
        values = [np.mean(summary[n][key]) for n in names]
        bars   = ax.bar(names, values, color=colors[:len(names)], edgecolor="black", linewidth=0.7, width=0.4)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("policy_comparison.png", dpi=120)
    print("  Plot saved: policy_comparison.png")
    plt.show()


def main():
    print(BANNER)

    args        = sys.argv[1:]
    do_train    = "--train"   in args or len(args) == 0
    do_evaluate = "--evaluate" in args or len(args) == 0
    do_compare  = "--compare"  in args
    do_demo     = "--demo"     in args
    use_v2      = "--v2"       in args

    if do_train:
        from train import train_all_policies
        train_all_policies(show_plot=True)

    if do_evaluate:
        from evaluate import evaluate
        evaluate(n_eval_episodes=50, show_plot=True)

    if do_compare:
        compare_policies()

    if do_demo:
        path = "policy_v2_explored.pkl" if use_v2 else "policy_v1.pkl"
        demo(q_table_path=path, n_days=30)

    if not (do_train or do_evaluate or do_compare or do_demo):
        print("  Unknown argument. Use --train, --evaluate, --compare, or --demo.")


if __name__ == "__main__":
    main()
