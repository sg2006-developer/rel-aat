# Smart Irrigation Controller — RL Q-Learning

> **SDG 2 — Zero Hunger** | Efficient crop watering via intelligent decision-making using Reinforcement Learning

---

## Project Overview

This project implements a **Tabular Q-Learning agent** that learns an optimal irrigation policy for a simulated crop field. The agent observes soil moisture, weather, and time of day, then decides whether to water (and how much), aiming to maximise crop health while minimising water waste.

### Environment
- **State**: `(moisture_level, weather, time_of_day)` → 45 discrete states
- **Actions**: `No Water`, `Water Low`, `Water High`
- **Reward**: +10 for optimal moisture, −10 for drought, penalties for over-watering

### SDG Connection
Approximately 70% of global freshwater is used in agriculture. This RL agent achieves efficient irrigation by learning from simulated experience, supporting **SDG 2 (Zero Hunger)** and **SDG 6 (Clean Water and Sanitation)** by reducing water waste and improving crop resilience.

---

## Repository Structure

```
rel-aat/
├── environment.py          # Irrigation simulation (state, action, reward)
├── agent.py                # Tabular Q-Learning agent (ε-greedy)
├── train.py                # Training loop + YAML config support + experiment logging
├── evaluate.py             # RL vs fixed-baseline evaluation
├── main.py                 # CLI entry point
├── qlearning_v1.yaml       # Config for V1 (conservative)
├── qlearning_v2.yaml       # Config for V2 (exploratory)
├── results_v1.csv          # Experiment log for V1 run
├── results_v2.csv          # Experiment log for V2 run
├── log.json                # Aggregated JSON log of all runs
├── policy_v1.pkl           # Trained Q-table: V1 conservative
├── policy_v2_explored.pkl  # Trained Q-table: V2 exploratory
└── REPORT.md               # Final evaluation report
```

---

## Setup

```bash
git clone https://github.com/Sandy-383/rel-aat.git
cd rel-aat
pip install numpy matplotlib pyyaml
```

---

## Reproducibility

Anyone can reproduce any experiment from scratch by running:

```bash
# Reproduce V1 — Conservative policy (800 episodes)
python train.py --config qlearning_v1.yaml

# Reproduce V2 — Exploratory policy (1500 episodes)
python train.py --config qlearning_v2.yaml

# Train both policies at once
python train.py --all
```

Each run produces:
- A `.pkl` policy file (the trained Q-table)
- A `results_<run_id>.csv` with all hyperparameters and summary metrics
- An updated `log.json` aggregating all runs
- A `training_<run_id>.png` plot showing reward, water, and health over episodes

### Config File Format (`qlearning_v1.yaml`)

```yaml
run_id: "v1"
policy: "policy_v1"
n_episodes: 800
alpha: 0.1
gamma: 0.95
epsilon: 1.0
epsilon_min: 0.05
epsilon_decay: 0.995
save_path: "policy_v1.pkl"
plot_path: "training_v1.png"
log_csv: "results_v1.csv"
log_json: "log.json"
```

---

## Evaluation

```bash
# Evaluate latest policy vs fixed-timer baseline
python main.py --evaluate

# Compare V1 vs V2 side-by-side
python main.py --compare

# Run day-by-day demo (30 steps)
python main.py --demo
```

---

## Experiment Tracking

Every training run automatically logs to CSV and JSON:

| Field | Description |
|-------|-------------|
| `run_id` | Experiment identifier (e.g., `v1`, `v2`) |
| `episodes` | Number of training episodes |
| `avg_reward` | Mean total reward across all episodes |
| `avg_water_used` | Mean water consumed per episode |
| `avg_crop_health` | Mean crop health at end of episode |
| `epsilon` | Initial exploration rate |
| `alpha` | Learning rate |
| `gamma` | Discount factor |
| `epsilon_decay` | Decay rate per episode |
| `policy_file` | Path to the saved `.pkl` policy |

---

## Git Experiment Tags

| Tag | Description |
|-----|-------------|
| `exp-qlearning-1` | Initial Q-Learning implementation (V1 conservative policy) |
| `exp-qlearning-2` | MLOps layer added: YAML configs, CSV/JSON logging, README |

To checkout a specific experiment:
```bash
git checkout exp-qlearning-1   # original experiment
git checkout exp-qlearning-2   # with MLOps infrastructure
```

---

## Monitoring Plan

If this system were deployed in a real-world agricultural setting, we would monitor the following metrics in production:

> **"We would track average crop health per season (alert if below 70/100), total water consumed relative to the seasonal historical baseline (alert if >20% above baseline), drought event frequency (consecutive days with moisture = Very Dry), overwatering events (moisture = Saturated), and daily action distribution to detect policy drift. We would also re-trigger training if the reward moving average drops more than 15% below the training-time average over a 30-day window, which may indicate a shift in weather patterns or crop conditions not seen during training."**

---

## Results Summary

See [`REPORT.md`](REPORT.md) for the full baseline vs RL comparison table, SDG impact analysis, and discussion of limitations.
