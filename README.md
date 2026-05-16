# Smart Irrigation Controller — RL Q-Learning

> **SDG 2 — Zero Hunger | SDG 6 — Clean Water and Sanitation**  
> Efficient, context-aware crop irrigation via Reinforcement Learning (Tabular Q-Learning)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Algorithm](https://img.shields.io/badge/Algorithm-Q--Learning-green)](https://en.wikipedia.org/wiki/Q-learning)
[![MLOps](https://img.shields.io/badge/MLOps-YAML%20%7C%20CSV%20%7C%20JSON%20%7C%20Git%20Tags-orange)]()
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![SDG](https://img.shields.io/badge/SDG-2%20%7C%206-teal)]()

---

## Project Overview

This project implements a **Tabular Q-Learning agent** that learns an optimal irrigation policy for a simulated crop field. The agent observes soil moisture, weather condition, and time of day, then decides whether to water (and how much), aiming to maximise crop health while minimising water waste.

### The Problem
Traditional irrigation systems follow fixed schedules regardless of environmental conditions — wasting water and risking crop health through both drought and overwatering. This RL agent learns a context-aware policy purely from simulated experience, achieving **95–97/100 crop health** vs **0.4/100 for a fixed-timer baseline**.

### Environment Design
| Component | Details |
|-----------|---------|
| **State Space** | `(moisture_level, weather, time_of_day)` → **45 discrete states** |
| **Moisture levels** | Very Dry, Dry, Optimal, Moist, Saturated |
| **Weather** | Sunny, Cloudy, Rainy (Markov transitions) |
| **Time slots** | Morning, Afternoon, Evening |
| **Actions** | `No Water` (0), `Water Low` (+1 unit), `Water High` (+2 units) |
| **Reward** | +10 Optimal, +6 Moist, −10 drought, −8 overwatering, −6 watering in rain |
| **Episode length** | 365 steps (one simulated year) |

### SDG Connection
Approximately 70% of global freshwater is used in agriculture. This RL agent achieves intelligent, context-aware irrigation by learning from simulated experience, directly supporting:
- **SDG 2 (Zero Hunger)** — improving crop reliability by >99% vs fixed baseline
- **SDG 6 (Clean Water and Sanitation)** — reducing unnecessary water use
- **SDG 13 (Climate Action)** — adapting to stochastic weather patterns

---

## Repository Structure

```
rel-aat/
├── environment.py              # Irrigation simulation (state, action, reward, transitions)
├── agent.py                    # Tabular Q-Learning agent (ε-greedy, save/load)
├── train.py                    # Training loop + YAML config support + CSV/JSON logging + MLflow
├── evaluate.py                 # RL vs fixed-baseline evaluation + comparison plots + MLflow
├── main.py                     # CLI entry point (--train, --evaluate, --compare, --demo)
├── app.py                      # Tkinter desktop GUI (live simulation + training visualiser)
│
├── qlearning_v1.yaml           # Config: V1 Conservative (800 eps, α=0.1, γ=0.95)
├── qlearning_v2.yaml           # Config: V2 Exploratory (1500 eps, α=0.08, γ=0.97)
│
├── policy_v1.pkl               # Trained Q-table: V1 conservative policy
├── policy_v2_explored.pkl      # Trained Q-table: V2 exploratory policy
├── q_table.pkl                 # Generic pointer → most recently trained policy
│
├── results_v1.csv              # Experiment log: V1 run (hyperparams + summary metrics)
├── results_v2.csv              # Experiment log: V2 run (hyperparams + summary metrics)
├── log.json                    # Aggregated JSON log of all training runs
│
├── training_v1.png             # Training progress plot: V1 (reward, water, health)
├── training_v2_explored.png    # Training progress plot: V2 (reward, water, health)
├── evaluation_comparison.png   # Bar chart: RL agent vs fixed-timer baseline
│
├── mlruns/                     # MLflow tracking store (auto-created on first run)
├── mlflow.db                   # MLflow SQLite backend (auto-created on first run)
│
├── REPORT.md                   # Full evaluation report (results, analysis, SDG impact)
├── .gitignore                  # Excludes: __pycache__, venv, verify.py, mlops_tasks.pdf
└── README.md                   # This file
```

---

## Setup

### Prerequisites
- Python 3.8+
- `numpy`, `matplotlib`, `pyyaml`, `mlflow`
- `tkinter` (bundled with standard Python — required only for the GUI)

### Install

```bash
# Clone the repository
git clone https://github.com/Sandy-383/rel-aat.git
cd rel-aat

# Install dependencies
pip install numpy matplotlib pyyaml mlflow
```

> **Fork note**: A contributor fork is also available at `https://github.com/sg2006-developer/rel-aat.git`

---

## Reproducibility

Every experiment is fully reproducible from a YAML config file. **No code changes are needed between runs.**

```bash
# Reproduce V1 — Conservative policy (800 episodes)
python train.py --config qlearning_v1.yaml

# Reproduce V2 — Exploratory policy (1500 episodes)
python train.py --config qlearning_v2.yaml

# Train both policies at once (uses embedded POLICY_CONFIGS dict)
python train.py --all
```

Each run automatically produces:
| Output File | Description |
|-------------|-------------|
| `policy_<run_id>.pkl` | Trained Q-table (pickled numpy array) |
| `results_<run_id>.csv` | Hyperparameters + summary metrics |
| `log.json` | Aggregated JSON log (all runs appended) |
| `training_<run_id>.png` | 3-panel plot: reward, water usage, crop health over episodes |

### YAML Config Format

```yaml
# qlearning_v1.yaml — V1 Conservative
run_id: "v1"
policy: "policy_v1"

# Training hyperparameters
n_episodes: 800
alpha: 0.1          # learning rate
gamma: 0.95         # discount factor
epsilon: 1.0        # initial exploration rate
epsilon_min: 0.05   # floor for epsilon (5% random actions after convergence)
epsilon_decay: 0.995  # epsilon reaches ~0.05 around episode 600

# Output paths
save_path: "policy_v1.pkl"
plot_path: "training_v1.png"
log_csv: "results_v1.csv"
log_json: "log.json"
```

---

## CLI Usage

```bash
# Train both policies then run evaluation
python main.py

# Train only
python main.py --train

# Evaluate latest policy (q_table.pkl) vs fixed baseline
python main.py --evaluate

# Compare V1 vs V2 side-by-side (table + bar chart)
python main.py --compare

# Step-by-step demo — 30 days of agent decisions (V1)
python main.py --demo

# Step-by-step demo using V2 (exploratory) policy
python main.py --demo --v2
```

---

## Desktop GUI

A full **Tkinter desktop GUI** (`app.py`) is included for interactive exploration:

```bash
python app.py
```

**GUI Features:**
- **Live Simulation tab** — watch the agent make real-time day-by-day decisions with animated reward, moisture, and water charts
- **Training Progress tab** — visualise reward, water usage, and ε-decay live as training runs
- **Policy Comparison tab** — side-by-side bar chart: Fixed Baseline vs V1 vs V2
- **Controls panel** — train both policies, simulate either policy, compare all three
- **State panel** — real-time display of day, moisture, weather, time, last action, step reward, total water, and crop health
- **Speed slider** — control simulation playback speed

> **Requires**: `tkinter` (standard Python) + `matplotlib` with `TkAgg` backend.

---

## Experiment Tracking

Every training run is automatically logged to **both** flat files (CSV / JSON) **and** MLflow.

### CSV / JSON Logs

| Field | Description |
|-------|-------------|
| `run_id` | Experiment identifier (`v1`, `v2`) |
| `episodes` | Total training episodes |
| `avg_reward` | Mean total reward across all training episodes |
| `avg_water_used` | Mean water units consumed per episode |
| `avg_crop_health` | Mean crop health score at end of episode |
| `epsilon` | Initial exploration rate |
| `epsilon_min` | Minimum epsilon after decay |
| `epsilon_decay` | Per-episode decay multiplier |
| `alpha` | Learning rate |
| `gamma` | Discount factor |
| `policy_file` | Path to the saved `.pkl` Q-table |

### Logged Training Metrics (from `log.json`)

| Run | Episodes | Avg Reward (train) | Avg Water | Avg Crop Health |
|-----|----------|--------------------|-----------|-----------------|
| v1  | 800      | 289.75             | 255.2     | 74.0            |
| v2  | 1500     | 57.93              | 267.3     | 65.6            |

> **Note**: Training averages include early random-exploration episodes (high water, low reward). Evaluation metrics (on the converged policy) are significantly better — see [REPORT.md](REPORT.md) for the full baseline vs RL comparison.

---

## MLflow Experiment Tracking

This project integrates **MLflow** for professional experiment tracking, model versioning, and the MLflow Model Registry. All data is stored locally — no cloud account required.

### What Gets Logged

**Per training run** (`train.py`):

| Category | Items |
|----------|-------|
| **Parameters** | `run_id`, `n_episodes`, `alpha`, `gamma`, `epsilon`, `epsilon_min`, `epsilon_decay`, `policy_file`, `label` |
| **Step Metrics** | `avg_reward_100`, `avg_water_100`, `avg_health_100`, `epsilon` — logged every 100 episodes |
| **Final Metrics** | `final_avg_reward`, `final_avg_water_used`, `final_avg_crop_health` |
| **Artifacts** | Training plot (`.png`), CSV log, JSON log |
| **Registered Model** | `QlearningAgent-v1` / `QlearningAgent-v2` (versioned in Model Registry) |

**Per evaluation run** (`evaluate.py`):

| Metric | Description |
|--------|-------------|
| `eval_rl_avg_reward` | Mean total reward of the RL agent |
| `eval_rl_avg_water` | Mean water used by the RL agent |
| `eval_rl_avg_health` | Mean crop health of the RL agent |
| `eval_base_avg_reward` | Mean total reward of the fixed-schedule baseline |
| `eval_base_avg_water` | Mean water used by the baseline |
| `eval_base_avg_health` | Mean crop health of the baseline |
| `eval_water_saved_pct` | Water saving % of RL agent vs baseline |

### Running MLflow

```bash
# Step 1 — Train (automatically creates and logs MLflow runs)
python train.py --all
# OR reproduce a single config:
python train.py --config qlearning_v1.yaml

# Step 2 — Evaluate (logs evaluation metrics to MLflow)
python evaluate.py

# Step 3 — Launch the MLflow UI
mlflow ui
# Then open: http://127.0.0.1:5000

# One-liner: train → evaluate → open UI
python main.py --train && python main.py --evaluate && mlflow ui
```

### Navigating the UI

1. Open `http://127.0.0.1:5000` in your browser
2. Select the **`SmartIrrigation-QLearning`** experiment in the left sidebar
3. View all runs with their hyperparameters and metrics
4. Select two runs → click **Compare** for side-by-side charts
5. Click **Models** in the sidebar to browse `QlearningAgent-v1` and `QlearningAgent-v2` in the Model Registry
6. Each registered model version links back to its source training run and stores the full Q-table

### Model Registry

The Q-table agent is wrapped as an `mlflow.pyfunc` model, enabling:
- **Versioning**: each `train.py` run creates a new version (`v1`, `v2`, ...)
- **Lineage**: every model version is linked to its exact source run, hyperparameters, and artifacts
- **Serving** (optional): `mlflow models serve -m "models:/QlearningAgent-v1/1"` to expose a REST API

```python
# Load and use a registered model in Python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/QlearningAgent-v1/1")
actions = model.predict([0, 12, 44])   # pass state indices → returns action indices
```

> **Storage**: All MLflow data lives in `./mlruns/` and `./mlflow.db` inside the project folder. Nothing is sent to any external service.

---

## Results Summary

Evaluated over 100 episodes × 365 steps (one simulated year per episode):

### Policy V1 — Conservative (800 episodes, α=0.1, γ=0.95)

| Metric | Fixed-Timer Baseline | RL Agent (V1) | Improvement |
|--------|---------------------|----------------|-------------|
| Avg Total Reward | −2,114.54 | **974.90** | +146.1% |
| Avg Water Used (units/year) | 122.0 | 228.8 | — |
| Avg Crop Health (0–100) | 0.41 | **95.37** | +23,261% |

### Policy V2 — Exploratory (1500 episodes, α=0.08, γ=0.97)

| Metric | Fixed-Timer Baseline | RL Agent (V2) | Improvement |
|--------|---------------------|----------------|-------------|
| Avg Total Reward | −2,062.54 | **1,444.91** | +170.1% |
| Avg Water Used (units/year) | 122.0 | 203.5 | — |
| Avg Crop Health (0–100) | 0.43 | **96.60** | +22,372% |

> **Key finding**: The fixed-timer baseline scores near-zero crop health because it never adapts — it waters every morning regardless of rain or soil saturation. The RL agent learns context-aware irrigation, achieving 95–97/100 crop health consistently.

See [`REPORT.md`](REPORT.md) for full convergence tables, sensitivity analysis, SDG impact discussion, and limitations.

---

## Git Versioning & Experiment Tags

| Tag | Commit | Description |
|-----|--------|-------------|
| `exp-qlearning-1` | `07cd169` | Initial Q-Learning implementation with two policy variants |
| `exp-qlearning-2` | `c596e39` | MLOps layer: YAML configs, CSV/JSON logging, README, REPORT |

```bash
# Checkout a specific experiment snapshot
git checkout exp-qlearning-1   # original RL implementation
git checkout exp-qlearning-2   # with full MLOps infrastructure

# Return to latest
git checkout main
```

### Git History

| Commit | Message |
|--------|---------|
| `bcd96b2` | Merge pull request #1 from sg2006-developer/main |
| `5cf303c` | chore: add .gitignore (exclude pycache, venv, scratch files) |
| `c596e39` | feat(mlops): add YAML configs, experiment logging (CSV+JSON), README, and REPORT |
| `07cd169` | initial Q-Learning implementation with two policy variants |
| `4a7421c` | initial commit |

---

## MLOps Implementation Summary

| MLOps Requirement | Implementation | Evidence |
|-------------------|----------------|----------|
| **Versioning** | Git commits + annotated tags `exp-qlearning-1` / `exp-qlearning-2` | `git tag` |
| **Experiment Tracking** | Auto-logging to `results_v*.csv`, `log.json` **and MLflow** on every run | Log files + `mlflow ui` |
| **MLflow Model Registry** | Each training run registers `QlearningAgent-v1` / `QlearningAgent-v2` as versioned pyfunc models | MLflow Models tab |
| **Reproducibility** | `python train.py --config qlearning_v1.yaml` exactly reproduces V1 | YAML configs |
| **Config Management** | Hyperparameters decoupled from code via YAML files | `qlearning_v*.yaml` |
| **Monitoring Plan** | Defined in REPORT.md §7 (thresholds, drift detection, re-train triggers) | `REPORT.md` |
| **Desktop GUI** | Tkinter app for live interactive simulation and policy comparison | `app.py` |

---

## Monitoring Plan (Production)

If this agent were deployed in a real-world agricultural setting, we would monitor:

> **"We would track average crop health per season (alert if 7-day rolling average drops below 70/100), total water consumed relative to seasonal historical baseline (alert if >20% above baseline), drought event frequency (consecutive days with moisture = Very Dry — alert if >3 consecutive days), overwatering events (moisture = Saturated — alert if >5/week), and daily action distribution to detect policy drift. We would re-trigger training if the reward moving average (30-day window) drops more than 15% below the training-time average, which may indicate a shift in weather patterns or crop conditions not seen during training."**

---

## File Quick-Reference

| File | Purpose | Run with |
|------|---------|----------|
| `environment.py` | Irrigation simulation environment | (imported) |
| `agent.py` | Q-Learning agent (ε-greedy, save/load) | (imported) |
| `train.py` | Train from YAML config or run all + MLflow logging | `python train.py --config <yaml>` |
| `evaluate.py` | Evaluate RL vs baseline + MLflow logging | `python evaluate.py` |
| `main.py` | Full CLI (train / evaluate / compare / demo) | `python main.py --help` |
| `app.py` | Interactive desktop GUI | `python app.py` |
| `verify.py` | Quick sanity check (CSV, JSON, git) | `python verify.py` |
| `mlruns/` | MLflow local tracking store (auto-created) | `mlflow ui` |
| `mlflow.db` | MLflow SQLite backend (auto-created) | `mlflow ui` |
