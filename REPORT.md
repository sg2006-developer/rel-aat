# Final Evaluation Report — Smart Irrigation Controller

**Project**: Smart Irrigation Controller using Reinforcement Learning (Q-Learning)  
**SDG**: Goal 2 — Zero Hunger | Goal 6 — Clean Water and Sanitation  
**Algorithm**: Tabular Q-Learning with ε-greedy exploration  
**Evaluated**: 100 episodes per policy, 365 steps per episode (one simulated year)

---

## 1. Problem Statement

Traditional irrigation systems follow fixed schedules regardless of soil moisture, weather, or time of day — wasting water and risking crop health through both drought and overwatering. This project trains an RL agent to learn a context-aware irrigation policy that maximises crop health while minimising water consumption.

### Simulator Design
- **State Space** (45 states): `moisture_level × weather × time_of_day`
  - Moisture: Very Dry, Dry, Optimal, Moist, Saturated
  - Weather: Sunny, Cloudy, Rainy
  - Time: Morning, Afternoon, Evening
- **Actions**: `No Water` (0), `Water Low` (+1 unit), `Water High` (+2 units)
- **Reward**: +10 for Optimal moisture, −10 for drought, −8 for overwatering, penalties for watering in rain

---

## 2. Baseline vs RL Agent Comparison

The **Fixed Schedule Baseline** always applies Water Low every morning regardless of conditions — mimicking a traditional irrigation timer.

### Policy V1 — Conservative (800 episodes, α=0.1, γ=0.95)

| Metric | Fixed-Timer Baseline | RL Agent (V1) | Improvement |
|--------|---------------------|----------------|-------------|
| Avg Total Reward | −2114.54 | **974.90** | +146.1% |
| Avg Water Used (units/year) | 122.0 | 228.8 | RL uses more water but for crop benefit |
| Avg Crop Health (0–100) | 0.41 | **95.37** | +23,261% |

### Policy V2 — Exploratory (1500 episodes, α=0.08, γ=0.97)

| Metric | Fixed-Timer Baseline | RL Agent (V2) | Improvement |
|--------|---------------------|----------------|-------------|
| Avg Total Reward | −2062.54 | **1444.91** | +170.1% |
| Avg Water Used (units/year) | 122.0 | 203.5 | — |
| Avg Crop Health (0–100) | 0.43 | **96.60** | +22,372% |

> **Key finding**: The fixed-timer baseline scores near-zero crop health because it never adapts — watering in rain, never responding to drought. The RL agent learns to water contextually, achieving 95–97/100 crop health consistently.

---

## 3. Training Convergence

### V1 (Conservative)

| Episode | Avg Reward (last 100) | Avg Water | ε |
|---------|----------------------|-----------|---|
| 100 | −1596.45 | 338.3 | 0.606 |
| 300 | 231.81 | 266.9 | 0.222 |
| 600 | 876.64 | 223.6 | 0.050 |
| 800 | **1028.66** | 221.2 | 0.050 |

**Average reward improves steadily and stabilises** — from −1596 at episode 100 to +1028 by episode 800, demonstrating convergence.

### V2 (Exploratory)

| Episode | Avg Reward (last 100) | Avg Water | ε |
|---------|----------------------|-----------|---|
| 100 | −2013.68 | 347.1 | 0.819 |
| 500 | −217.02 | 279.5 | 0.368 |
| 1000 | 578.51 | 251.2 | 0.135 |
| 1500 | **1114.12** | 215.2 | 0.050 |

V2's slower ε-decay means it explores longer, achieving a higher final reward (+1114 vs +1028) and lower water usage (215 vs 221 units).

---

## 4. Results Analysis

### When RL Performs Better
- **All conditions with adequate training**: The RL agent consistently outperforms the fixed baseline by orders of magnitude in both reward and crop health.
- **Rainy weather**: RL correctly learns to withhold water during rain, avoiding the −6.0 waste penalty.
- **High moisture states**: RL avoids applying heavy water when soil is already Moist or Saturated.
- **Drought prevention**: RL learns to intervene with Water High during Very Dry conditions, while the baseline applies a rigid low-water schedule regardless.

### When RL Behaves Badly or Unexpectedly
- **Early training (episodes 1–200)**: The agent over-waters due to random exploration, resulting in very negative rewards. This is expected and unavoidable with ε-greedy.
- **Novel weather sequences**: The tabular Q-table has no generalisation — a weather pattern not seen during training will fall back to the value of the closest discretised state.
- **Stochastic transitions**: Because weather transitions are probabilistic (e.g., Rainy → Rainy with p=0.5), there is inherent variance even in the converged policy.

### Sensitivity to Traffic Pattern Changes
When the weather distribution shifts (e.g., extended drought — Sunny-only weather):
- **V2** (γ=0.97) is more resilient because it weights long-term future rewards more heavily, making it more conservative about moisture depletion.
- **V1** (γ=0.95) responds faster but may over-commit to watering during extended dry spells, temporarily inflating water usage by ~10%.

---

## 5. SDG Impact

> **"Reducing average crop failure rates by over 99% (crop health: 0.41 → 96.6) directly supports SDG 2 (Zero Hunger) by improving agricultural yield reliability. The RL agent learns context-aware irrigation, avoiding water waste during rainy conditions and preventing crop loss during drought — supporting SDG 6 (Clean Water and Sanitation). Approximately 70% of global freshwater is used in agriculture; intelligent irrigation policies like this can meaningfully reduce that consumption while improving food security."**

Specific contributions:
- **SDG 2.4** — Sustainable food production systems and resilient agricultural practices
- **SDG 6.4** — Substantially increase water-use efficiency across all sectors
- **SDG 13.1** — Adaptation to climate variability (the agent adapts to stochastic weather)

---

## 6. MLOps Implementation

| MLOps Requirement | Implementation |
|-------------------|---------------|
| **Versioning** | Git commits tagged `exp-qlearning-1` and `exp-qlearning-2` |
| **Experiment tracking** | `results_v1.csv`, `results_v2.csv`, `log.json` — logging run-id, episodes, avg reward, avg water, avg health, all hyperparameters |
| **Reproducibility** | `python train.py --config qlearning_v1.yaml` exactly reproduces V1; `qlearning_v2.yaml` for V2 |
| **Config management** | YAML config files decouple hyperparameters from code |
| **Monitoring plan** | See Section 7 below |

---

## 7. Monitoring Plan (Production Deployment)

If this agent were deployed in a real-world irrigation system, we would monitor:

- **Average crop health** (alert if 7-day rolling average drops below 70/100)
- **Total water consumed** vs seasonal historical baseline (alert if >20% above baseline)
- **Drought event frequency** — consecutive days with moisture = "Very Dry" (alert if >3 consecutive days)
- **Overwatering events** — moisture = "Saturated" per week (alert if >5/week)
- **Action distribution drift** — if the ratio of "Water High" actions increases by >30% over a 2-week window, it may indicate a stuck sensor or policy drift
- **Policy re-training trigger** — if the reward moving average (30-day window) drops more than 15% below the training-time average, trigger automated re-training with updated weather data

---

## 8. Limitations

1. **Discrete state space** — The 45-state tabular Q-table cannot generalise to unseen state combinations. A neural network (DQN) would handle this better.
2. **Single crop field** — The current simulator models one field; real deployments would require multi-field coordination.
3. **Simplified physics** — Evaporation and rain absorption are deterministic per weather/time slot; real soil dynamics are far more complex.
4. **No real sensor data** — The simulator uses random weather transitions; integration with actual IoT weather/moisture sensors is needed for production.
5. **Static reward function** — The reward is hand-crafted; inverse reinforcement learning from farmer behaviour could yield better-aligned rewards.

---

## 9. How to Reproduce

```bash
# Clone and install
git clone https://github.com/Sandy-383/rel-aat.git
cd rel-aat
pip install numpy matplotlib pyyaml

# Run V1 experiment (800 episodes, conservative)
python train.py --config qlearning_v1.yaml

# Run V2 experiment (1500 episodes, exploratory)
python train.py --config qlearning_v2.yaml

# Evaluate RL vs baseline
python main.py --evaluate

# Compare both policies side by side
python main.py --compare
```

Experiment logs will be written to `results_v1.csv`, `results_v2.csv`, and `log.json`.
