# 🛡️ Women's Safety Path Recommendation using Q-Learning & MLOps

🌐 **Live Demo:** https://women-safety-api-ksyv.onrender.com/ui

> An AI-powered route recommendation system that learns safer paths for women 
> navigating urban environments, using Reinforcement Learning optimised for 
> safety over shortest distance.

---

## 📌 Problem Statement

Standard navigation apps (Google Maps, Apple Maps) optimise for time and 
distance — not safety. Women navigating cities, especially at night, face 
risks that shortest-path algorithms completely ignore.

This project uses Q-Learning to recommend routes that minimise exposure to 
high-risk zones, adapting recommendations based on time of day (day/night). 
The agent learns to trade marginal distance for significantly safer paths.

**SDG Link — SDG 11: Sustainable Cities and Communities**
Safe urban mobility is a core pillar of SDG 11. This project directly 
addresses the gap in safety-aware navigation for vulnerable populations.

---

## 🏆 Key Results

| Metric | RL Agent | BFS Baseline | Improvement |
|--------|----------|--------------|-------------|
| Day Risk Score | 12 | 20 | **40% safer** |
| Night Risk Score | 33 | 68 | **51.5% safer** |
| Path Length (Day) | 13 steps | 13 steps | Same efficiency |
| Path Length (Night) | 13 steps | 13 steps | Same efficiency |
| Q-table Coverage | 98% | — | 384/392 state-action pairs |

> The RL agent finds routes **40-51% safer** than the shortest path, 
> at identical path length. Safety without sacrificing efficiency.

---

## 🧠 RL Methodology

**Algorithm:** Q-Learning  
**Justification:** The state space (7×7 grid × 2 time modes = 98 states) 
is small and discrete, making tabular Q-learning fast, memory-efficient, 
and fully interpretable without neural network approximation.

### State, Action, Reward
- **State:** `(x, y, time_of_day)` — grid position + day/night context
- **Actions:** 4 directions — up, down, left, right
- **Reward:** `-(risk × 5) - 5` per step, `-10` loop penalty, 
  `-20` boundary penalty, `+50` on reaching goal

### Exploration Strategy
- ε-greedy with multiplicative decay: `1.0 → 0.05` over ~600 episodes
- Tie-breaking among equal Q-values prevents action bias
- Visited-cell penalty prevents the agent from looping

### Training Convergence
The agent's reward rises from approximately −6000 to −60 over 1000 episodes,
plateauing after episode ~400, indicating Q-table convergence. Epsilon decays 
to its minimum of 0.05 by episode ~600, shifting the agent from exploration 
to exploitation. Final Q-table coverage: 98% (384/392 state-action pairs).

---

## 🗂️ Project Structure

```
REL-MLOPS-PROJECT/
├── sim/
│   └── environment.py        # Grid environment with day/night risk maps
├── rl/
│   └── qlearning.py          # Q-Learning agent
├── api/
│   └── app.py                # FastAPI REST endpoint
├── tests/
│   ├── test_environment.py   # Unit tests (4 tests)
│   └── test_integration.py   # Integration tests (2 tests)
├── configs/
│   └── qlearning.yaml        # Hyperparameters and config
├── experiments/
│   ├── logs/
│   ├── plots/
│   └── policies/
│       ├── policy_v1.pkl
│       └── policy_v2_final.pkl
├── .github/workflows/
│   └── train.yml
├── Dockerfile
├── requirements.txt
├── train.py                  # Main training script
└── visualize.py              # Grid path visualisation
```


---

## ⚙️ Setup and Installation

### Local Setup
```bash
# Clone the repository
git clone https://github.com/DhanyaAR/Women_Safety_Path_Recommendation_Q_learning_MLOPS.git
cd Women_Safety_Path_Recommendation_Q_learning_MLOPS

# Install dependencies
pip install -r requirements.txt

# Create required folders
mkdir -p experiments/logs experiments/policies experiments/plots
```

### Run Training
```bash
python train.py
```

---

## 🔁 Reproducing Results

To reproduce the exact experiment results:
```bash
python train.py
```
Hyperparameters are loaded from `configs/qlearning.yaml`. The default 
config (alpha=0.1, gamma=0.9, epsilon_decay=0.995) produces:
- Day risk reduction: 40.0%
- Night risk reduction: 51.5%
- Q-table coverage: 98%

To run a different experiment, edit `configs/qlearning.yaml` and re-run. 
Each run is automatically tracked in MLflow and logged to 
`experiments/logs/training_log.csv`.

---

## 🚀 FastAPI — Route Recommendation Endpoint

```bash
# Start the API
python -m uvicorn api.app:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Get safe route for day or night |
| `/predict/day` | GET | Get safe day route |
| `/predict/night` | GET | Get safe night route |

**Example response:**
```json
{
  "time_of_day": "night",
  "total_risk": 33,
  "steps": 13,
  "bfs_baseline_risk": 68,
  "risk_reduction_vs_bfs": "51.5%",
  "verdict": "RL route is 51.5% safer than the shortest path"
}
```

---

## 🐳 Docker

```bash
# Build
docker build -t rel-mlops-project .

# Run
docker run rel-mlops-project
```

---

## 📊 MLflow Experiment Tracking

```bash
# View experiment dashboard
mlflow ui
# Open http://127.0.0.1:5000
```

Tracks per run: alpha, gamma, epsilon, reward per episode, 
day/night risk scores, risk reduction %, Q-table coverage.

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

**6/6 tests passing:**
- `test_reset_returns_valid_state`
- `test_step_boundary_penalty`
- `test_goal_reached`
- `test_risk_maps_loaded`
- `test_full_training_loop`
- `test_api_predict_response_structure`

---

## ⚙️ CI/CD

GitHub Actions automatically runs on every push to `main`:
1. Install dependencies
2. Run all 6 tests
3. Run full training script

---

## 📈 Monitoring Plan

If deployed in a real urban navigation system, the following would be monitored:
- **Route risk score per request** — detect if recommendations degrade over time
- **API response time** — ensure real-time usability
- **Prediction drift** — if risk patterns change seasonally, the risk map needs updating
- **Request volume by time of day** — validate day/night split matches real usage
- **Alert threshold** — if average risk score rises above BFS baseline, 
  the RL policy is no longer outperforming naive routing

---

## 🔮 Future Work

- Replace static risk maps with real crime/safety data from city APIs
- Extend to larger city grids using Deep Q-Network (DQN)
- Add SARSA as comparison algorithm
- Implement risk map seasonal updates via MLflow model registry
- Add save/load methods to QLearningAgent class
- Expand test coverage for loop penalty and night map validation

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| RL Algorithm | Q-Learning (tabular) |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow + CSV logger |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Config Management | YAML |
| Testing | pytest |
| Visualization | Matplotlib |

---

## 📚 References

- Sutton & Barto — Reinforcement Learning: An Introduction (2018)
- UN SDG 11 — Sustainable Cities and Communities
- MLflow Documentation — https://mlflow.org
- FastAPI Documentation — https://fastapi.tiangolo.com
