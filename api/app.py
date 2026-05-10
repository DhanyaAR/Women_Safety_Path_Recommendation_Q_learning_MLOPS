from fastapi import FastAPI
import pickle
from collections import deque
from sim.environment import Environment

app = FastAPI(
    title="Women's Safety Route Recommendation API",
    description="Returns the safest route learned by the RL agent.",
    version="1.0"
)

# Load trained policy
with open("experiments/policies/policy_v2_final.pkl", "rb") as f:
    q_table = pickle.load(f)

env = Environment()

# ================= HELPERS =================
def get_q(state, action):
    return q_table.get((state, action), 0.0)

def get_rl_path(time_mode):
    env.reset()
    env.time_of_day = time_mode
    state = (env.start[0], env.start[1], time_mode)
    path = [(env.start[0], env.start[1])]
    for _ in range(50):
        q_values = [get_q(state, a) for a in env.actions]
        action = q_values.index(max(q_values))
        next_state, reward, done = env.step(action)
        path.append((next_state[0], next_state[1]))
        state = next_state
        if done:
            break
    return path

def get_bfs_path():
    start, goal = env.start, env.goal
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for move in env.actions.values():
            nx, ny = current[0] + move[0], current[1] + move[1]
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                queue.append(((nx, ny), path + [(nx, ny)]))
    return []

def calculate_risk(path, time_mode):
    total = 0
    for x, y in path:
        total += env.day_risk_map[x][y] if time_mode == "day" else env.night_risk_map[x][y]
    return total

# ================= ROUTES =================
@app.get("/")
def home():
    return {"message": "Women's Safety Route Recommendation API is running."}

@app.post("/predict")
def predict(time_of_day: str):
    if time_of_day not in ["day", "night"]:
        return {"error": "time_of_day must be 'day' or 'night'"}

    rl_path  = get_rl_path(time_of_day)
    bfs_path = get_bfs_path()

    rl_risk  = calculate_risk(rl_path, time_of_day)
    bfs_risk = calculate_risk(bfs_path, time_of_day)
    reduction = round((1 - rl_risk / bfs_risk) * 100, 1)

    return {
        "time_of_day":           time_of_day,
        "path":                  rl_path,
        "total_risk":            rl_risk,
        "steps":                 len(rl_path),
        "bfs_baseline_risk":     bfs_risk,
        "risk_reduction_vs_bfs": f"{reduction}%",
        "verdict":               f"RL route is {reduction}% safer than the shortest path"
    }

@app.get("/predict/day")
def predict_day():
    rl_path  = get_rl_path("day")
    bfs_path = get_bfs_path()
    rl_risk  = calculate_risk(rl_path, "day")
    bfs_risk = calculate_risk(bfs_path, "day")
    reduction = round((1 - rl_risk / bfs_risk) * 100, 1)

    return {
        "time_of_day":           "day",
        "path":                  rl_path,
        "total_risk":            rl_risk,
        "steps":                 len(rl_path),
        "bfs_baseline_risk":     bfs_risk,
        "risk_reduction_vs_bfs": f"{reduction}%",
        "verdict":               f"RL route is {reduction}% safer than the shortest path"
    }

@app.get("/predict/night")
def predict_night():
    rl_path  = get_rl_path("night")
    bfs_path = get_bfs_path()
    rl_risk  = calculate_risk(rl_path, "night")
    bfs_risk = calculate_risk(bfs_path, "night")
    reduction = round((1 - rl_risk / bfs_risk) * 100, 1)

    return {
        "time_of_day":           "night",
        "path":                  rl_path,
        "total_risk":            rl_risk,
        "steps":                 len(rl_path),
        "bfs_baseline_risk":     bfs_risk,
        "risk_reduction_vs_bfs": f"{reduction}%",
        "verdict":               f"RL route is {reduction}% safer than the shortest path"
    }