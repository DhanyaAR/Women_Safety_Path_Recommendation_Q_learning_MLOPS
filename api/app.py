from fastapi import FastAPI
import pickle
from collections import deque
from sim.environment import Environment
from fastapi.responses import HTMLResponse

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


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Women's Safety Route Recommendation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; background: #0f0f0f; color: white; padding: 20px; }
        h1 { color: #ff6b9d; text-align: center; }
        p { text-align: center; color: #aaa; }
        .btn-group { display: flex; gap: 20px; justify-content: center; margin: 30px 0; }
        button { padding: 15px 40px; font-size: 16px; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; }
        .day-btn { background: #f9c74f; color: black; }
        .night-btn { background: #4361ee; color: white; }
        .result { background: #1a1a2e; border-radius: 12px; padding: 20px; margin-top: 20px; display: none; }
        .verdict { font-size: 20px; color: #4cc9f0; font-weight: bold; text-align: center; margin-bottom: 15px; }
        .metric { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #333; }
        .label { color: #aaa; }
        .value { color: #4cc9f0; font-weight: bold; }
        .safe { color: #4ade80; }
        .path { font-size: 12px; color: #888; margin-top: 10px; word-break: break-all; }
    </style>
</head>
<body>
    <h1>🛡️ Women's Safety Route Recommendation</h1>
    <p>AI-powered route recommendation that learns safer paths using Reinforcement Learning</p>
    <p style="color:#ff6b9d">SDG 11 — Sustainable Cities and Communities</p>

    <div class="btn-group">
        <button class="day-btn" onclick="getRoute('day')">☀️ Get Day Route</button>
        <button class="night-btn" onclick="getRoute('night')">🌙 Get Night Route</button>
    </div>

    <div class="result" id="result">
        <div class="verdict" id="verdict"></div>
        <div class="metric">
            <span class="label">Time of Day</span>
            <span class="value" id="time"></span>
        </div>
        <div class="metric">
            <span class="label">RL Agent Risk Score</span>
            <span class="value safe" id="rl_risk"></span>
        </div>
        <div class="metric">
            <span class="label">BFS Shortest Path Risk</span>
            <span class="value" id="bfs_risk"></span>
        </div>
        <div class="metric">
            <span class="label">Risk Reduction</span>
            <span class="value safe" id="reduction"></span>
        </div>
        <div class="metric">
            <span class="label">Path Length</span>
            <span class="value" id="steps"></span>
        </div>
        <div class="path" id="path"></div>
    </div>

    <script>
        async function getRoute(time) {
            const res = await fetch(`/predict/${time}`);
            const data = await res.json();
            document.getElementById('result').style.display = 'block';
            document.getElementById('verdict').innerText = data.verdict;
            document.getElementById('time').innerText = data.time_of_day.toUpperCase();
            document.getElementById('rl_risk').innerText = data.total_risk;
            document.getElementById('bfs_risk').innerText = data.bfs_baseline_risk;
            document.getElementById('reduction').innerText = data.risk_reduction_vs_bfs;
            document.getElementById('steps').innerText = data.steps + ' steps';
            document.getElementById('path').innerText = 'Path: ' + JSON.stringify(data.path);
        }
    </script>
</body>
</html>
"""