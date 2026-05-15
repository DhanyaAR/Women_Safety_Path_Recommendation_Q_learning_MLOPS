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
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; background: #0f0f0f; color: white; padding: 20px; }
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
        .grid-section { display: flex; gap: 20px; justify-content: center; margin-top: 20px; flex-wrap: wrap; }
        .grid-wrap { text-align: center; }
        .grid-title { color: #aaa; margin-bottom: 8px; font-size: 13px; }
        canvas { border-radius: 8px; border: 1px solid #333; }
        .legend { display: flex; gap: 16px; justify-content: center; margin-top: 12px; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #aaa; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
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

        <div class="grid-section">
            <div class="grid-wrap">
                <div class="grid-title">🟠 RL Agent Path</div>
                <canvas id="rlCanvas" width="280" height="280"></canvas>
            </div>
            <div class="grid-wrap">
                <div class="grid-title">🔵 BFS Shortest Path</div>
                <canvas id="bfsCanvas" width="280" height="280"></canvas>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#4ade80"></div> Start</div>
            <div class="legend-item"><div class="legend-dot" style="background:#f87171"></div> Goal</div>
            <div class="legend-item"><div class="legend-dot" style="background:#f97316"></div> RL Path</div>
            <div class="legend-item"><div class="legend-dot" style="background:#60a5fa"></div> BFS Path</div>
            <div class="legend-item"><div class="legend-dot" style="background:#1e1b4b"></div> Safe (low risk)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div> Dangerous (high risk)</div>
        </div>
    </div>

    <script>
        const DAY_RISK = [
            [1,1,2,8,9,9,1],[1,9,9,9,9,9,1],[1,1,1,1,1,9,1],
            [9,9,9,9,1,9,1],[1,1,1,9,1,1,1],[1,9,1,9,9,9,1],[1,1,1,1,1,1,0]
        ];
        const NIGHT_RISK = [
            [1,1,2,5,6,6,1],[1,9,9,9,9,9,1],[1,8,8,8,1,9,1],
            [9,9,9,9,1,9,1],[1,1,1,9,7,7,7],[1,9,1,9,9,9,1],[9,9,9,9,9,9,0]
        ];

        function riskColor(risk) {
            const t = risk / 9;
            const r = Math.round(30 + t * 220);
            const g = Math.round(27 + (1-t) * 150);
            const b = Math.round(75 * (1-t));
            return `rgb(${r},${g},${b})`;
        }

        function drawGrid(canvasId, riskMap, path, pathColor) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const size = 40;

            // Draw risk map
            for (let r = 0; r < 7; r++) {
                for (let c = 0; c < 7; c++) {
                    ctx.fillStyle = riskColor(riskMap[r][c]);
                    ctx.fillRect(c * size, r * size, size, size);
                    ctx.strokeStyle = '#111';
                    ctx.strokeRect(c * size, r * size, size, size);

                    // Risk number
                    ctx.fillStyle = 'rgba(255,255,255,0.4)';
                    ctx.font = '10px Arial';
                    ctx.fillText(riskMap[r][c], c * size + 15, r * size + 25);
                }
            }

            // Draw path
            ctx.strokeStyle = pathColor;
            ctx.lineWidth = 3;
            ctx.beginPath();
            for (let i = 0; i < path.length; i++) {
                const [r, c] = path[i];
                const x = c * size + size/2;
                const y = r * size + size/2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw dots on path
            for (let i = 0; i < path.length; i++) {
                const [r, c] = path[i];
                const x = c * size + size/2;
                const y = r * size + size/2;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = pathColor;
                ctx.fill();
            }

            // Start — green
            ctx.beginPath();
            ctx.arc(size/2, size/2, 8, 0, Math.PI * 2);
            ctx.fillStyle = '#4ade80';
            ctx.fill();

            // Goal — red
            ctx.beginPath();
            ctx.arc(6*size + size/2, 6*size + size/2, 8, 0, Math.PI * 2);
            ctx.fillStyle = '#f87171';
            ctx.fill();
        }

        function bfsPath() {
            const grid = 7;
            const start = [0,0], goal = [6,6];
            const queue = [[start, [start]]];
            const visited = new Set();
            while (queue.length) {
                const [curr, path] = queue.shift();
                const key = curr.join(',');
                if (curr[0]===goal[0] && curr[1]===goal[1]) return path;
                if (visited.has(key)) continue;
                visited.add(key);
                for (const [dr,dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                    const nr = curr[0]+dr, nc = curr[1]+dc;
                    if (nr>=0&&nr<grid&&nc>=0&&nc<grid)
                        queue.push([[nr,nc], [...path,[nr,nc]]]);
                }
            }
            return [];
        }

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

            const riskMap = time === 'day' ? DAY_RISK : NIGHT_RISK;
            const naivePath = bfsPath();

            drawGrid('rlCanvas', riskMap, data.path, '#f97316');
            drawGrid('bfsCanvas', riskMap, naivePath, '#60a5fa');
        }
    </script>
</body>
</html>
"""