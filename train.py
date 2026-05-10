from sim.environment import Environment
from rl.qlearning import QLearningAgent
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from visualize import plot_grid
import pickle
import yaml
import os
import csv

# ================= LOAD CONFIG =================
with open("configs/qlearning.yaml", "r") as f:
    config = yaml.safe_load(f)

hp = config["hyperparameters"]
tr = config["training"]
log_file="experiments/logs/training_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run_id", "episode", "total_reward", "avg_reward","epsilon", "alpha", "gamma", "qtable_coverage"])

# ================= INIT =================
env = Environment()
agent = QLearningAgent(
    actions=env.actions,
    alpha=hp["alpha"],
    gamma=hp["gamma"],
    epsilon=hp["epsilon"],
    epsilon_decay=hp["epsilon_decay"],
    epsilon_min=hp["epsilon_min"]
)

episodes = tr["episodes"]
max_steps = tr["max_steps_per_episode"]   # 🔥 guard added
episode_rewards = []

# ================= TRAINING =================
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0                              # 🔥 step counter

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        steps += 1                         # 🔥 increment

        if done or steps >= max_steps:     # 🔥 guard — won't affect good episodes
            break

    agent.decay_epsilon()

    episode_rewards.append(total_reward)

    # Save training metrics to CSV
    coverage = len(agent.q_table) / (7 * 7 * 2 * 4) * 100
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            1,                        # run_id (increment manually for each experiment)
            ep + 1,                   # episode
            total_reward,             # total reward this episode
            round(total_reward / (ep + 1), 2),  # avg reward so far
            round(agent.epsilon, 5),  # epsilon
            hp["alpha"],              # alpha from config
            hp["gamma"],              # gamma from config
            round(coverage, 1)        # q-table coverage %
        ])

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    if ep == tr["policy_mid_checkpoint"] - 1:
        with open("experiments/policies/policy_v1.pkl", "wb") as f:   # 🔥 renamed
            pickle.dump(agent.q_table, f)
        print(f"\n✅ Saved policy_v1.pkl at episode {ep+1}")


# ================= RL PATH (DAY & NIGHT) =================
def get_rl_path(env, agent, time_mode):
    env.reset()
    env.time_of_day = time_mode
    state = (env.start[0], env.start[1], time_mode)
    path = [(env.start[0], env.start[1])]

    for _ in range(50):
        q_values = [agent.get_q(state, a) for a in env.actions]
        action = q_values.index(max(q_values))
        next_state, reward, done = env.step(action)
        path.append((next_state[0], next_state[1]))
        state = next_state
        if done:
            break

    return path


with open("experiments/policies/policy_v2_final.pkl", "wb") as f:     # 🔥 renamed
    pickle.dump(agent.q_table, f)
print("✅ Saved policy_v2_final.pkl")

# 🔥 Q-table size printout
print(f"\n📊 Q-table: {len(agent.q_table)} state-action pairs explored")
print(f"   Max possible: {7 * 7 * 2 * 4} (7x7 grid × 2 time modes × 4 actions)")
coverage = len(agent.q_table) / (7 * 7 * 2 * 4) * 100
print(f"   Coverage: {coverage:.1f}%")

print("\n--- RL Learned Paths ---")
rl_path_day = get_rl_path(env, agent, "day")
rl_path_night = get_rl_path(env, agent, "night")
print("Day Path:", rl_path_day)
print("Night Path:", rl_path_night)


# ================= BASELINE (BFS) =================
def shortest_path(env):
    start = env.start
    goal = env.goal
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
            nx = current[0] + move[0]
            ny = current[1] + move[1]
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                queue.append(((nx, ny), path + [(nx, ny)]))
    return []


print("\n--- Baseline (Shortest Path - BFS) ---")
naive_path = shortest_path(env)
print("Naive Path:", naive_path)


# ================= RISK CALCULATION =================
def calculate_total_risk(path, env, time_mode):
    total = 0
    for (x, y) in path:
        risk = env.day_risk_map[x][y] if time_mode == "day" else env.night_risk_map[x][y]
        total += risk
    return total


# ================= METRICS =================
print("\n--- Metrics Comparison ---")
day_rl_risk   = calculate_total_risk(rl_path_day, env, "day")
day_bfs_risk  = calculate_total_risk(naive_path, env, "day")
night_rl_risk = calculate_total_risk(rl_path_night, env, "night")
night_bfs_risk= calculate_total_risk(naive_path, env, "night")

print("DAY:")
print(f"  RL Risk: {day_rl_risk}   Naive Risk: {day_bfs_risk}   Reduction: {(1 - day_rl_risk/day_bfs_risk)*100:.1f}%")
print(f"  RL Length: {len(rl_path_day)}   Naive Length: {len(naive_path)}")

print("\nNIGHT:")
print(f"  RL Risk: {night_rl_risk}   Naive Risk: {night_bfs_risk}   Reduction: {(1 - night_rl_risk/night_bfs_risk)*100:.1f}%")
print(f"  RL Length: {len(rl_path_night)}   Naive Length: {len(naive_path)}")


# ================= GRAPHS =================

# — Smoothed training curve
window = 50
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.figure()
plt.plot(smoothed, color='steelblue')
plt.title("Smoothed Training Performance")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.tight_layout()
plt.show()

# — Risk comparison (🔥 coloured bars: orange=RL, steelblue=Naive)
labels = ['Day RL', 'Day Naive', 'Night RL', 'Night Naive']
values = [day_rl_risk, day_bfs_risk, night_rl_risk, night_bfs_risk]
colors = ['darkorange', 'steelblue', 'darkorange', 'steelblue']

plt.figure()
bars = plt.bar(labels, values, color=colors)
plt.title("Risk Comparison — RL vs Naive")
plt.ylabel("Total Risk")

# legend patch
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='darkorange', label='RL Agent'),
                   Patch(facecolor='steelblue',  label='BFS Naive')]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.show()

# — Path length comparison (same colour treatment)
lengths = [len(rl_path_day), len(naive_path), len(rl_path_night), len(naive_path)]
plt.figure()
plt.bar(labels, lengths, color=colors)
plt.title("Path Length Comparison — RL vs Naive")
plt.ylabel("Steps")
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.show()

# — Grid visualizations
plot_grid(env, rl_path_day,   naive_path, "day")
plot_grid(env, rl_path_night, naive_path, "night")