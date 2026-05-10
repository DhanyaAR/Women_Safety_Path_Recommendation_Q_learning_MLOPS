import matplotlib.pyplot as plt
import numpy as np

def plot_grid(env, rl_path, naive_path, time_mode):
    # Select correct grid
    if time_mode == "day":
        grid = np.array(env.day_risk_map)
        title = "Day - RL vs Naive Path"
    else:
        grid = np.array(env.night_risk_map)
        title = "Night - RL vs Naive Path"

    plt.figure(figsize=(6, 6))
    plt.imshow(grid)

    # ---------------- RL PATH ----------------
    xs_rl = [p[1] for p in rl_path]
    ys_rl = [p[0] for p in rl_path]
    plt.plot(xs_rl, ys_rl, marker='o', linewidth=2, label='RL Path', color='orange')

    # ---------------- NAIVE PATH ----------------
    xs_nv = [p[1] for p in naive_path]
    ys_nv = [p[0] for p in naive_path]
    plt.plot(xs_nv, ys_nv, linestyle='--', linewidth=2, label='Naive Path')

    # ---------------- START & GOAL ----------------
    plt.scatter(env.start[1], env.start[0], c='green', s=120, label='Start')
    plt.scatter(env.goal[1], env.goal[0], c='red', s=120, label='Goal')

    # ---------------- GRID SETTINGS ----------------
    plt.title(title)
    plt.colorbar(label="Risk Level")

    plt.xticks(range(env.grid_size))
    plt.yticks(range(env.grid_size))
    plt.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()