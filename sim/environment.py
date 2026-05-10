import random

class Environment:
    def __init__(self):
        self.grid_size = 7

        # 🌞 Day risk map
        self.day_risk_map = [
            [1, 1, 2, 8, 9, 9, 1],
            [1, 9, 9, 9, 9, 9, 1],
            [1, 1, 1, 1, 1, 9, 1],
            [9, 9, 9, 9, 1, 9, 1],
            [1, 1, 1, 9, 1, 1, 1],
            [1, 9, 1, 9, 9, 9, 1],
            [1, 1, 1, 1, 1, 1, 0]
        ]

        # 🌙 Night risk map (harder but solvable)
        self.night_risk_map = [
            [1, 1, 2, 5, 6, 6, 1],
            [1, 9, 9, 9, 9, 9, 1],
            [1, 8, 8, 8, 1, 9, 1],
            [9, 9, 9, 9, 1, 9, 1],
            [1, 1, 1, 9, 7, 7, 7],
            [1, 9, 1, 9, 9, 9, 1],
            [9, 9, 9, 9, 9, 9, 0]   # goal still reachable
        ]

        self.start = (0, 0)
        self.goal = (6, 6)

        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        self.reset()

    # ================= RESET =================
    def reset(self):
        self.agent_pos = self.start
        self.time_of_day = random.choice(["day", "night"])

        # 🔥 Track visited cells to prevent loops
        self.visited = set()
        self.visited.add(self.agent_pos)

        return (*self.agent_pos, self.time_of_day)

    # ================= STEP =================
    def step(self, action):
        move = self.actions[action]

        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]

        # 🚫 Boundary penalty
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return (*self.agent_pos, self.time_of_day), -20, False

        self.agent_pos = (new_x, new_y)
        x, y = self.agent_pos

        # 🌞🌙 Select risk map
        if self.time_of_day == "day":
            risk = self.day_risk_map[x][y]
        else:
            risk = self.night_risk_map[x][y]

        # 🔥 Base reward (risk + time penalty)
        reward = -(risk * 5) - 5

        # 🔥 Loop penalty (VERY IMPORTANT)
        if self.agent_pos in self.visited:
            reward -= 10

        # Track visited
        self.visited.add(self.agent_pos)

        done = False

        # 🎯 Goal reward
        if self.agent_pos == self.goal:
            reward = 50
            done = True

        return (*self.agent_pos, self.time_of_day), reward, done