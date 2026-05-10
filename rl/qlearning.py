import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma

        # 🔥 Improved exploration strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        # ε-greedy with decay
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.actions.keys()))
        else:
            q_values = [self.get_q(state, a) for a in self.actions]
            max_q = max(q_values)

            # 🔥 handle ties (important improvement)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)

        next_q_values = [self.get_q(next_state, a) for a in self.actions]
        max_next_q = max(next_q_values)

        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        # 🔥 gradually reduce exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay