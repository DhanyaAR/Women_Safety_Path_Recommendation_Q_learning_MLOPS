from sim.environment import Environment
from rl.qlearning import QLearningAgent

def test_full_training_loop():
    """Agent should reach goal within 200 steps after 500 training episodes."""
    env = Environment()
    agent = QLearningAgent(actions=env.actions)

    # Quick training run
    for ep in range(500):
        state = env.reset()
        for _ in range(200):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
        agent.decay_epsilon()

    # After training, agent should find goal greedily
    agent.epsilon = 0  # pure exploitation
    env.reset()
    env.time_of_day = "day"
    state = (0, 0, "day")
    reached_goal = False

    for _ in range(200):
        q_values = [agent.get_q(state, a) for a in env.actions]
        action = q_values.index(max(q_values))
        next_state, reward, done = env.step(action)
        state = next_state
        if done:
            reached_goal = True
            break

    assert reached_goal, "Agent failed to reach goal after training"

def test_api_predict_response_structure():
    """API response should contain all required fields."""
    from fastapi.testclient import TestClient
    from api.app import app

    client = TestClient(app)
    response = client.get("/predict/day")

    assert response.status_code == 200
    data = response.json()
    assert "path" in data
    assert "total_risk" in data
    assert "bfs_baseline_risk" in data
    assert "risk_reduction_vs_bfs" in data
    assert "verdict" in data