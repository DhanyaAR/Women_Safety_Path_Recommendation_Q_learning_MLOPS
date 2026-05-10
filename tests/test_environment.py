from sim.environment import Environment

def test_reset_returns_valid_state():
    env = Environment()
    state = env.reset()
    assert state[0] == 0 and state[1] == 0
    assert state[2] in ["day", "night"]

def test_step_boundary_penalty():
    env = Environment()
    env.reset()
    # Agent at (0,0), action UP (-1,0) hits boundary
    _, reward, done = env.step(0)
    assert reward == -20
    assert done == False

def test_goal_reached():
    env = Environment()
    env.reset()
    env.agent_pos = (5, 6)
    _, reward, done = env.step(1)  # move down to (6,6)
    assert done == True
    assert reward == 50

def test_risk_maps_loaded():
    env = Environment()
    assert len(env.day_risk_map) == 7
    assert len(env.night_risk_map) == 7
    assert env.day_risk_map[6][6] == 0   # goal is zero risk