from agents.random_agent import RandomAgent


def test_random_agent_action_is_valid():
    agent = RandomAgent(action_size=2)
    action = agent.select_action()
    assert action in [0, 1]


def test_random_agent_multiple_actions_are_valid():
    agent = RandomAgent(action_size=2)
    actions = [agent.select_action() for _ in range(20)]
    assert all(action in [0, 1] for action in actions)