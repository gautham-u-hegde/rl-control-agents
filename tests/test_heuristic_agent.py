from agents.heuristic_agent import HeuristicAgent


def test_heuristic_negative_angle():
    agent = HeuristicAgent()
    state = [0, 0, -0.1, 0]
    assert agent.select_action(state) == 0


def test_heuristic_positive_angle():
    agent = HeuristicAgent()
    state = [0, 0, 0.1, 0]
    assert agent.select_action(state) == 1