class HeuristicAgent:
    """
    Simple rule-based agent using pole angle.
    """

    def select_action(self, state):
        angle = state[2]
        return 0 if angle < 0 else 1