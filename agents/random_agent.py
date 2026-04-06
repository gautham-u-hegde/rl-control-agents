import random


class RandomAgent:
    """
    Random baseline agent for CartPole.
    Selects an action uniformly at random.
    """

    def __init__(self, action_size: int = 2):
        self.action_size = action_size

    def select_action(self) -> int:
        return random.randint(0, self.action_size - 1)