
import numpy as np


class StateDiscretizer:
    """
    Discretizes continuous CartPole states into finite bins
    for tabular Q-learning.
    """

    def __init__(self, bins=(6, 6, 12, 12)):
        self.bins = bins

        self.lower_bounds = np.array([-4.8, -4.0, -0.418, -4.0])
        self.upper_bounds = np.array([4.8, 4.0, 0.418, 4.0])

    def discretize(self, state):
        state = np.array(state)

        ratios = (state - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        ratios = np.clip(ratios, 0, 0.9999)

        discrete_state = [
            int(ratios[i] * self.bins[i])
            for i in range(len(state))
        ]

        return tuple(discrete_state)