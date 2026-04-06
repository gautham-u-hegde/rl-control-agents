import gymnasium as gym
import numpy as np

class CartPoleWrapper:
    """
    Wrapper for CartPole-v1 with deterministic seeding.
    """

    def __init__(self, seed: int = 42, render_mode=None):
        self.seed = seed
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self._set_seed(seed)

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action: int):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        self.env.close()
