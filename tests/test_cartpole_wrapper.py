from envs.cartpole_wrapper import CartPoleWrapper


def test_reset_returns_state():
    env = CartPoleWrapper(seed=123)
    state = env.reset()
    assert len(state) == 4
    env.close()


def test_step_returns_expected_values():
    env = CartPoleWrapper(seed=123)
    env.reset()
    next_state, reward, done = env.step(0)
    assert len(next_state) == 4
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()