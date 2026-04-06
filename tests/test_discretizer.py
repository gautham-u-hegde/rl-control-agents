from envs.discretizer import StateDiscretizer


def test_discretizer_returns_tuple():
    discretizer = StateDiscretizer()
    state = [0.0, 0.0, 0.0, 0.0]
    discrete_state = discretizer.discretize(state)

    assert isinstance(discrete_state, tuple)
    assert len(discrete_state) == 4


def test_discretizer_values_are_in_range():
    discretizer = StateDiscretizer(bins=(6, 6, 12, 12))
    state = [0.1, -0.2, 0.03, 0.4]
    discrete_state = discretizer.discretize(state)

    assert 0 <= discrete_state[0] < 6
    assert 0 <= discrete_state[1] < 6
    assert 0 <= discrete_state[2] < 12
    assert 0 <= discrete_state[3] < 12