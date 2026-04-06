import torch
from dqn.model import DQN


def test_dqn_output_shape_single_input():
    model = DQN(state_dim=4, action_dim=2)
    sample_input = torch.randn(1, 4)
    output = model(sample_input)
    assert output.shape == (1, 2)


def test_dqn_output_shape_batch_input():
    model = DQN(state_dim=4, action_dim=2)
    sample_input = torch.randn(5, 4)
    output = model(sample_input)
    assert output.shape == (5, 2)