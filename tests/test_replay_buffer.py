from dqn.replay_buffer import ReplayBuffer


def test_replay_buffer_add_increases_length():
    buffer = ReplayBuffer(capacity=5)
    buffer.add((1, 2, 3, 4), 0, 1.0, (1, 2, 3, 5), False)
    assert len(buffer) == 1


def test_replay_buffer_sample_returns_correct_batch_size():
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.add((i, i, i, i), 0, 1.0, (i + 1, i + 1, i + 1, i + 1), False)

    sample = buffer.sample(3)
    assert len(sample) == 3


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=3)

    for i in range(5):
        buffer.add((i, i, i, i), 0, 1.0, (i + 1, i + 1, i + 1, i + 1), False)

    assert len(buffer) == 3