import numpy as np
import torch

from fliwbo_core.PR_optimizer import PRDimension, PROptimizerConfig, optimize_with_restarts


def test_discrete_pr_returns_valid_choice_indices():
    config = PROptimizerConfig(num_restarts=1, num_steps=2, num_samples=4)

    x_best, value = optimize_with_restarts(
        lambda z: -torch.abs(z[:, 0] - 2.0),
        [PRDimension.discrete([0.0, 1.0, 2.0])],
        config,
        seed=11,
    )

    assert x_best.shape == (1,)
    assert x_best[0] in {0, 1, 2}
    assert np.isfinite(value)


def test_continuous_pr_returns_value_inside_warped_bounds():
    config = PROptimizerConfig(num_restarts=1, num_steps=3, num_samples=4)

    x_best, value = optimize_with_restarts(
        lambda z: -torch.square(z[:, 0] - 0.7),
        [PRDimension.continuous((0.1, 0.9))],
        config,
        seed=12,
    )

    assert x_best.shape == (1,)
    assert 0.1 <= x_best[0] <= 0.9
    assert np.isfinite(value)


def test_mixed_pr_returns_valid_coordinates():
    config = PROptimizerConfig(num_restarts=1, num_steps=3, num_samples=4)

    x_best, value = optimize_with_restarts(
        lambda z: -torch.square(z[:, 0] - 1.0) - torch.square(z[:, 1] - 0.75),
        [
            PRDimension.discrete([0.0, 1.0, 2.0]),
            PRDimension.continuous((0.2, 0.8)),
        ],
        config,
        seed=13,
    )

    assert x_best.shape == (2,)
    assert x_best[0] in {0, 1, 2}
    assert 0.2 <= x_best[1] <= 0.8
    assert np.isfinite(value)
