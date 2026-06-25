import pytest

from fliwbo_core import Discrete, FLIWBOConfig, FLIWBOOptimizer, SearchSpace
from fliwbo_core.BO_utils import beta_t


def test_beta_t_uses_explicit_scaling():
    unscaled = beta_t(3, N_eps=11, beta_scaling=1.0)
    scaled = beta_t(3, N_eps=11, beta_scaling=5.0)

    assert scaled == pytest.approx(unscaled / 5.0)


def test_beta_t_uses_explicit_dimension():
    one_dim = beta_t(3, N_eps=11, beta_scaling=5.0, dim=1)
    three_dim = beta_t(3, N_eps=11, beta_scaling=5.0, dim=3)

    assert three_dim != pytest.approx(one_dim)


def test_default_optimizer_beta_schedule_reads_config_scaling():
    optimizer = FLIWBOOptimizer(
        SearchSpace([Discrete(2)]),
        config=FLIWBOConfig(beta_scaling=5.0),
    )

    expected = beta_t(3, N_eps=11, beta_scaling=5.0)

    assert optimizer._beta_value(3, 11) == pytest.approx(expected)


def test_default_optimizer_beta_schedule_uses_search_space_dimension():
    optimizer = FLIWBOOptimizer(
        SearchSpace([Discrete(2), Discrete(3), Discrete(4)]),
        config=FLIWBOConfig(beta_scaling=5.0),
    )

    expected = beta_t(3, N_eps=11, beta_scaling=5.0, dim=3)

    assert optimizer._beta_value(3, 11) == pytest.approx(expected)


def test_custom_beta_fn_keeps_existing_two_argument_call_shape():
    optimizer = FLIWBOOptimizer(
        SearchSpace([Discrete(2)]),
        config=FLIWBOConfig(beta_scaling=5.0),
        beta_fn=lambda iteration, N_eps: iteration + N_eps,
    )

    assert optimizer._beta_value(3, 11) == 14.0


def test_beta_scaling_must_be_positive():
    with pytest.raises(ValueError, match="beta_scaling"):
        FLIWBOOptimizer(
            SearchSpace([Discrete(2)]),
            config=FLIWBOConfig(beta_scaling=0.0),
        )
