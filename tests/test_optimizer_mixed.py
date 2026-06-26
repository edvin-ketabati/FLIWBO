import numpy as np
import pytest

from fliwbo_core import Continuous, Discrete, FLIWBOConfig, FLIWBOOptimizer
from fliwbo_core import PROptimizerConfig, SearchSpace


@pytest.mark.parametrize("backend", ["sklearn", "torch", "auto"])
def test_optimizer_runs_one_mixed_ask_tell_iteration(tmp_path, backend):
    search_space = SearchSpace([Discrete(3), Continuous(-1.0, 1.0)])

    def objective(x_vector):
        return -float((x_vector[0] - 1) ** 2 + (x_vector[1] - 0.25) ** 2)

    X_init = np.array([[0, -1.0], [2, 1.0], [1, 0.0]])
    y_init = np.array([objective(row) for row in X_init])
    config = FLIWBOConfig(
        n_iters=1,
        epsilon_warp=30.0,
        warp_search_sweeps=1,
        warp_search_n_jobs=1,
        backend=backend,
        device="cpu",
        pr_config=PROptimizerConfig(num_restarts=1, num_steps=2, num_samples=4),
        pr_seed=123,
    )
    optimizer = FLIWBOOptimizer(
        search_space,
        config=config,
        beta_fn=lambda iteration, N_eps: 1.0,
    )

    run = optimizer.start(X_init, y_init, run_dir=tmp_path / f"mixed_run_{backend}")
    proposal = run.ask()

    assert isinstance(proposal.x_vector[0], int)
    assert isinstance(proposal.x_vector[1], float)
    assert 0 <= proposal.x_vector[0] <= 2
    assert -1.0 <= proposal.x_vector[1] <= 1.0

    record = run.tell(proposal, objective(np.asarray(proposal.x_vector, dtype=float)))

    assert record.x_vector == proposal.x_vector
    assert run.is_complete
