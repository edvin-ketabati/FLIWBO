"""Compatibility wrapper for older scripts.

New integrations should prefer FLIWBOOptimizer directly. This module keeps the
older run_bo_warped(...) function available while routing it through the newer
durable optimizer API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .BO_config import (
    LENGTHSCALE,
    N_ITERS,
    OBS_NOISE,
    USE_WARP_PRIOR,
    WARP_UNITY_PRIOR_TAU,
    WARP_UNITY_PRIOR_WEIGHT,
)
from .BO_utils import beta_t
from .optimizer import (
    FLIWBOConfig,
    FLIWBOOptimizer,
    ObjectiveFunction,
    OptimizationResult,
    default_pr_config,
)
from .PR_optimizer import PROptimizerConfig


def run_bo_warped(
    objective_fn: ObjectiveFunction,
    X_init: np.ndarray,
    y_init: np.ndarray,
    beta_fn=beta_t,
    n_iters=N_ITERS,
    noise_std=OBS_NOISE,
    lengthscale=LENGTHSCALE,
    use_warp_prior=USE_WARP_PRIOR,
    warp_prior_weight=WARP_UNITY_PRIOR_WEIGHT,
    warp_prior_tau=WARP_UNITY_PRIOR_TAU,
    choice_sizes: list[int] | None = None,
    pr_config: PROptimizerConfig | None = None,
    pr_seed: int | None = None,
    metadata_dir: str | Path | None = None,
) -> OptimizationResult:
    """
    Backwards-compatible wrapper around the public FLIWBOOptimizer API.

    objective_fn is called as objective_fn(x_vector). Iteration-specific naming,
    logging, or runtime setup should live inside the objective adapter.
    """
    if choice_sizes is None:
        raise ValueError(
            "choice_sizes must be provided. Domain-specific defaults belong in the "
            "example adapter, not in fliwbo_core."
        )

    config = FLIWBOConfig(
        n_iters=n_iters,
        noise_std=noise_std,
        lengthscale=lengthscale,
        use_warp_prior=use_warp_prior,
        warp_prior_weight=warp_prior_weight,
        warp_prior_tau=warp_prior_tau,
        pr_config=pr_config or default_pr_config(),
        pr_seed=pr_seed,
        log_csv=True,
        metadata_dir=metadata_dir or Path("BO metadata"),
        verbose=True,
    )
    optimizer = FLIWBOOptimizer(
        choice_sizes,
        config=config,
        beta_fn=beta_fn,
    )
    return optimizer.run(objective_fn, X_init, y_init)
