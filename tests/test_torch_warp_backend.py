import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from fliwbo_core.torch_gp import TorchFixedGaussianProcess
from fliwbo_core.warp_optimizer import optimize_warp_coordinatewise


def test_torch_warp_search_matches_sklearn_indices_and_score():
    X = np.array(
        [
            [0.02, 0.12],
            [0.24, 0.31],
            [0.44, 0.72],
            [0.78, 0.55],
            [0.96, 0.88],
        ],
        dtype=float,
    )
    y = np.array([0.1, -0.2, 0.7, 0.4, -0.1], dtype=float)
    warp_pairs = [
        (0.5, 1.0),
        (1.0, 1.0),
        (2.0, 0.75),
        (4.0, 2.0),
    ]
    length_scale = 0.37
    noise_level = 0.021
    prior_weight = 0.005
    prior_tau = 0.75

    sklearn_template = GaussianProcessRegressor(
        kernel=Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=noise_level),
        optimizer=None,
        normalize_y=False,
    )
    torch_template = TorchFixedGaussianProcess(
        length_scale=length_scale,
        noise_level=noise_level,
        device="cpu",
    )

    sklearn_result = optimize_warp_coordinatewise(
        X=X,
        y=y,
        gpr_template=sklearn_template,
        one_dim_warp_pairs=warp_pairs,
        prior_weight=prior_weight,
        prior_tau=prior_tau,
        n_sweeps=2,
        n_jobs=1,
    )
    torch_result = optimize_warp_coordinatewise(
        X=X,
        y=y,
        gpr_template=torch_template,
        one_dim_warp_pairs=warp_pairs,
        prior_weight=prior_weight,
        prior_tau=prior_tau,
        n_sweeps=2,
        n_jobs=1,
    )

    np.testing.assert_array_equal(torch_result.indices, sklearn_result.indices)
    np.testing.assert_allclose(torch_result.alpha, sklearn_result.alpha)
    np.testing.assert_allclose(torch_result.beta, sklearn_result.beta)
    assert torch_result.score == pytest.approx(sklearn_result.score, rel=1e-10, abs=1e-10)
    assert torch_result.n_scored == sklearn_result.n_scored == len(warp_pairs) * X.shape[1] * 2 + 1
