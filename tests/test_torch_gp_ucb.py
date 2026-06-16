import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from fliwbo_core import Continuous, Discrete, SearchSpace
from fliwbo_core.BO_utils import beta_warp_nd
from fliwbo_core.PR_optimizer import TorchGaussianProcessUCB


def test_torch_gp_ucb_matches_sklearn_predict():
    space = SearchSpace([Discrete(3), Continuous(-1.0, 1.0)])
    X_raw = np.array([[0, -1.0], [1, 0.0], [2, 1.0]])
    y = np.array([0.0, 1.0, 0.25])
    alpha = np.array([1.0, 1.0])
    beta = np.array([1.0, 1.0])
    Z_train = beta_warp_nd(space.normalize_matrix(X_raw), alpha, beta)

    gpr = GaussianProcessRegressor(
        kernel=Matern(length_scale=0.35, nu=2.5) + WhiteKernel(noise_level=1e-4),
        optimizer=None,
        normalize_y=False,
    )
    gpr.fit(Z_train, y)

    X_test = np.array([[0, -0.5], [2, 0.75]])
    Z_test = beta_warp_nd(space.normalize_matrix(X_test), alpha, beta)
    beta_value = 1.7

    mu, std = gpr.predict(Z_test, return_std=True)
    expected = mu + np.sqrt(beta_value) * std
    actual = TorchGaussianProcessUCB(gpr, beta_value)(
        torch.as_tensor(Z_test, dtype=torch.float64)
    ).detach().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
