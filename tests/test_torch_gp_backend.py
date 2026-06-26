import numpy as np
import pytest
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from fliwbo_core.PR_optimizer import TorchGaussianProcessUCB
from fliwbo_core.torch_gp import TorchFixedGaussianProcess


def test_torch_fixed_gp_matches_sklearn_log_likelihood_and_predict():
    X_train = np.array(
        [
            [0.02, 0.18],
            [0.24, 0.51],
            [0.62, 0.37],
            [0.96, 0.88],
        ],
        dtype=float,
    )
    y = np.array([0.2, -0.3, 0.8, 0.1], dtype=float)
    X_test = np.array([[0.12, 0.2], [0.55, 0.45], [0.9, 0.7]], dtype=float)
    length_scale = 0.41
    noise_level = 0.017

    sklearn_gp = GaussianProcessRegressor(
        kernel=Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=noise_level),
        optimizer=None,
        normalize_y=False,
    ).fit(X_train, y)
    torch_gp = TorchFixedGaussianProcess(
        length_scale=length_scale,
        noise_level=noise_level,
        device="cpu",
    ).fit(X_train, y)

    assert torch_gp.log_marginal_likelihood_value_ == pytest.approx(
        sklearn_gp.log_marginal_likelihood_value_,
        rel=1e-10,
        abs=1e-10,
    )

    expected_mean, expected_std = sklearn_gp.predict(X_test, return_std=True)
    actual_mean, actual_std = torch_gp.predict(X_test, return_std=True)

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_std, expected_std, rtol=1e-10, atol=1e-10)


def test_torch_fixed_gp_ucb_matches_sklearn_ucb():
    X_train = np.array([[0.1], [0.4], [0.8]], dtype=float)
    y = np.array([0.0, 1.0, 0.25], dtype=float)
    X_test = np.array([[0.2], [0.6]], dtype=float)
    beta_value = 1.7
    noise_level = 1e-4

    sklearn_gp = GaussianProcessRegressor(
        kernel=Matern(length_scale=0.35, nu=2.5) + WhiteKernel(noise_level=noise_level),
        optimizer=None,
        normalize_y=False,
    ).fit(X_train, y)
    torch_gp = TorchFixedGaussianProcess(
        length_scale=0.35,
        noise_level=noise_level,
        device="cpu",
    ).fit(X_train, y)

    mean, std = sklearn_gp.predict(X_test, return_std=True)
    expected = mean + np.sqrt(beta_value) * std
    actual = TorchGaussianProcessUCB(torch_gp, beta_value)(
        torch.as_tensor(X_test, dtype=torch.float64)
    ).detach().cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_fixed_gp_cuda_device_matches_cpu_prediction():
    X_train = np.array([[0.1], [0.4], [0.8]], dtype=float)
    y = np.array([0.0, 1.0, 0.25], dtype=float)
    X_test = np.array([[0.2], [0.6]], dtype=float)

    cpu_gp = TorchFixedGaussianProcess(
        length_scale=0.35,
        noise_level=1e-4,
        device="cpu",
    ).fit(X_train, y)
    cuda_gp = TorchFixedGaussianProcess(
        length_scale=0.35,
        noise_level=1e-4,
        device="cuda",
    ).fit(X_train, y)

    assert cuda_gp.X_train_ is not None
    assert cuda_gp.X_train_.device.type == "cuda"

    cpu_mean, cpu_std = cpu_gp.predict(X_test, return_std=True)
    cuda_mean, cuda_std = cuda_gp.predict(X_test, return_std=True)
    np.testing.assert_allclose(cuda_mean, cpu_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cuda_std, cpu_std, rtol=1e-10, atol=1e-10)
