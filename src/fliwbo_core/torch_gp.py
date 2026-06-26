"""Torch fixed-kernel Gaussian process helpers.

This module mirrors the subset of sklearn.gaussian_process.GaussianProcessRegressor
that FLIWBO uses: a fixed Matern-5/2 kernel plus WhiteKernel noise, no
hyperparameter optimization, and normalize_y=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


SKLEARN_DEFAULT_ALPHA = 1e-10
_LOG_2_PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class BackendSelection:
    """Resolved optimizer backend and Torch device, if applicable."""

    backend: Literal["sklearn", "torch"]
    device: torch.device | None


def resolve_backend_device(backend: str, device: str) -> BackendSelection:
    """Resolve user-facing backend/device strings to an executable selection."""

    if backend not in {"auto", "sklearn", "torch"}:
        raise ValueError(f"backend must be 'auto', 'sklearn', or 'torch', got {backend!r}")
    if device != "auto" and device != "cpu" and not device.startswith("cuda"):
        raise ValueError(
            "device must be 'auto', 'cpu', 'cuda', or 'cuda:N', "
            f"got {device!r}"
        )

    if backend == "sklearn":
        return BackendSelection(backend="sklearn", device=None)

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    if torch_device.type == "cuda" and not torch.cuda.is_available():
        if backend == "auto" and device == "auto":
            torch_device = torch.device("cpu")
        else:
            raise ValueError(f"Requested Torch device {device!r}, but CUDA is not available")

    return BackendSelection(backend="torch", device=torch_device)


class TorchFixedGaussianProcess:
    """Fixed Matern-5/2 GP matching the sklearn model used by FLIWBO."""

    def __init__(
        self,
        *,
        length_scale: float | np.ndarray,
        noise_level: float,
        alpha: float = SKLEARN_DEFAULT_ALPHA,
        device: torch.device | str = "cpu",
    ):
        self.length_scale_value = np.asarray(length_scale, dtype=float)
        if self.length_scale_value.ndim > 1:
            raise ValueError(
                "TorchFixedGaussianProcess only supports scalar or 1D length_scale"
            )
        self.noise_level = float(noise_level)
        self.alpha = float(alpha)
        self.device = torch.device(device)

        self.length_scale = torch.as_tensor(
            self.length_scale_value,
            dtype=torch.float64,
            device=self.device,
        )
        self.X_train_: torch.Tensor | None = None
        self.y_train_: torch.Tensor | None = None
        self.alpha_: torch.Tensor | None = None
        self.L_: torch.Tensor | None = None
        self.log_marginal_likelihood_value_: float | None = None
        self._y_train_mean = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self._y_train_std = torch.tensor(1.0, dtype=torch.float64, device=self.device)

    def clone_unfitted(self) -> "TorchFixedGaussianProcess":
        return TorchFixedGaussianProcess(
            length_scale=self.length_scale_value.copy(),
            noise_level=self.noise_level,
            alpha=self.alpha,
            device=self.device,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchFixedGaussianProcess":
        X_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float64, device=self.device).reshape(-1)
        if X_t.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {tuple(X_t.shape)}")
        if y_t.shape[0] != X_t.shape[0]:
            raise ValueError(
                f"Expected {X_t.shape[0]} y values, got {y_t.shape[0]}"
            )

        K = matern25_kernel(X_t, X_t, self.length_scale)
        eye = torch.eye(X_t.shape[0], dtype=torch.float64, device=self.device)
        K = K + (self.noise_level + self.alpha) * eye
        L = torch.linalg.cholesky(K)
        y_rhs = y_t.reshape(-1, 1)
        alpha_vec = torch.cholesky_solve(y_rhs, L, upper=False).reshape(-1)

        log_likelihood = _log_marginal_likelihood_from_cholesky(y_t, alpha_vec, L)

        self.X_train_ = X_t
        self.y_train_ = y_t
        self.alpha_ = alpha_vec
        self.L_ = L
        self.log_marginal_likelihood_value_ = float(
            log_likelihood.detach().cpu().item()
        )
        return self

    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        mean_t, std_t = self.predict_torch(X, return_std=True)
        mean = mean_t.detach().cpu().numpy()
        if not return_std:
            return mean
        std = std_t.detach().cpu().numpy()
        return mean, std

    def predict_torch(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        return_std: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.X_train_ is None or self.alpha_ is None or self.L_ is None:
            raise RuntimeError("TorchFixedGaussianProcess must be fit before predict")

        X_t = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)

        K_trans = matern25_kernel(X_t, self.X_train_, self.length_scale)
        mean = K_trans.matmul(self.alpha_)
        if not return_std:
            return mean

        v = torch.linalg.solve_triangular(self.L_, K_trans.T, upper=False)
        variance = 1.0 + self.noise_level - torch.sum(v * v, dim=0)
        variance = torch.clamp(variance, min=0.0)
        return mean, torch.sqrt(variance)

    def batch_log_marginal_likelihood(
        self,
        X_batch: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Return sklearn-style log marginal likelihood for each X batch item."""

        X_t = torch.as_tensor(X_batch, dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float64, device=self.device).reshape(-1)
        if X_t.ndim != 3:
            raise ValueError(f"X_batch must be 3D, got shape {tuple(X_t.shape)}")
        if y_t.shape[0] != X_t.shape[1]:
            raise ValueError(
                f"Expected {X_t.shape[1]} y values, got {y_t.shape[0]}"
            )

        batch_size, n_train, _dim = X_t.shape
        K = matern25_kernel(X_t, X_t, self.length_scale)
        eye = torch.eye(n_train, dtype=torch.float64, device=self.device).expand(
            batch_size,
            n_train,
            n_train,
        )
        K = K + (self.noise_level + self.alpha) * eye
        L = torch.linalg.cholesky(K)
        y_rhs = y_t.reshape(1, n_train, 1).expand(batch_size, n_train, 1)
        alpha = torch.cholesky_solve(y_rhs, L, upper=False).reshape(batch_size, n_train)

        data_fit = torch.sum(y_t.reshape(1, n_train) * alpha, dim=1)
        log_det = torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)
        lml = -0.5 * data_fit - log_det - 0.5 * n_train * _LOG_2_PI
        return lml.detach().cpu().numpy()


def matern25_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    length_scale: torch.Tensor,
) -> torch.Tensor:
    """Matern nu=2.5 kernel with sklearn's unit signal variance."""

    scaled_X = X / length_scale
    scaled_Y = Y / length_scale
    distances = torch.cdist(scaled_X, scaled_Y)
    sqrt5_distances = np.sqrt(5.0) * distances
    return (1.0 + sqrt5_distances + 5.0 / 3.0 * distances * distances) * torch.exp(
        -sqrt5_distances
    )


def _log_marginal_likelihood_from_cholesky(
    y: torch.Tensor,
    alpha: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    n_train = y.shape[0]
    return (
        -0.5 * torch.dot(y, alpha)
        - torch.sum(torch.log(torch.diagonal(L)))
        - 0.5 * n_train * _LOG_2_PI
    )
