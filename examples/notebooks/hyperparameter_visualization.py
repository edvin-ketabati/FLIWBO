"""Plotting helpers for the FLIWBO hyperparameter visualization notebook."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betainc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from fliwbo_core.BO_utils import beta_warp_nd, make_warp_library
from fliwbo_core.warp_optimizer import optimize_warp_coordinatewise


LATENT_ALPHA_WARP = 25.093
LATENT_BETA_WARP = 8.073

X_OBS = np.array(
    [0.05, 0.15, 0.30, 0.50, 0.62, 0.68, 0.72, 0.76, 0.80, 0.86, 0.92, 0.97],
    dtype=float,
).reshape(-1, 1)
X_SPARSE_OBS = np.array([0.05, 0.68, 0.76, 0.82], dtype=float).reshape(-1, 1)
X_GRID = np.linspace(0.001, 0.999, 400, dtype=float).reshape(-1, 1)


def toy_objective(x: np.ndarray) -> np.ndarray:
    """Non-stationary 1D objective built from a smoother latent coordinate."""

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    z = betainc(LATENT_ALPHA_WARP, LATENT_BETA_WARP, np.clip(x, 1e-12, 1 - 1e-12))

    baseline = 0.50 + 0.08 * z
    ripple_1 = 0.055 * np.sin(2.0 * np.pi * z + 0.2)
    ripple_2 = 0.035 * np.sin(8.0 * np.pi * z - 0.7)
    bump_broad = 0.16 * np.exp(-0.5 * ((z - 0.36) / 0.11) ** 2)
    valley_mid = -0.09 * np.exp(-0.5 * ((z - 0.60) / 0.07) ** 2)
    peak_sharp = 0.24 * np.exp(-0.5 * ((z - 0.86) / 0.035) ** 2)

    y = baseline + ripple_1 + ripple_2 + bump_broad + valley_mid + peak_sharp
    return y.ravel()


Y_OBS = toy_objective(X_OBS.ravel())
Y_SPARSE_OBS = toy_objective(X_SPARSE_OBS.ravel())
Y_GRID = toy_objective(X_GRID.ravel())
Y_CENTER = float(np.mean(Y_OBS))
Y_SCALE = float(np.std(Y_OBS))

DEFAULT_LENGTHSCALE = 0.18
DEFAULT_NOISE_STD = 0.03
DEFAULT_EPSILON_WARP = 3.0
DEFAULT_WARP_PRIOR_WEIGHT = 0.005
DEFAULT_WARP_PRIOR_TAU = 0.75


@dataclass(frozen=True)
class FitResult:
    """Posterior fit and warp metadata for one plotted setting."""

    mean: np.ndarray
    std: np.ndarray
    alpha: float
    beta: float
    warp_score: float | None
    n_scored: int
    library_size: int
    x_obs: np.ndarray
    y_obs: np.ndarray


def plot_toy_data():
    """Plot the toy objective and the observed points used by every section."""

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(X_GRID.ravel(), Y_GRID, color="0.35", linestyle="--", label="toy objective")
    ax.scatter(X_OBS.ravel(), Y_OBS, color="black", s=42, zorder=5, label="observed scores")
    ax.set_title("Non-stationary 1D objective")
    ax.set_xlabel("normalized input x")
    ax.set_ylabel("objective score")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    return fig


def plot_lengthscale_panel(values=(0.06, DEFAULT_LENGTHSCALE, 0.60)):
    """Show how the base GP lengthscale changes smoothness."""

    fig, axes = plt.subplots(1, len(values), figsize=(5.1 * len(values), 3.8), sharey=True)
    for ax, value in zip(np.ravel(axes), values):
        fit = fixed_warp_fit(lengthscale=float(value), noise_std=DEFAULT_NOISE_STD)
        _plot_fit(ax, fit, f"lengthscale = {value:g}")
    fig.suptitle("Base kernel lengthscale", y=1.03)
    fig.tight_layout()
    return fig


def plot_noise_panel(values=(0.01, DEFAULT_NOISE_STD, 0.24)):
    """Show how assumed observation noise changes the posterior fit."""

    fig, axes = plt.subplots(1, len(values), figsize=(5.1 * len(values), 3.8), sharey=True)
    for ax, value in zip(np.ravel(axes), values):
        fit = fixed_warp_fit(lengthscale=DEFAULT_LENGTHSCALE, noise_std=float(value))
        _plot_fit(ax, fit, f"noise_std = {value:g}")
    fig.suptitle("Assumed observation noise", y=1.03)
    fig.tight_layout()
    return fig


def plot_warp_regularization_panel():
    """Show how regularization affects selected warps with sparse observations."""

    settings = [
        ("off\nweight=0, tau=0.75", 0.0, DEFAULT_WARP_PRIOR_TAU),
        ("balanced\nweight=0.005, tau=0.75", DEFAULT_WARP_PRIOR_WEIGHT, DEFAULT_WARP_PRIOR_TAU),
        ("too strong\nweight=5, tau=0.75", 5.0, DEFAULT_WARP_PRIOR_TAU),
    ]
    return _plot_warp_comparison(
        settings,
        title="Warp regularization with sparse early data",
        epsilon_warp=DEFAULT_EPSILON_WARP,
        data_mode="sparse",
    )


def plot_epsilon_warp_panel(values=(8.0, 2.0)):
    """Plot the Beta-CDF warp curves available in each finite library."""

    fig, axes = plt.subplots(1, len(values), figsize=(5.2 * len(values), 4.0), sharex=True, sharey=True)
    for ax, epsilon in zip(np.ravel(axes), values):
        pairs = make_warp_library(epsilon=float(epsilon))
        for alpha, beta in pairs:
            curve = beta_warp_nd(X_GRID, np.array([alpha]), np.array([beta])).ravel()
            is_unity = np.isclose(alpha, 1.0) and np.isclose(beta, 1.0)
            ax.plot(
                X_GRID.ravel(),
                curve,
                color="black" if is_unity else "#0072B2",
                alpha=0.85 if is_unity else 0.075,
                linewidth=2.2 if is_unity else 0.8,
                linestyle="--" if is_unity else "-",
            )
        ax.set_title(f"epsilon = {epsilon:g}\n{len(pairs)} library warps")
        ax.set_xlabel("input x")
        ax.set_ylabel("warped x")
        ax.grid(True, alpha=0.18)
    fig.suptitle("Finite warp-library coverage", y=1.03)
    fig.tight_layout()
    return fig


@lru_cache(maxsize=128)
def fixed_warp_fit(
    *,
    lengthscale: float,
    noise_std: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> FitResult:
    """Fit the GP with a fixed Beta-CDF warp."""

    alpha_vec = np.array([float(alpha)])
    beta_vec = np.array([float(beta)])
    gpr = _gpr_template(lengthscale=lengthscale, noise_std=noise_std)
    y_scaled, y_center, y_scale = _scaled_y(Y_OBS)
    gpr.fit(beta_warp_nd(X_OBS, alpha_vec, beta_vec), y_scaled)
    mean_scaled, std_scaled = gpr.predict(
        beta_warp_nd(X_GRID, alpha_vec, beta_vec),
        return_std=True,
    )
    return FitResult(
        mean=_unscale_y(mean_scaled, y_center, y_scale),
        std=std_scaled * y_scale,
        alpha=float(alpha),
        beta=float(beta),
        warp_score=None,
        n_scored=0,
        library_size=1,
        x_obs=X_OBS.copy(),
        y_obs=Y_OBS.copy(),
    )


@lru_cache(maxsize=128)
def selected_warp_fit(
    *,
    lengthscale: float = DEFAULT_LENGTHSCALE,
    noise_std: float = DEFAULT_NOISE_STD,
    epsilon_warp: float = DEFAULT_EPSILON_WARP,
    warp_prior_weight: float = DEFAULT_WARP_PRIOR_WEIGHT,
    warp_prior_tau: float = DEFAULT_WARP_PRIOR_TAU,
    data_mode: str = "full",
) -> FitResult:
    """Select a finite-library warp, then fit and predict with that warp."""

    x_obs, y_obs = _observed_data(data_mode)
    y_scaled, y_center, y_scale = _scaled_y(y_obs)
    one_dim_warp_pairs = make_warp_library(epsilon=epsilon_warp)
    result = optimize_warp_coordinatewise(
        X=x_obs,
        y=y_scaled,
        gpr_template=_gpr_template(lengthscale=lengthscale, noise_std=noise_std),
        one_dim_warp_pairs=one_dim_warp_pairs,
        prior_weight=warp_prior_weight,
        prior_tau=warp_prior_tau,
        n_sweeps=1,
        n_jobs=1,
    )
    mean_scaled, std_scaled = result.gpr.predict(
        beta_warp_nd(X_GRID, result.alpha, result.beta),
        return_std=True,
    )
    return FitResult(
        mean=_unscale_y(mean_scaled, y_center, y_scale),
        std=std_scaled * y_scale,
        alpha=float(result.alpha[0]),
        beta=float(result.beta[0]),
        warp_score=float(result.score),
        n_scored=int(result.n_scored),
        library_size=len(one_dim_warp_pairs),
        x_obs=x_obs.copy(),
        y_obs=y_obs.copy(),
    )


def _plot_warp_comparison(
    settings,
    *,
    title: str,
    epsilon_warp: float | None = None,
    data_mode: str = "full",
):
    n_cols = len(settings)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.1 * n_cols, 6.6), sharex="row")
    axes = np.asarray(axes)

    for col_idx, setting in enumerate(settings):
        if len(setting) == 3:
            label, weight, tau = setting
            epsilon = DEFAULT_EPSILON_WARP if epsilon_warp is None else epsilon_warp
        else:
            label, weight, tau, epsilon = setting

        fit = selected_warp_fit(
            epsilon_warp=float(epsilon),
            warp_prior_weight=float(weight),
            warp_prior_tau=float(tau),
            data_mode=data_mode,
        )

        _plot_warp_curve(axes[0, col_idx], fit, label)
        _plot_fit(
            axes[1, col_idx],
            fit,
            f"library={fit.library_size}, scored={fit.n_scored}",
        )

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def _plot_warp_curve(ax, fit: FitResult, title: str) -> None:
    alpha_vec = np.array([fit.alpha])
    beta_vec = np.array([fit.beta])
    warped_grid = beta_warp_nd(X_GRID, alpha_vec, beta_vec).ravel()
    ax.plot(X_GRID.ravel(), X_GRID.ravel(), color="0.55", linestyle="--", label="unity")
    ax.plot(X_GRID.ravel(), warped_grid, color="#0072B2", linewidth=2.0, label="selected")
    ax.set_title(f"{title}\nalpha={fit.alpha:.3g}, beta={fit.beta:.3g}")
    ax.set_xlabel("input x")
    ax.set_ylabel("warped x")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best")


def _plot_fit(ax, fit: FitResult, title: str) -> None:
    x = X_GRID.ravel()
    ax.plot(x, Y_GRID, color="0.35", linestyle="--", linewidth=1.4, label="toy objective")
    ax.plot(x, fit.mean, color="#009E73", linewidth=2.0, label="GP mean")
    ax.fill_between(
        x,
        fit.mean - 2.0 * fit.std,
        fit.mean + 2.0 * fit.std,
        color="#009E73",
        alpha=0.16,
        linewidth=0,
        label="+/- 2 std",
    )
    ax.scatter(fit.x_obs.ravel(), fit.y_obs, color="black", s=34, zorder=5, label="observed")
    ax.set_title(title)
    ax.set_xlabel("normalized input x")
    ax.set_ylabel("objective score")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best", fontsize=8)


def _gpr_template(*, lengthscale: float, noise_std: float) -> GaussianProcessRegressor:
    kernel = Matern(length_scale=float(lengthscale), nu=2.5) + WhiteKernel(
        noise_level=float(noise_std) ** 2
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=False,
    )


def _observed_data(data_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if data_mode == "full":
        return X_OBS, Y_OBS
    if data_mode == "sparse":
        return X_SPARSE_OBS, Y_SPARSE_OBS
    raise ValueError(f"Unknown data_mode: {data_mode!r}")


def _scaled_y(y_obs: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.mean(y_obs))
    scale = float(np.std(y_obs))
    if scale == 0.0:
        scale = 1.0
    return (np.asarray(y_obs, dtype=float) - center) / scale, center, scale


def _unscale_y(y_scaled: np.ndarray, center: float, scale: float) -> np.ndarray:
    return float(center) + float(scale) * np.asarray(y_scaled, dtype=float)
