"""Mathematical helper functions used by the FLIWBO core."""

from scipy.special import betainc  # regularized incomplete beta = Beta CDF on [0,1]
import numpy as np

from .BO_config import BETA_SCALING


def beta_warp_nd(X, alpha, beta):
    """
    Vectorized Beta-CDF warping for X in [0,1]^D.

    X: (N, D)
    alpha, beta: either scalars (1D) or arrays shape (D,)
    returns: (N, D)
    """
    X = np.asarray(X, dtype=float)
    X = np.clip(X, 1e-12, 1 - 1e-12)

    alpha = np.asarray(alpha, dtype=float)
    beta  = np.asarray(beta, dtype=float)

    if alpha.ndim == 0:
        # 1D scalar case
        return betainc(float(alpha), float(beta), X)

    # D-dim case
    if X.shape[1] != alpha.shape[0] or alpha.shape != beta.shape:
        raise ValueError(f"Shape mismatch: X {X.shape}, alpha {alpha.shape}, beta {beta.shape}")

    Z = np.empty_like(X)
    for d in range(X.shape[1]):
        Z[:, d] = betainc(alpha[d], beta[d], X[:, d])
    return Z


def make_warp_library(
    epsilon=5.0,
    alpha_min=0.1,
    alpha_max=30.0,
    beta_min=0.1,
    beta_max=30.0,
    dim=None,
):
    """
    Build an epsilon-net over the compact parameter box [alpha_min, alpha_max] x [beta_min, beta_max].

    The grid spacing is chosen so that every point in the box is within Euclidean distance epsilon
    of at least one library element.
    """
    step = epsilon / np.sqrt(2.0)

    n_alpha = int(np.ceil((alpha_max - alpha_min) / step)) + 1
    n_beta = int(np.ceil((beta_max - beta_min) / step)) + 1

    alpha_values = np.linspace(alpha_min, alpha_max, n_alpha)
    beta_values = np.linspace(beta_min, beta_max, n_beta)

    if dim is None:
        return [(float(alpha), float(beta)) for alpha in alpha_values for beta in beta_values]

    return [
        (
            np.full(dim, float(alpha), dtype=float),
            np.full(dim, float(beta), dtype=float),
        )
        for alpha in alpha_values
        for beta in beta_values
    ]


def gamma_t(t, dim = 1 , nu = 2.5, c_gamma = 1.5, log_power = 1):
    """Exploration-growth helper used by the beta_t schedule."""

    alpha = (dim * (dim + 1)) / (2.0 * nu + dim * (dim + 1))
    return c_gamma * (t ** alpha) * (np.log1p(t) ** log_power)

def beta_t(t, N_eps, delta = 0.1, Cwarp = 1):
    """UCB exploration schedule for the finite warp library."""

    return (2 * (Cwarp ** 2) + 300.0 * gamma_t(t) * (np.log(t * N_eps / delta) ** 3))/BETA_SCALING


# Larger tau means weaker prior
def log_prior_unity_weak(a, b, tau=0.75):
    """Weak log prior that mildly favors the identity-like warp alpha=beta=1."""

    la = np.log(a)
    lb = np.log(b)
    return float(-0.5 * np.sum(la * la + lb * lb) / (tau * tau))
