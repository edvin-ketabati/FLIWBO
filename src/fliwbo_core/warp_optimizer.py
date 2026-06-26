"""Finite-library input-warp search.

This module chooses one Beta-CDF warp per input coordinate. The search is
coordinate-wise: hold all other warp choices fixed, score the finite library for
one coordinate, keep the best, and continue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from .BO_utils import beta_warp_nd, log_prior_unity_weak
from .torch_gp import TorchFixedGaussianProcess


@dataclass(frozen=True)
class CoordinateWarpSearchResult:
    """Best warp found for the current BO model fit."""

    alpha: np.ndarray
    beta: np.ndarray
    indices: np.ndarray
    score: float
    gpr: object
    n_scored: int


def optimize_warp_coordinatewise(
    *,
    X: np.ndarray,
    y: np.ndarray,
    gpr_template,
    one_dim_warp_pairs: list[tuple[float, float]],
    prior_weight: float,
    prior_tau: float = 0.75,
    n_sweeps: int = 1,
    n_jobs: int = -1,
    initial_indices: np.ndarray | None = None,
) -> CoordinateWarpSearchResult:
    """
    Choose a factorized warp from the finite one-dimensional warp library.

    prior_weight multiplies the unity-warp log prior in the scoring objective.
    prior_tau controls the width of that prior around alpha=beta=1.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    dim = X.shape[1]

    if initial_indices is None:
        current_indices = np.full(
            dim,
            _unity_like_pair_index(one_dim_warp_pairs),
            dtype=int,
        )
    else:
        current_indices = np.asarray(initial_indices, dtype=int).copy()
        if current_indices.shape != (dim,):
            raise ValueError(f"Expected initial_indices shape {(dim,)}, got {current_indices.shape}")

    n_scored = 0

    for _sweep in range(n_sweeps):
        for coord_idx in range(dim):
            candidate_pair_indices = range(len(one_dim_warp_pairs))
            results = _score_coordinate_candidates(
                candidate_pair_indices,
                coord_idx,
                current_indices,
                one_dim_warp_pairs,
                X,
                y,
                gpr_template,
                prior_weight,
                prior_tau,
                n_jobs,
            )
            n_scored += len(results)

            best_candidate_idx, _best_score = max(results, key=lambda item: item[1])
            current_indices[coord_idx] = best_candidate_idx

    alpha_vec, beta_vec = indices_to_warp_vectors(current_indices, one_dim_warp_pairs)
    score, gpr = fit_and_score_warp(
        alpha_vec,
        beta_vec,
        X,
        y,
        gpr_template,
        prior_weight,
        prior_tau,
    )
    n_scored += 1

    return CoordinateWarpSearchResult(
        alpha=alpha_vec,
        beta=beta_vec,
        indices=current_indices,
        score=score,
        gpr=gpr,
        n_scored=n_scored,
    )


def indices_to_warp_vectors(
    indices: np.ndarray,
    one_dim_warp_pairs: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert library indices into alpha and beta vectors."""

    pairs = [one_dim_warp_pairs[int(idx)] for idx in indices]
    alpha_vec = np.asarray([pair[0] for pair in pairs], dtype=float)
    beta_vec = np.asarray([pair[1] for pair in pairs], dtype=float)
    return alpha_vec, beta_vec


def full_factorized_library_size(n_one_dim_pairs: int, dim: int) -> int:
    """Return the size of the full Cartesian warp library."""

    return int(n_one_dim_pairs) ** int(dim)


def fit_and_score_warp(
    alpha_vec: np.ndarray,
    beta_vec: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    gpr_template,
    prior_weight: float,
    prior_tau: float = 0.75,
) -> tuple[float, object]:
    """
    Fit a cloned GP on warped inputs and return its score and model.

    The score is GP log marginal likelihood plus the weighted unity-warp prior.
    """

    Z = beta_warp_nd(X, alpha_vec, beta_vec)
    gpr = _clone_gpr_template(gpr_template)
    gpr.fit(Z, y)

    lml = float(gpr.log_marginal_likelihood_value_)
    log_prior = log_prior_unity_weak(alpha_vec, beta_vec, tau=prior_tau)
    score = lml + prior_weight * log_prior
    return float(score), gpr


def _score_coordinate_candidate(
    candidate_pair_idx: int,
    coord_idx: int,
    current_indices: np.ndarray,
    one_dim_warp_pairs: list[tuple[float, float]],
    X: np.ndarray,
    y: np.ndarray,
    gpr_template,
    prior_weight: float,
    prior_tau: float,
) -> tuple[int, float]:
    candidate_indices = current_indices.copy()
    candidate_indices[coord_idx] = candidate_pair_idx

    alpha_vec, beta_vec = indices_to_warp_vectors(candidate_indices, one_dim_warp_pairs)
    score, _gpr = fit_and_score_warp(
        alpha_vec,
        beta_vec,
        X,
        y,
        gpr_template,
        prior_weight,
        prior_tau,
    )
    return candidate_pair_idx, score


def _score_coordinate_candidates(
    candidate_pair_indices,
    coord_idx: int,
    current_indices: np.ndarray,
    one_dim_warp_pairs: list[tuple[float, float]],
    X: np.ndarray,
    y: np.ndarray,
    gpr_template,
    prior_weight: float,
    prior_tau: float,
    n_jobs: int,
) -> list[tuple[int, float]]:
    if isinstance(gpr_template, TorchFixedGaussianProcess):
        return _score_coordinate_candidates_torch(
            candidate_pair_indices,
            coord_idx,
            current_indices,
            one_dim_warp_pairs,
            X,
            y,
            gpr_template,
            prior_weight,
            prior_tau,
        )

    if n_jobs == 1:
        return [
            _score_coordinate_candidate(
                candidate_pair_idx,
                coord_idx,
                current_indices,
                one_dim_warp_pairs,
                X,
                y,
                gpr_template,
                prior_weight,
                prior_tau,
            )
            for candidate_pair_idx in candidate_pair_indices
        ]

    try:
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_score_coordinate_candidate)(
                candidate_pair_idx,
                coord_idx,
                current_indices,
                one_dim_warp_pairs,
                X,
                y,
                gpr_template,
                prior_weight,
                prior_tau,
            )
            for candidate_pair_idx in candidate_pair_indices
        )
    except OSError as exc:
        print(f"Parallel warp scoring unavailable ({exc}); falling back to sequential scoring.")
        return _score_coordinate_candidates(
            candidate_pair_indices,
            coord_idx,
            current_indices,
            one_dim_warp_pairs,
            X,
            y,
            gpr_template,
            prior_weight,
            prior_tau,
            n_jobs=1,
        )


def _score_coordinate_candidates_torch(
    candidate_pair_indices,
    coord_idx: int,
    current_indices: np.ndarray,
    one_dim_warp_pairs: list[tuple[float, float]],
    X: np.ndarray,
    y: np.ndarray,
    gpr_template: TorchFixedGaussianProcess,
    prior_weight: float,
    prior_tau: float,
) -> list[tuple[int, float]]:
    candidate_pair_indices = list(candidate_pair_indices)
    if not candidate_pair_indices:
        return []

    alpha_batch = np.empty((len(candidate_pair_indices), X.shape[1]), dtype=float)
    beta_batch = np.empty_like(alpha_batch)
    Z_batch = np.empty((len(candidate_pair_indices), X.shape[0], X.shape[1]), dtype=float)

    for batch_idx, candidate_pair_idx in enumerate(candidate_pair_indices):
        candidate_indices = current_indices.copy()
        candidate_indices[coord_idx] = candidate_pair_idx
        alpha_vec, beta_vec = indices_to_warp_vectors(candidate_indices, one_dim_warp_pairs)
        alpha_batch[batch_idx] = alpha_vec
        beta_batch[batch_idx] = beta_vec
        Z_batch[batch_idx] = beta_warp_nd(X, alpha_vec, beta_vec)

    lml_values = gpr_template.batch_log_marginal_likelihood(Z_batch, y)
    results: list[tuple[int, float]] = []
    for candidate_pair_idx, alpha_vec, beta_vec, lml in zip(
        candidate_pair_indices,
        alpha_batch,
        beta_batch,
        lml_values,
    ):
        log_prior = log_prior_unity_weak(alpha_vec, beta_vec, tau=prior_tau)
        score = float(lml) + prior_weight * log_prior
        results.append((int(candidate_pair_idx), float(score)))
    return results


def _unity_like_pair_index(one_dim_warp_pairs: list[tuple[float, float]]) -> int:
    log_distances = [
        np.log(alpha) ** 2 + np.log(beta) ** 2
        for alpha, beta in one_dim_warp_pairs
    ]
    return int(np.argmin(log_distances))


def _clone_gpr_template(gpr_template):
    if isinstance(gpr_template, TorchFixedGaussianProcess):
        return gpr_template.clone_unfitted()
    return clone(gpr_template)
