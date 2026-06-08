"""Helpers for integer-vector search spaces.

The optimizer works with discrete integer vectors, while the Gaussian process
expects continuous inputs in a bounded domain. These helpers clip integer
vectors to valid bounds and map them into the unit interval.
"""

from __future__ import annotations

import numpy as np

from .BO_config import X_DOMAIN_TAU


def normalize_discrete_matrix(X: np.ndarray, choice_sizes: list[int]) -> np.ndarray:
    """Map integer design vectors into [tau, 1 - tau]^D for GP modeling."""

    if not 0.0 <= X_DOMAIN_TAU < 0.5:
        raise ValueError(f"X_DOMAIN_TAU must be in [0, 0.5), got {X_DOMAIN_TAU}")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    choice_sizes_arr = np.asarray(choice_sizes, dtype=float)
    if X.shape[1] != choice_sizes_arr.shape[0]:
        raise ValueError(
            f"Expected vectors of length {choice_sizes_arr.shape[0]}, got {X.shape[1]}"
        )

    denominators = np.maximum(choice_sizes_arr - 1.0, 1.0)
    clipped = clip_discrete_matrix(X, choice_sizes)
    unit_domain = clipped / denominators
    return X_DOMAIN_TAU + (1.0 - 2.0 * X_DOMAIN_TAU) * unit_domain


def normalize_discrete_vector(x: np.ndarray, choice_sizes: list[int]) -> np.ndarray:
    """Normalize one integer vector and return it as a flat array."""

    return normalize_discrete_matrix(np.asarray(x), choice_sizes).ravel()


def clip_discrete_matrix(X: np.ndarray, choice_sizes: list[int]) -> np.ndarray:
    """Round and clip vectors so every coordinate is a valid discrete choice."""

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    upper = np.asarray(choice_sizes, dtype=float) - 1.0
    if X.shape[1] != upper.shape[0]:
        raise ValueError(f"Expected vectors of length {upper.shape[0]}, got {X.shape[1]}")

    return np.rint(np.clip(X, 0.0, upper)).astype(int)


def discrete_vector_to_jsonable(x: np.ndarray) -> list[int]:
    """Convert a numpy vector into plain Python ints for JSON/CSV storage."""

    return [int(v) for v in np.asarray(x).ravel()]
