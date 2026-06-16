"""Typed search-space utilities for FLIWBO vectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import betainc, betaincinv

from .BO_config import X_DOMAIN_TAU


Number = int | float


@dataclass(frozen=True)
class Discrete:
    """One finite categorical/integer coordinate."""

    num_choices: int

    def __post_init__(self) -> None:
        if int(self.num_choices) <= 0:
            raise ValueError("Discrete variables must have at least one choice")
        object.__setattr__(self, "num_choices", int(self.num_choices))

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "discrete", "num_choices": self.num_choices}


@dataclass(frozen=True)
class Continuous:
    """One bounded continuous coordinate."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not lower < upper:
            raise ValueError(f"Continuous lower bound must be < upper bound, got {lower}, {upper}")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "continuous", "lower": self.lower, "upper": self.upper}


Variable = Discrete | Continuous


class SearchSpace:
    """Product space made of discrete and continuous coordinates."""

    def __init__(self, variables: Sequence[Variable]):
        parsed = tuple(_coerce_variable(variable) for variable in variables)
        if not parsed:
            raise ValueError("SearchSpace must contain at least one coordinate")
        self.variables = parsed

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "SearchSpace":
        if "variables" in data:
            variables = [_variable_from_jsonable(item) for item in data["variables"]]
            return cls(variables)

        raise ValueError("Search-space manifest must contain variables")

    @property
    def dimension(self) -> int:
        return len(self.variables)

    @property
    def discrete_indices(self) -> tuple[int, ...]:
        return tuple(
            idx for idx, variable in enumerate(self.variables)
            if isinstance(variable, Discrete)
        )

    @property
    def continuous_indices(self) -> tuple[int, ...]:
        return tuple(
            idx for idx, variable in enumerate(self.variables)
            if isinstance(variable, Continuous)
        )

    @property
    def is_all_discrete(self) -> bool:
        return len(self.discrete_indices) == self.dimension

    @property
    def array_dtype(self) -> type[np.integer] | type[np.floating]:
        return int if self.is_all_discrete else float

    def to_jsonable(self) -> dict[str, Any]:
        return {"variables": [variable.to_jsonable() for variable in self.variables]}

    def project_matrix(self, X: np.ndarray) -> np.ndarray:
        """Clip continuous values and round/clip discrete values."""

        X_2d = self._as_2d(X)
        projected = np.empty_like(X_2d, dtype=float)

        for idx, variable in enumerate(self.variables):
            values = X_2d[:, idx]
            if isinstance(variable, Discrete):
                upper = float(variable.num_choices - 1)
                projected[:, idx] = np.rint(np.clip(values, 0.0, upper))
            else:
                projected[:, idx] = np.clip(values, variable.lower, variable.upper)

        if self.is_all_discrete:
            return projected.astype(int)
        return projected

    def project_vector(self, x: np.ndarray) -> np.ndarray:
        """Project one vector into the valid search space."""

        return self.project_matrix(np.asarray(x)).ravel()

    def normalize_matrix(self, X: np.ndarray) -> np.ndarray:
        """Map raw vectors into [tau, 1 - tau]^D for GP modeling."""

        _validate_domain_tau()
        projected = self.project_matrix(X).astype(float)
        normalized = np.empty_like(projected, dtype=float)

        for idx, variable in enumerate(self.variables):
            if isinstance(variable, Discrete):
                denominator = max(float(variable.num_choices - 1), 1.0)
                unit_values = projected[:, idx] / denominator
            else:
                unit_values = (projected[:, idx] - variable.lower) / (variable.upper - variable.lower)
            normalized[:, idx] = X_DOMAIN_TAU + (1.0 - 2.0 * X_DOMAIN_TAU) * unit_values

        return normalized

    def normalize_vector(self, x: np.ndarray) -> np.ndarray:
        """Normalize one raw vector and return it as a flat array."""

        return self.normalize_matrix(np.asarray(x)).ravel()

    def unnormalize_matrix(self, X_model: np.ndarray) -> np.ndarray:
        """Map model-domain vectors in [tau, 1 - tau]^D back to raw coordinates."""

        _validate_domain_tau()
        X_2d = self._as_2d(X_model)
        unit_values = (X_2d - X_DOMAIN_TAU) / (1.0 - 2.0 * X_DOMAIN_TAU)
        unit_values = np.clip(unit_values, 0.0, 1.0)
        raw = np.empty_like(unit_values, dtype=float)

        for idx, variable in enumerate(self.variables):
            if isinstance(variable, Discrete):
                raw[:, idx] = unit_values[:, idx] * max(float(variable.num_choices - 1), 1.0)
            else:
                raw[:, idx] = variable.lower + unit_values[:, idx] * (variable.upper - variable.lower)

        return self.project_matrix(raw)

    def unnormalize_vector(self, x_model: np.ndarray) -> np.ndarray:
        """Unnormalize one model-domain vector and return it as a flat array."""

        return self.unnormalize_matrix(np.asarray(x_model)).ravel()

    def vector_to_jsonable(self, x: np.ndarray) -> list[Number]:
        """Convert a projected vector into plain JSON-safe Python numbers."""

        projected = self.project_vector(np.asarray(x))
        values: list[Number] = []
        for idx, variable in enumerate(self.variables):
            if isinstance(variable, Discrete):
                values.append(int(projected[idx]))
            else:
                values.append(float(projected[idx]))
        return values

    def objective_array(self, x: Sequence[Number] | np.ndarray) -> np.ndarray:
        """Return the vector shape used when calling user objective functions."""

        projected = self.project_vector(np.asarray(x, dtype=float))
        if self.is_all_discrete:
            return projected.astype(int)
        return projected.astype(float)

    def pr_dimension_specs(self, alpha: np.ndarray, beta: np.ndarray) -> list[dict[str, Any]]:
        """Return PR-ready warped-domain specs for each coordinate."""

        alpha = np.asarray(alpha, dtype=float).ravel()
        beta = np.asarray(beta, dtype=float).ravel()
        if alpha.shape != (self.dimension,) or beta.shape != (self.dimension,):
            raise ValueError(
                f"Expected alpha/beta shape {(self.dimension,)}, got {alpha.shape}, {beta.shape}"
            )

        specs: list[dict[str, Any]] = []
        for idx, variable in enumerate(self.variables):
            if isinstance(variable, Discrete):
                denominator = max(float(variable.num_choices - 1), 1.0)
                choices = np.arange(variable.num_choices, dtype=float)
                model_values = X_DOMAIN_TAU + (1.0 - 2.0 * X_DOMAIN_TAU) * (choices / denominator)
                warped_choices = betainc(alpha[idx], beta[idx], model_values)
                specs.append({"type": "discrete", "warped_choices": warped_choices})
            else:
                lower = float(betainc(alpha[idx], beta[idx], X_DOMAIN_TAU))
                upper = float(betainc(alpha[idx], beta[idx], 1.0 - X_DOMAIN_TAU))
                specs.append({"type": "continuous", "warped_bounds": (lower, upper)})
        return specs

    def raw_from_pr_vector(
        self,
        x_pr: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Convert a PR vector to raw coordinates.

        Discrete coordinates in x_pr are choice indices. Continuous coordinates
        are optimized in the warped GP-input domain and inverted through the
        selected Beta-CDF warp before raw-domain unnormalization.
        """

        x_pr = np.asarray(x_pr, dtype=float).ravel()
        if x_pr.shape != (self.dimension,):
            raise ValueError(f"Expected PR vector length {self.dimension}, got {x_pr.shape[0]}")

        alpha = np.asarray(alpha, dtype=float).ravel()
        beta = np.asarray(beta, dtype=float).ravel()
        x_model = np.empty(self.dimension, dtype=float)

        for idx, variable in enumerate(self.variables):
            if isinstance(variable, Discrete):
                denominator = max(float(variable.num_choices - 1), 1.0)
                choice = np.rint(np.clip(x_pr[idx], 0.0, float(variable.num_choices - 1)))
                unit_value = choice / denominator
                x_model[idx] = X_DOMAIN_TAU + (1.0 - 2.0 * X_DOMAIN_TAU) * unit_value
            else:
                z_lower = float(betainc(alpha[idx], beta[idx], X_DOMAIN_TAU))
                z_upper = float(betainc(alpha[idx], beta[idx], 1.0 - X_DOMAIN_TAU))
                z_value = float(np.clip(x_pr[idx], z_lower, z_upper))
                x_model[idx] = float(betaincinv(alpha[idx], beta[idx], z_value))

        x_model = np.clip(x_model, X_DOMAIN_TAU, 1.0 - X_DOMAIN_TAU)
        return self.unnormalize_vector(x_model)

    def _as_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be a 1D or 2D array, got shape {X.shape}")
        if X.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of length {self.dimension}, got {X.shape[1]}")
        return X


def _coerce_variable(variable: Variable) -> Variable:
    if isinstance(variable, Discrete | Continuous):
        return variable
    raise TypeError(
        "SearchSpace variables must be Discrete(...) or Continuous(...); "
        f"got {type(variable).__name__}"
    )


def _variable_from_jsonable(data: dict[str, Any]) -> Variable:
    variable_type = data.get("type")
    if variable_type == "discrete":
        return Discrete(int(data["num_choices"]))
    if variable_type == "continuous":
        return Continuous(float(data["lower"]), float(data["upper"]))
    raise ValueError(f"Unknown search-space variable type: {variable_type!r}")


def _validate_domain_tau() -> None:
    if not 0.0 <= X_DOMAIN_TAU < 0.5:
        raise ValueError(f"X_DOMAIN_TAU must be in [0, 0.5), got {X_DOMAIN_TAU}")
