"""Probabilistic reparameterization search over typed vectors.

FLIWBO uses this module to maximize the acquisition function over mixed spaces.
Discrete coordinates are represented by factorized categorical distributions;
continuous coordinates are optimized directly in the warped GP-input domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.optim as optim

from .torch_gp import TorchFixedGaussianProcess, matern25_kernel


@dataclass(frozen=True)
class PROptimizerConfig:
    """Knobs for acquisition-function optimization using PR."""

    num_restarts: int = 20
    num_steps: int = 30
    num_samples: int = 64
    learning_rate: float = 0.1
    tau_init: float = 1.0
    tau_decay: float = 0.98
    tau_min: float = 0.01


@dataclass(frozen=True)
class PRDimension:
    """One coordinate in the PR acquisition optimizer."""

    kind: str
    warped_choices: tuple[float, ...] = ()
    warped_bounds: tuple[float, float] | None = None

    @classmethod
    def discrete(cls, warped_choices: Sequence[float]) -> "PRDimension":
        choices = tuple(float(value) for value in warped_choices)
        if not choices:
            raise ValueError("Discrete PR dimensions must contain at least one choice")
        return cls(kind="discrete", warped_choices=choices)

    @classmethod
    def continuous(cls, warped_bounds: tuple[float, float]) -> "PRDimension":
        lower, upper = (float(warped_bounds[0]), float(warped_bounds[1]))
        if not lower < upper:
            raise ValueError(f"Continuous PR bounds must satisfy lower < upper, got {lower}, {upper}")
        return cls(kind="continuous", warped_bounds=(lower, upper))


class TorchGaussianProcessUCB:
    """Differentiable UCB mirror for the fitted sklearn GP."""

    def __init__(
        self,
        gpr: Any,
        beta_value: float,
        *,
        device: torch.device | str | None = None,
    ):
        if isinstance(gpr, TorchFixedGaussianProcess):
            resolved_device = torch.device(device) if device is not None else gpr.device
            if gpr.X_train_ is None or gpr.alpha_ is None or gpr.L_ is None:
                raise RuntimeError("TorchFixedGaussianProcess must be fit before UCB use")
            self.X_train = gpr.X_train_.detach().to(dtype=torch.float64, device=resolved_device)
            self.alpha = gpr.alpha_.detach().to(dtype=torch.float64, device=resolved_device).reshape(-1)
            self.L = gpr.L_.detach().to(dtype=torch.float64, device=resolved_device)
            self.length_scale = gpr.length_scale.detach().to(
                dtype=torch.float64,
                device=resolved_device,
            )
            self.noise_level = float(gpr.noise_level)
            self.y_train_mean = torch.as_tensor(
                gpr._y_train_mean,
                dtype=torch.float64,
                device=resolved_device,
            ).reshape(-1)[0]
            self.y_train_std = torch.as_tensor(
                gpr._y_train_std,
                dtype=torch.float64,
                device=resolved_device,
            ).reshape(-1)[0]
        else:
            resolved_device = torch.device(device) if device is not None else torch.device("cpu")
            matern_kernel, noise_level = _extract_matern25_and_noise(gpr.kernel_)
            self.X_train = torch.as_tensor(
                gpr.X_train_,
                dtype=torch.float64,
                device=resolved_device,
            )
            self.alpha = torch.as_tensor(
                gpr.alpha_,
                dtype=torch.float64,
                device=resolved_device,
            ).reshape(-1)
            self.L = torch.as_tensor(gpr.L_, dtype=torch.float64, device=resolved_device)
            self.length_scale = torch.as_tensor(
                matern_kernel.length_scale,
                dtype=torch.float64,
                device=resolved_device,
            )
            self.noise_level = float(noise_level)
            self.y_train_mean = torch.as_tensor(
                getattr(gpr, "_y_train_mean", 0.0),
                dtype=torch.float64,
                device=resolved_device,
            ).reshape(-1)[0]
            self.y_train_std = torch.as_tensor(
                getattr(gpr, "_y_train_std", 1.0),
                dtype=torch.float64,
                device=resolved_device,
            ).reshape(-1)[0]

        self.device = resolved_device
        self.beta_value = float(beta_value)

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        Z = Z.to(dtype=torch.float64, device=self.device)

        K_trans = matern25_kernel(Z, self.X_train, self.length_scale)
        mean = K_trans.matmul(self.alpha)

        v = torch.linalg.solve_triangular(self.L, K_trans.T, upper=False)
        variance = 1.0 + self.noise_level - torch.sum(v * v, dim=0)
        variance = torch.clamp(variance, min=0.0)

        mean = self.y_train_std * mean + self.y_train_mean
        std = self.y_train_std * torch.sqrt(variance)
        return mean + np.sqrt(self.beta_value) * std


class FactorizedProbabilisticReparameterization:
    """
    PR optimizer over a product of discrete and continuous variables.

    Discrete variables use REINFORCE-style score-function gradients. Continuous
    variables receive ordinary pathwise gradients through the Torch acquisition.
    """

    def __init__(
        self,
        dimensions: Sequence[PRDimension],
        *,
        learning_rate: float = 0.1,
        tau_init: float = 1.0,
        tau_decay: float = 0.98,
        tau_min: float = 0.01,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ):
        if not dimensions:
            raise ValueError("PR dimensions must contain at least one coordinate")

        if seed is not None:
            torch.manual_seed(seed)

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dimensions = tuple(dimensions)
        invalid_kinds = [
            dimension.kind for dimension in self.dimensions
            if dimension.kind not in {"discrete", "continuous"}
        ]
        if invalid_kinds:
            raise ValueError(f"Unknown PR dimension kind(s): {invalid_kinds}")
        for dimension in self.dimensions:
            if dimension.kind == "discrete" and not dimension.warped_choices:
                raise ValueError("Discrete PR dimensions must contain at least one choice")
            if dimension.kind == "continuous" and dimension.warped_bounds is None:
                raise ValueError("Continuous PR dimensions must include warped_bounds")

        self.dimension = len(self.dimensions)
        self.discrete_positions = [
            idx for idx, dimension in enumerate(self.dimensions)
            if dimension.kind == "discrete"
        ]
        self.continuous_positions = [
            idx for idx, dimension in enumerate(self.dimensions)
            if dimension.kind == "continuous"
        ]
        self.logits = [
            torch.nn.Parameter(
                torch.randn(
                    len(self.dimensions[idx].warped_choices),
                    dtype=torch.float64,
                    device=self.device,
                )
            )
            for idx in self.discrete_positions
        ]
        self.discrete_warped_choices = [
            torch.as_tensor(
                self.dimensions[idx].warped_choices,
                dtype=torch.float64,
                device=self.device,
            )
            for idx in self.discrete_positions
        ]

        self.continuous_lower: torch.Tensor | None = None
        self.continuous_upper: torch.Tensor | None = None
        self.continuous_unconstrained: torch.nn.Parameter | None = None
        if self.continuous_positions:
            bounds = [
                self.dimensions[idx].warped_bounds
                for idx in self.continuous_positions
            ]
            lower = torch.tensor(
                [bound[0] for bound in bounds if bound is not None],
                dtype=torch.float64,
                device=self.device,
            )
            upper = torch.tensor(
                [bound[1] for bound in bounds if bound is not None],
                dtype=torch.float64,
                device=self.device,
            )
            init_unit = torch.rand(
                len(self.continuous_positions),
                dtype=torch.float64,
                device=self.device,
            ).clamp(1e-6, 1.0 - 1e-6)
            self.continuous_lower = lower
            self.continuous_upper = upper
            self.continuous_unconstrained = torch.nn.Parameter(torch.logit(init_unit))

        parameters: list[torch.nn.Parameter] = list(self.logits)
        if self.continuous_unconstrained is not None:
            parameters.append(self.continuous_unconstrained)
        self.optimizer = optim.Adam(parameters, lr=learning_rate)
        self.tau = tau_init
        self.tau_decay = tau_decay
        self.tau_min = tau_min

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.empty(
            (num_samples, self.dimension),
            dtype=torch.float64,
            device=self.device,
        )
        x_pr = torch.empty(
            (num_samples, self.dimension),
            dtype=torch.float64,
            device=self.device,
        )
        log_probs = torch.zeros(num_samples, dtype=torch.float64, device=self.device)

        for logits, warped_choices, dim_idx in zip(
            self.logits,
            self.discrete_warped_choices,
            self.discrete_positions,
        ):
            probabilities = torch.nn.functional.softmax(logits / self.tau, dim=0)
            distribution = torch.distributions.Categorical(probabilities)
            samples = distribution.sample((num_samples,))
            z[:, dim_idx] = warped_choices[samples]
            x_pr[:, dim_idx] = samples.to(dtype=torch.float64)
            log_probs = log_probs + distribution.log_prob(samples)

        continuous_z = self._continuous_z()
        if continuous_z is not None:
            for local_idx, dim_idx in enumerate(self.continuous_positions):
                z[:, dim_idx] = continuous_z[local_idx]
                x_pr[:, dim_idx] = continuous_z[local_idx]

        return z, x_pr, log_probs

    def modal_candidate(self) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.empty(self.dimension, dtype=torch.float64, device=self.device)
        x_pr = torch.empty(self.dimension, dtype=torch.float64, device=self.device)

        for logits, warped_choices, dim_idx in zip(
            self.logits,
            self.discrete_warped_choices,
            self.discrete_positions,
        ):
            choice_idx = int(torch.argmax(logits).item())
            z[dim_idx] = warped_choices[choice_idx]
            x_pr[dim_idx] = float(choice_idx)

        continuous_z = self._continuous_z()
        if continuous_z is not None:
            for local_idx, dim_idx in enumerate(self.continuous_positions):
                z[dim_idx] = continuous_z[local_idx]
                x_pr[dim_idx] = continuous_z[local_idx]

        return z, x_pr

    def optimize_acquisition(
        self,
        acquisition: Callable[[torch.Tensor], torch.Tensor],
        *,
        num_steps: int,
        num_samples: int,
    ) -> tuple[np.ndarray, float]:
        best_x: np.ndarray | None = None
        best_value = -np.inf

        for _ in range(num_steps):
            self.optimizer.zero_grad()

            z_samples, x_pr_samples, log_probs = self.sample(num_samples)
            values = acquisition(z_samples)
            if values.ndim != 1 or values.shape[0] != num_samples:
                raise ValueError(
                    f"Acquisition must return shape {(num_samples,)}, got {tuple(values.shape)}"
                )

            baseline = values.detach().mean()
            score_term = torch.mean(log_probs * (values.detach() - baseline))
            loss = -values.mean() - score_term
            loss.backward()
            self.optimizer.step()

            self.tau = max(self.tau_min, self.tau * self.tau_decay)

            step_best_value, step_best_idx = torch.max(values.detach(), dim=0)
            step_best_float = float(step_best_value.cpu().item())
            if step_best_float > best_value:
                best_value = step_best_float
                best_x = x_pr_samples.detach().cpu().numpy()[int(step_best_idx.cpu().item())]

        modal_z, modal_x_pr = self.modal_candidate()
        modal_value = float(acquisition(modal_z.reshape(1, -1)).detach().cpu().numpy()[0])
        if modal_value > best_value:
            best_value = modal_value
            best_x = modal_x_pr.detach().cpu().numpy()

        if best_x is None:
            raise RuntimeError("PR optimization did not produce any candidate")

        return best_x, best_value

    def _continuous_z(self) -> torch.Tensor | None:
        if self.continuous_unconstrained is None:
            return None
        if self.continuous_lower is None or self.continuous_upper is None:
            raise RuntimeError("Continuous PR state is incomplete")
        unit_values = torch.sigmoid(self.continuous_unconstrained)
        return self.continuous_lower + (self.continuous_upper - self.continuous_lower) * unit_values


def optimize_with_restarts(
    acquisition: Callable[[torch.Tensor], torch.Tensor],
    dimensions: Sequence[PRDimension] | Sequence[dict[str, Any]],
    config: PROptimizerConfig,
    *,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, float]:
    """Run several PR optimizers and return the best vector/value pair found."""

    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    pr_dimensions = _coerce_pr_dimensions(dimensions)
    batch_acquisition = _coerce_batch_acquisition(acquisition, device=resolved_device)
    best_x: np.ndarray | None = None
    best_value = -np.inf

    for restart_idx in range(config.num_restarts):
        restart_seed = None if seed is None else seed + restart_idx
        optimizer = FactorizedProbabilisticReparameterization(
            pr_dimensions,
            learning_rate=config.learning_rate,
            tau_init=config.tau_init,
            tau_decay=config.tau_decay,
            tau_min=config.tau_min,
            seed=restart_seed,
            device=resolved_device,
        )
        candidate_x, candidate_value = optimizer.optimize_acquisition(
            batch_acquisition,
            num_steps=config.num_steps,
            num_samples=config.num_samples,
        )

        if candidate_value > best_value:
            best_value = candidate_value
            best_x = candidate_x

    if best_x is None:
        raise RuntimeError("PR restarts did not produce any candidate")

    if all(dimension.kind == "discrete" for dimension in pr_dimensions):
        best_x = np.rint(best_x).astype(int)
    return best_x, best_value


def _coerce_pr_dimensions(
    dimensions: Sequence[PRDimension] | Sequence[dict[str, Any]],
) -> list[PRDimension]:
    coerced: list[PRDimension] = []
    for dimension in dimensions:
        if isinstance(dimension, PRDimension):
            coerced.append(dimension)
        elif isinstance(dimension, dict):
            if dimension.get("type") == "discrete":
                coerced.append(PRDimension.discrete(dimension["warped_choices"]))
            elif dimension.get("type") == "continuous":
                coerced.append(PRDimension.continuous(tuple(dimension["warped_bounds"])))
            else:
                raise ValueError(f"Unknown PR dimension type: {dimension.get('type')!r}")
        else:
            raise TypeError(f"Expected PRDimension or dict PR spec, got {type(dimension).__name__}")
    return coerced


def _coerce_batch_acquisition(
    acquisition: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def batch_acquisition(z: torch.Tensor) -> torch.Tensor:
        values = acquisition(z)
        if not isinstance(values, torch.Tensor):
            return torch.as_tensor(values, dtype=torch.float64, device=device)
        return values.to(dtype=torch.float64, device=device)

    return batch_acquisition


def _extract_matern25_and_noise(kernel: Any) -> tuple[Any, float]:
    matern_kernel = None
    noise_level = 0.0
    stack = [kernel]

    while stack:
        current = stack.pop()
        name = type(current).__name__
        if name == "Matern":
            matern_kernel = current
        elif name == "WhiteKernel":
            noise_level += float(current.noise_level)
        else:
            if hasattr(current, "k1") and hasattr(current, "k2"):
                stack.extend([current.k1, current.k2])

    if matern_kernel is None:
        raise ValueError("Torch GP mirror requires a fitted sklearn Matern kernel")
    if not np.isclose(float(matern_kernel.nu), 2.5):
        raise ValueError(f"Torch GP mirror only supports Matern nu=2.5, got {matern_kernel.nu}")
    return matern_kernel, noise_level
