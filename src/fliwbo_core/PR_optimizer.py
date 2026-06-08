"""Probabilistic reparameterization search over discrete vectors.

FLIWBO uses this module to maximize the acquisition function over a product of
categorical choices. It samples integer vectors, estimates which choices look
good, and keeps the best vector found across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.optim as optim


@dataclass(frozen=True)
class PROptimizerConfig:
    """Knobs for acquisition-function optimization over the discrete space."""

    num_restarts: int = 20
    num_steps: int = 30
    num_samples: int = 64
    learning_rate: float = 0.1
    tau_init: float = 1.0
    tau_decay: float = 0.98
    tau_min: float = 0.01


class FactorizedProbabilisticReparameterization:
    """
    REINFORCE-style PR optimizer over a product of categorical variables.

    The acquisition callable receives one hard integer vector and returns a scalar.
    """

    def __init__(
        self,
        choice_sizes: list[int],
        *,
        learning_rate: float = 0.1,
        tau_init: float = 1.0,
        tau_decay: float = 0.98,
        tau_min: float = 0.01,
        seed: int | None = None,
    ):
        if any(size <= 0 for size in choice_sizes):
            raise ValueError("All categorical choice sizes must be positive")

        if seed is not None:
            torch.manual_seed(seed)

        self.choice_sizes = [int(size) for size in choice_sizes]
        self.logits = [
            torch.nn.Parameter(torch.randn(size), requires_grad=True)
            for size in self.choice_sizes
        ]
        self.optimizer = optim.Adam(self.logits, lr=learning_rate)
        self.tau = tau_init
        self.tau_decay = tau_decay
        self.tau_min = tau_min

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        columns = []
        log_prob_columns = []

        for logits in self.logits:
            probabilities = torch.nn.functional.softmax(logits / self.tau, dim=0)
            distribution = torch.distributions.Categorical(probabilities)
            samples = distribution.sample((num_samples,))
            columns.append(samples)
            log_prob_columns.append(distribution.log_prob(samples))

        sample_matrix = torch.stack(columns, dim=1)
        log_probs = torch.stack(log_prob_columns, dim=1).sum(dim=1)
        return sample_matrix, log_probs

    def optimize_acquisition(
        self,
        acquisition: Callable[[np.ndarray], float],
        *,
        num_steps: int,
        num_samples: int,
    ) -> tuple[np.ndarray, float]:
        best_x: np.ndarray | None = None
        best_value = -np.inf

        for _ in range(num_steps):
            self.optimizer.zero_grad()

            samples, log_probs = self.sample(num_samples)
            samples_np = samples.detach().cpu().numpy()
            values_np = np.asarray(
                [float(acquisition(row)) for row in samples_np],
                dtype=np.float32,
            )

            values = torch.tensor(values_np, dtype=torch.float32)
            baseline = values.mean()
            loss = -torch.mean(log_probs * (values - baseline))
            loss.backward()
            self.optimizer.step()

            self.tau = max(self.tau_min, self.tau * self.tau_decay)

            best_idx = int(np.argmax(values_np))
            if float(values_np[best_idx]) > best_value:
                best_value = float(values_np[best_idx])
                best_x = samples_np[best_idx].astype(int)

        if best_x is None:
            raise RuntimeError("PR optimization did not produce any candidate")

        return best_x, best_value


def optimize_with_restarts(
    acquisition: Callable[[np.ndarray], float],
    choice_sizes: list[int],
    config: PROptimizerConfig,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Run several PR optimizers and return the best vector/value pair found."""

    best_x: np.ndarray | None = None
    best_value = -np.inf

    for restart_idx in range(config.num_restarts):
        restart_seed = None if seed is None else seed + restart_idx
        optimizer = FactorizedProbabilisticReparameterization(
            choice_sizes,
            learning_rate=config.learning_rate,
            tau_init=config.tau_init,
            tau_decay=config.tau_decay,
            tau_min=config.tau_min,
            seed=restart_seed,
        )
        candidate_x, candidate_value = optimizer.optimize_acquisition(
            acquisition,
            num_steps=config.num_steps,
            num_samples=config.num_samples,
        )

        if candidate_value > best_value:
            best_value = candidate_value
            best_x = candidate_x

    if best_x is None:
        raise RuntimeError("PR restarts did not produce any candidate")

    return best_x, best_value
