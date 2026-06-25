"""Public optimizer API for finite-library input-warped Bayesian optimization.

This module contains the package surface most users should touch:

- FLIWBOOptimizer for simple runs and durable ask/tell runs.
- SearchSpace for declaring typed vector bounds.
- FLIWBOConfig and PROptimizerConfig for optimizer knobs.
- OptimizationRun, OptimizationProposal, and OptimizationResult for orchestration.

The optimizer never builds or evaluates a real system. It proposes typed
vectors. User code evaluates those vectors and returns scalar scores.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .BO_config import (
    BETA_SCALING,
    EPSILON_WARP,
    LENGTHSCALE,
    N_ITERS,
    OBS_NOISE,
    PR_LEARNING_RATE,
    PR_NUM_RESTARTS,
    PR_NUM_SAMPLES,
    PR_NUM_STEPS,
    PR_TAU_DECAY,
    PR_TAU_INIT,
    PR_TAU_MIN,
    USE_WARP_PRIOR,
    WARP_UNITY_PRIOR_TAU,
    WARP_UNITY_PRIOR_WEIGHT,
    WARP_SEARCH_N_JOBS,
    WARP_SEARCH_SWEEPS,
    Y_CENTER,
    Y_SCALE,
)
from .BO_utils import beta_t as default_beta_fn
from .BO_utils import beta_warp_nd, make_warp_library
from .PR_optimizer import PROptimizerConfig, TorchGaussianProcessUCB, optimize_with_restarts
from .search_space import Number, SearchSpace
from .warp_optimizer import full_factorized_library_size, optimize_warp_coordinatewise


ObjectiveFunction = Callable[[np.ndarray], float]
BetaSchedule = Callable[..., float]


def default_pr_config() -> PROptimizerConfig:
    return PROptimizerConfig(
        num_restarts=PR_NUM_RESTARTS,
        num_steps=PR_NUM_STEPS,
        num_samples=PR_NUM_SAMPLES,
        learning_rate=PR_LEARNING_RATE,
        tau_init=PR_TAU_INIT,
        tau_decay=PR_TAU_DECAY,
        tau_min=PR_TAU_MIN,
    )


@dataclass(frozen=True)
class FLIWBOConfig:
    """Configuration for one optimizer instance.

    The defaults are tuned for the current QuixBugs-style experiments, but the
    object is deliberately domain-neutral. Users can pass different values for
    synthetic tests, cheaper smoke runs, or high-value production runs.

    The warp prior is centered on the unity warp alpha=beta=1. Increase
    warp_prior_weight to resist aggressive warps, or increase warp_prior_tau to
    make the prior wider and easier to move away from unity. For the current
    quadratic log-prior penalty, the effective strength is proportional to
    warp_prior_weight / warp_prior_tau**2.
    """

    n_iters: int = N_ITERS
    noise_std: float = OBS_NOISE
    lengthscale: float = LENGTHSCALE
    use_warp_prior: bool = USE_WARP_PRIOR
    y_center: float = Y_CENTER
    y_scale: float = Y_SCALE
    epsilon_warp: float = EPSILON_WARP
    beta_scaling: float = BETA_SCALING
    warp_search_sweeps: int = WARP_SEARCH_SWEEPS
    warp_search_n_jobs: int = WARP_SEARCH_N_JOBS
    pr_config: PROptimizerConfig = field(default_factory=default_pr_config)
    pr_seed: int | None = None
    log_csv: bool = False
    metadata_dir: str | Path | None = None
    verbose: bool = False
    warp_prior_weight: float = WARP_UNITY_PRIOR_WEIGHT
    warp_prior_tau: float = WARP_UNITY_PRIOR_TAU


@dataclass(frozen=True)
class OptimizationProposal:
    """A vector proposed by ask() before the expensive objective is evaluated."""

    iteration: int
    x_vector: list[Number]
    acquisition_value: float
    warp_alpha: list[float]
    warp_beta: list[float]
    warp_indices: list[int]
    warp_score: float
    warp_search_scored: int


@dataclass(frozen=True)
class BOIterationRecord:
    """A completed proposal plus the objective value returned by user code."""

    iteration: int
    x_vector: list[Number]
    y_value: float
    acquisition_value: float
    warp_alpha: list[float]
    warp_beta: list[float]
    warp_indices: list[int]
    warp_score: float
    warp_search_scored: int


@dataclass(frozen=True)
class OptimizationResult:
    """Convenient in-memory view of an optimization run.

    This is useful for downstream code after a run finishes. For crash recovery,
    the durable run directory is the source of truth.
    """

    initial_x: np.ndarray
    initial_y: np.ndarray
    x_observed: np.ndarray
    y_observed: np.ndarray
    iterations: tuple[BOIterationRecord, ...]
    run_dir: Path | None = None
    metadata_path: Path | None = None

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.y_observed))

    @property
    def best_x(self) -> np.ndarray:
        return self.x_observed[self.best_index].copy()

    @property
    def best_y(self) -> float:
        return float(self.y_observed[self.best_index])


class FLIWBOOptimizer:
    """Finite-library input-warped Bayesian optimizer over typed search spaces."""

    def __init__(
        self,
        search_space: SearchSpace,
        config: FLIWBOConfig | None = None,
        *,
        beta_fn: BetaSchedule = default_beta_fn,
    ):
        if not isinstance(search_space, SearchSpace):
            raise TypeError("search_space must be a SearchSpace instance")
        self.search_space = search_space

        self.config = config or FLIWBOConfig()
        self.beta_fn = beta_fn
        self._validate_config()

    def start(
        self,
        X_init: np.ndarray,
        y_init: np.ndarray,
        *,
        run_dir: str | Path | None = None,
    ) -> "OptimizationRun":
        """Start a new optimization run and optionally persist it under run_dir."""
        resolved_run_dir = self._resolve_run_dir(run_dir)
        if resolved_run_dir is not None:
            self._log(f"Keeping BO metadata in: {resolved_run_dir}")
        return OptimizationRun.start(self, X_init, y_init, run_dir=resolved_run_dir)

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        beta_fn: BetaSchedule = default_beta_fn,
    ) -> "OptimizationRun":
        """Resume a durable run created by start(..., run_dir=...) or run(..., run_dir=...)."""
        manifest = _read_json(Path(run_dir) / _MANIFEST_FILE)
        search_space = SearchSpace.from_jsonable(manifest["search_space"])
        config = _config_from_jsonable(manifest["config"])
        optimizer = cls(search_space, config=config, beta_fn=beta_fn)
        return OptimizationRun.resume(optimizer, run_dir)

    def run(
        self,
        objective_fn: ObjectiveFunction,
        X_init: np.ndarray,
        y_init: np.ndarray,
        *,
        run_dir: str | Path | None = None,
    ) -> OptimizationResult:
        """
        Convenience API for objective_fn(x_vector) -> float.

        If run_dir is provided, or if config.log_csv is true, every proposal and
        completed observation is persisted before the final OptimizationResult is returned.
        """
        run = self.start(X_init, y_init, run_dir=run_dir)

        while not run.is_complete:
            proposal = run.ask()
            try:
                y_next = float(objective_fn(self.search_space.objective_array(proposal.x_vector)))
            except Exception as exc:
                run.record_objective_error(proposal, exc)
                raise
            run.tell(proposal, y_next)

        return run.result()

    def _compute_next_proposal(
        self,
        X_observed: np.ndarray,
        y_raw: np.ndarray,
        iteration: int,
    ) -> OptimizationProposal:
        """Fit the current BO model and choose the next typed vector."""

        X_projected = self.search_space.project_matrix(X_observed)
        X = self.search_space.normalize_matrix(X_projected)
        y = self._normalize_y(y_raw)
        noise_var = (self.config.noise_std / self.config.y_scale) ** 2

        kernel = Matern(length_scale=self.config.lengthscale, nu=2.5) + WhiteKernel(
            noise_level=noise_var
        )
        gpr_template = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,
            normalize_y=False,
        )

        one_dim_warp_pairs = make_warp_library(epsilon=self.config.epsilon_warp)
        n_full_warp_library = full_factorized_library_size(
            len(one_dim_warp_pairs),
            self.search_space.dimension,
        )

        self._log(f"N_eps per coordinate: {len(one_dim_warp_pairs)}")
        self._log(f"N_eps full factorized library: {n_full_warp_library}")
        self._log(f"BO iteration {iteration}/{self.config.n_iters}")

        beta_value = self._beta_value(iteration, n_full_warp_library)
        prior_weight = self.config.warp_prior_weight if self.config.use_warp_prior else 0.0

        warp_result = optimize_warp_coordinatewise(
            X=X,
            y=y,
            gpr_template=gpr_template,
            one_dim_warp_pairs=one_dim_warp_pairs,
            prior_weight=prior_weight,
            prior_tau=self.config.warp_prior_tau,
            n_sweeps=self.config.warp_search_sweeps,
            n_jobs=self.config.warp_search_n_jobs,
        )

        alpha_chosen = warp_result.alpha
        beta_chosen = warp_result.beta
        winning_gpr = warp_result.gpr

        torch_acquisition = TorchGaussianProcessUCB(winning_gpr, beta_value)
        pr_dimensions = self.search_space.pr_dimension_specs(alpha_chosen, beta_chosen)

        x_next_pr, _pr_acquisition_value = optimize_with_restarts(
            torch_acquisition,
            pr_dimensions,
            self.config.pr_config,
            seed=self.config.pr_seed,
        )
        x_next_raw = self.search_space.raw_from_pr_vector(
            x_next_pr,
            alpha_chosen,
            beta_chosen,
        )
        acquisition_value = self._sklearn_ucb_value(
            winning_gpr,
            x_next_raw,
            alpha_chosen,
            beta_chosen,
            beta_value,
        )

        return OptimizationProposal(
            iteration=iteration,
            x_vector=self.search_space.vector_to_jsonable(x_next_raw),
            acquisition_value=float(acquisition_value),
            warp_alpha=[float(value) for value in alpha_chosen.tolist()],
            warp_beta=[float(value) for value in beta_chosen.tolist()],
            warp_indices=[int(idx) for idx in warp_result.indices.tolist()],
            warp_score=float(warp_result.score),
            warp_search_scored=int(warp_result.n_scored),
        )

    def _coerce_x_init(self, X_init: np.ndarray) -> np.ndarray:
        X = np.asarray(X_init, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(f"X_init must be a 1D or 2D array, got shape {X.shape}")
        if X.shape[1] != self.search_space.dimension:
            raise ValueError(
                f"Expected X vectors of length {self.search_space.dimension}, "
                f"got {X.shape[1]}"
            )
        return self.search_space.project_matrix(X)

    def _sklearn_ucb_value(
        self,
        gpr: GaussianProcessRegressor,
        x_raw: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        beta_value: float,
    ) -> float:
        x_model = self.search_space.normalize_matrix(np.asarray(x_raw).reshape(1, -1))
        z = beta_warp_nd(x_model, alpha, beta)
        mu, std = gpr.predict(z, return_std=True)
        return float(mu[0] + np.sqrt(beta_value) * std[0])

    def _normalize_y(self, y_raw: np.ndarray) -> np.ndarray:
        return (np.asarray(y_raw, dtype=float).ravel() - self.config.y_center) / self.config.y_scale

    def _beta_value(self, iteration: int, n_full_warp_library: int) -> float:
        if self.beta_fn is default_beta_fn:
            return float(
                self.beta_fn(
                    iteration,
                    N_eps=n_full_warp_library,
                    beta_scaling=self.config.beta_scaling,
                    dim=self.search_space.dimension,
                )
            )
        return float(self.beta_fn(iteration, N_eps=n_full_warp_library))

    def _validate_config(self) -> None:
        if self.config.n_iters < 0:
            raise ValueError(f"n_iters must be non-negative, got {self.config.n_iters}")
        if self.config.y_scale <= 0.0:
            raise ValueError(f"y_scale must be positive, got {self.config.y_scale}")
        if self.config.epsilon_warp <= 0.0:
            raise ValueError(f"epsilon_warp must be positive, got {self.config.epsilon_warp}")
        if self.config.beta_scaling <= 0.0:
            raise ValueError(f"beta_scaling must be positive, got {self.config.beta_scaling}")
        if self.config.warp_prior_weight < 0.0:
            raise ValueError(
                f"warp_prior_weight must be non-negative, got {self.config.warp_prior_weight}"
            )
        if self.config.warp_prior_tau <= 0.0:
            raise ValueError(f"warp_prior_tau must be positive, got {self.config.warp_prior_tau}")

    def _resolve_run_dir(self, run_dir: str | Path | None) -> Path | None:
        if run_dir is not None:
            return Path(run_dir)

        if not self.config.log_csv:
            return None

        metadata_dir = Path(self.config.metadata_dir or "BO metadata")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return metadata_dir / f"run_{timestamp}"

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message)


class OptimizationRun:
    """Stateful ask/tell optimization run with optional crash-resistant journaling.

    ask() proposes the next vector and writes it to disk before returning.
    tell() accepts the score for that vector and writes the completed observation.
    resume() reconstructs this object from the durable run directory.
    """

    def __init__(
        self,
        optimizer: FLIWBOOptimizer,
        *,
        initial_x: np.ndarray,
        initial_y: np.ndarray,
        x_observed: np.ndarray,
        y_observed: np.ndarray,
        records: list[BOIterationRecord],
        pending_proposals: list[OptimizationProposal],
        run_dir: Path | None,
    ):
        self.optimizer = optimizer
        self.initial_x = initial_x.copy()
        self.initial_y = initial_y.copy()
        self.x_observed = x_observed.copy()
        self.y_observed = y_observed.copy()
        self.records = list(records)
        self._pending_proposals = list(pending_proposals)
        self.run_dir = run_dir
        self.journal = _RunJournal(run_dir) if run_dir is not None else None

    @classmethod
    def start(
        cls,
        optimizer: FLIWBOOptimizer,
        X_init: np.ndarray,
        y_init: np.ndarray,
        *,
        run_dir: Path | None,
    ) -> "OptimizationRun":
        X_projected = optimizer._coerce_x_init(X_init)
        y_raw = np.asarray(y_init, dtype=float).ravel()

        if y_raw.shape[0] != X_projected.shape[0]:
            raise ValueError(
                f"X_init contains {X_projected.shape[0]} rows, "
                f"but y_init contains {y_raw.shape[0]} values"
            )

        if run_dir is not None:
            journal = _RunJournal(run_dir)
            journal.initialize(optimizer, X_projected, y_raw)

        return cls(
            optimizer,
            initial_x=X_projected,
            initial_y=y_raw,
            x_observed=X_projected,
            y_observed=y_raw,
            records=[],
            pending_proposals=[],
            run_dir=run_dir,
        )

    @classmethod
    def resume(
        cls,
        optimizer: FLIWBOOptimizer,
        run_dir: str | Path,
    ) -> "OptimizationRun":
        journal = _RunJournal(Path(run_dir))
        state = journal.load()

        return cls(
            optimizer,
            initial_x=state["initial_x"],
            initial_y=state["initial_y"],
            x_observed=state["x_observed"],
            y_observed=state["y_observed"],
            records=state["records"],
            pending_proposals=state["pending_proposals"],
            run_dir=Path(run_dir),
        )

    @property
    def completed_iterations(self) -> int:
        return len(self.records)

    @property
    def is_complete(self) -> bool:
        return self.completed_iterations >= self.optimizer.config.n_iters

    @property
    def pending_proposal(self) -> OptimizationProposal | None:
        if not self._pending_proposals:
            return None
        if len(self._pending_proposals) > 1:
            raise RuntimeError("Run contains more than one pending proposal")
        return self._pending_proposals[0]

    def ask(self) -> OptimizationProposal:
        """
        Return the next proposal and persist it before returning.

        If the run already has a pending proposal, ask() returns that proposal
        instead of generating a new one. This is what makes resume-after-crash safe.
        """
        if self.is_complete:
            raise RuntimeError("Optimization run is already complete")

        pending = self.pending_proposal
        if pending is not None:
            return pending

        iteration = self.completed_iterations + 1
        proposal = self.optimizer._compute_next_proposal(
            self.x_observed,
            self.y_observed,
            iteration,
        )

        if self.journal is not None:
            self.journal.write_proposal(proposal)

        self._pending_proposals.append(proposal)
        return proposal

    def tell(
        self,
        proposal: OptimizationProposal | int,
        y_value: float,
    ) -> BOIterationRecord:
        """Complete a pending proposal with its observed objective value."""
        pending = self._resolve_pending_proposal(proposal)
        y_float = float(y_value)

        record = BOIterationRecord(
            iteration=pending.iteration,
            x_vector=list(pending.x_vector),
            y_value=y_float,
            acquisition_value=pending.acquisition_value,
            warp_alpha=list(pending.warp_alpha),
            warp_beta=list(pending.warp_beta),
            warp_indices=list(pending.warp_indices),
            warp_score=pending.warp_score,
            warp_search_scored=pending.warp_search_scored,
        )

        if self.journal is not None:
            self.journal.write_completed_observation(record)

        self.records.append(record)
        self._pending_proposals = [
            item for item in self._pending_proposals
            if item.iteration != pending.iteration
        ]
        next_x = self.optimizer.search_space.objective_array(record.x_vector)
        self.x_observed = np.vstack([self.x_observed, next_x])
        self.y_observed = np.concatenate([self.y_observed, [record.y_value]])
        return record

    def record_objective_error(self, proposal: OptimizationProposal, exc: Exception) -> None:
        if self.journal is None:
            return
        self.journal.write_event(
            "objective_error",
            {
                "iteration": proposal.iteration,
                "x_vector": proposal.x_vector,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    def result(self) -> OptimizationResult:
        metadata_path = None
        if self.run_dir is not None:
            metadata_path = self.run_dir / _PROPOSALS_FILE

        return OptimizationResult(
            initial_x=self.initial_x.copy(),
            initial_y=self.initial_y.copy(),
            x_observed=self.x_observed.copy(),
            y_observed=self.y_observed.copy(),
            iterations=tuple(self.records),
            run_dir=self.run_dir,
            metadata_path=metadata_path,
        )

    def _resolve_pending_proposal(
        self,
        proposal: OptimizationProposal | int,
    ) -> OptimizationProposal:
        iteration = proposal.iteration if isinstance(proposal, OptimizationProposal) else int(proposal)

        matches = [item for item in self._pending_proposals if item.iteration == iteration]
        if not matches:
            raise ValueError(f"No pending proposal for iteration {iteration}")

        pending = matches[0]
        if isinstance(proposal, OptimizationProposal) and proposal.x_vector != pending.x_vector:
            raise ValueError(f"Proposal for iteration {iteration} does not match pending x_vector")
        return pending


class _RunJournal:
    """Durable on-disk journal used by OptimizationRun.

    The journal writes files atomically where practical and appends an event log
    for human inspection. It is private because callers should interact through
    OptimizationRun rather than manipulating files directly.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.manifest_path = run_dir / _MANIFEST_FILE
        self.observations_path = run_dir / _OBSERVATIONS_FILE
        self.proposals_path = run_dir / _PROPOSALS_FILE
        self.events_path = run_dir / _EVENTS_FILE

    def initialize(
        self,
        optimizer: FLIWBOOptimizer,
        X_init: np.ndarray,
        y_init: np.ndarray,
    ) -> None:
        """Create a new durable run directory and write initial observations."""

        if self.manifest_path.exists():
            raise FileExistsError(
                f"Run directory already contains {_MANIFEST_FILE}: {self.run_dir}. "
                "Use FLIWBOOptimizer.resume(...) to continue it."
            )

        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "search_space": optimizer.search_space.to_jsonable(),
            "config": _config_to_jsonable(optimizer.config),
            "files": {
                "observations": _OBSERVATIONS_FILE,
                "proposals": _PROPOSALS_FILE,
                "events": _EVENTS_FILE,
            },
        }
        _write_json_atomic(self.manifest_path, manifest)

        observation_rows = [
            {
                "source": "initial",
                "iteration": 0,
                "x_vector": json.dumps(optimizer.search_space.vector_to_jsonable(row)),
                "y_value": float(y_value),
            }
            for row, y_value in zip(X_init, y_init)
        ]
        _write_csv_atomic(self.observations_path, _OBSERVATION_FIELDS, observation_rows)
        _write_csv_atomic(self.proposals_path, _PROPOSAL_FIELDS, [])
        self.write_event(
            "run_started",
            {
                "n_initial_observations": int(X_init.shape[0]),
                "dimension": int(X_init.shape[1]),
            },
        )

    def load(self) -> dict[str, Any]:
        """Load completed and pending run state from the durable files."""

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"No {_MANIFEST_FILE} found in run directory: {self.run_dir}")

        manifest = _read_json(self.manifest_path)
        search_space = SearchSpace.from_jsonable(manifest["search_space"])
        observation_rows = _read_csv_rows(self.observations_path)
        proposal_rows = _read_csv_rows(self.proposals_path)

        initial_x_rows: list[list[Number]] = []
        initial_y_values: list[float] = []
        completed_y_by_iteration: dict[int, float] = {}

        for row in observation_rows:
            source = row.get("source")
            iteration = int(row["iteration"])
            x_vector = _json_vector(row["x_vector"])
            y_value = float(row["y_value"])

            if source == "initial":
                initial_x_rows.append(x_vector)
                initial_y_values.append(y_value)
            elif source == "completed":
                completed_y_by_iteration[iteration] = y_value

        records: list[BOIterationRecord] = []
        pending_proposals: list[OptimizationProposal] = []

        for row in proposal_rows:
            proposal = _proposal_from_row(row)
            y_value = completed_y_by_iteration.get(proposal.iteration)
            if y_value is None and row.get("status") == "completed" and row.get("y_value"):
                y_value = float(row["y_value"])

            if y_value is None:
                if row.get("status") == "pending":
                    pending_proposals.append(proposal)
                continue

            records.append(_record_from_proposal(proposal, y_value))

        records.sort(key=lambda record: record.iteration)
        _validate_contiguous_iterations(records)

        initial_x = search_space.project_matrix(np.asarray(initial_x_rows, dtype=float))
        initial_y = np.asarray(initial_y_values, dtype=float)
        completed_rows = [record.x_vector for record in records]
        completed_x = (
            search_space.project_matrix(np.asarray(completed_rows, dtype=float))
            if completed_rows
            else np.empty((0, search_space.dimension), dtype=search_space.array_dtype)
        )
        completed_y = np.asarray([record.y_value for record in records], dtype=float)

        if completed_x.size == 0:
            x_observed = initial_x.copy()
        else:
            x_observed = np.vstack([initial_x, completed_x])
        y_observed = np.concatenate([initial_y, completed_y])

        return {
            "initial_x": initial_x,
            "initial_y": initial_y,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "records": records,
            "pending_proposals": pending_proposals,
        }

    def write_proposal(self, proposal: OptimizationProposal) -> None:
        """Persist a pending proposal before it leaves the optimizer."""

        rows = _read_csv_rows(self.proposals_path)
        if any(int(row["iteration"]) == proposal.iteration for row in rows):
            raise ValueError(f"Proposal for iteration {proposal.iteration} already exists")

        rows.append(_proposal_to_row(proposal, status="pending"))
        _write_csv_atomic(self.proposals_path, _PROPOSAL_FIELDS, rows)
        self.write_event("proposal_created", asdict(proposal))

    def write_completed_observation(self, record: BOIterationRecord) -> None:
        """Persist a completed objective value and mark its proposal completed."""

        observation_rows = _read_csv_rows(self.observations_path)
        if any(
            row.get("source") == "completed" and int(row["iteration"]) == record.iteration
            for row in observation_rows
        ):
            raise ValueError(f"Completed observation for iteration {record.iteration} already exists")

        observation_rows.append(
            {
                "source": "completed",
                "iteration": record.iteration,
                "x_vector": json.dumps(record.x_vector),
                "y_value": record.y_value,
            }
        )
        _write_csv_atomic(self.observations_path, _OBSERVATION_FIELDS, observation_rows)

        proposal_rows = _read_csv_rows(self.proposals_path)
        updated_rows = []
        found = False
        for row in proposal_rows:
            if int(row["iteration"]) == record.iteration:
                row = dict(row)
                row["status"] = "completed"
                row["completed_at_utc"] = _utc_now()
                row["y_value"] = record.y_value
                found = True
            updated_rows.append(row)

        if not found:
            raise ValueError(f"No proposal row found for iteration {record.iteration}")

        _write_csv_atomic(self.proposals_path, _PROPOSAL_FIELDS, updated_rows)
        self.write_event("observation_completed", asdict(record))

    def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp_utc": _utc_now(),
            "event": event_type,
            "payload": payload,
        }
        _append_jsonl(self.events_path, event)


def _config_to_jsonable(config: FLIWBOConfig) -> dict[str, Any]:
    data = asdict(config)
    data["metadata_dir"] = None if config.metadata_dir is None else str(config.metadata_dir)
    return data


def _config_from_jsonable(data: dict[str, Any]) -> FLIWBOConfig:
    pr_data = dict(data["pr_config"])
    config_data = dict(data)
    config_data["pr_config"] = PROptimizerConfig(**pr_data)
    return FLIWBOConfig(**config_data)


def _proposal_to_row(proposal: OptimizationProposal, *, status: str) -> dict[str, Any]:
    return {
        "iteration": proposal.iteration,
        "status": status,
        "x_vector": json.dumps(proposal.x_vector),
        "acquisition_value": proposal.acquisition_value,
        "warp_alpha": json.dumps(proposal.warp_alpha),
        "warp_beta": json.dumps(proposal.warp_beta),
        "warp_indices": json.dumps(proposal.warp_indices),
        "warp_score": proposal.warp_score,
        "warp_search_scored": proposal.warp_search_scored,
        "proposed_at_utc": _utc_now(),
        "completed_at_utc": "",
        "y_value": "",
    }


def _proposal_from_row(row: dict[str, str]) -> OptimizationProposal:
    return OptimizationProposal(
        iteration=int(row["iteration"]),
        x_vector=_json_vector(row["x_vector"]),
        acquisition_value=float(row["acquisition_value"]),
        warp_alpha=[float(value) for value in _json_list(row["warp_alpha"])],
        warp_beta=[float(value) for value in _json_list(row["warp_beta"])],
        warp_indices=[int(value) for value in _json_list(row["warp_indices"])],
        warp_score=float(row["warp_score"]),
        warp_search_scored=int(row["warp_search_scored"]),
    )


def _record_from_proposal(proposal: OptimizationProposal, y_value: float) -> BOIterationRecord:
    return BOIterationRecord(
        iteration=proposal.iteration,
        x_vector=list(proposal.x_vector),
        y_value=float(y_value),
        acquisition_value=proposal.acquisition_value,
        warp_alpha=list(proposal.warp_alpha),
        warp_beta=list(proposal.warp_beta),
        warp_indices=list(proposal.warp_indices),
        warp_score=proposal.warp_score,
        warp_search_scored=proposal.warp_search_scored,
    )


def _validate_contiguous_iterations(records: list[BOIterationRecord]) -> None:
    expected = list(range(1, len(records) + 1))
    actual = [record.iteration for record in records]
    if actual != expected:
        raise ValueError(f"Completed iterations must be contiguous; got {actual}")


def _json_list(raw_value: str) -> list[Any]:
    value = json.loads(raw_value)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list, got: {raw_value!r}")
    return value


def _json_vector(raw_value: str) -> list[Number]:
    values = _json_list(raw_value)
    vector: list[Number] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Expected numeric x_vector values, got: {raw_value!r}")
        if isinstance(value, int):
            vector.append(int(value))
        else:
            vector.append(float(value))
    return vector


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_MANIFEST_FILE = "manifest.json"
_OBSERVATIONS_FILE = "observations.csv"
_PROPOSALS_FILE = "proposals.csv"
_EVENTS_FILE = "events.jsonl"

_OBSERVATION_FIELDS = [
    "source",
    "iteration",
    "x_vector",
    "y_value",
]

_PROPOSAL_FIELDS = [
    "iteration",
    "status",
    "x_vector",
    "acquisition_value",
    "warp_alpha",
    "warp_beta",
    "warp_indices",
    "warp_score",
    "warp_search_scored",
    "proposed_at_utc",
    "completed_at_utc",
    "y_value",
]
