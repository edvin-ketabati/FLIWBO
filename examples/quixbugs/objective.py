"""QuixBugs objective adapter.

This is the bridge between the pure optimizer and the QuixBugs example runtime.
It receives a discrete typed vector, builds the corresponding MAS, evaluates
that MAS, and returns one scalar score.
"""

from __future__ import annotations

import json
from pathlib import Path

from fliwbo_core.BO_config import OBJECTIVE_EVALUATION_LIMIT, TIME_WEIGHT, TOKEN_WEIGHT

from .features_to_spec import features_to_spec
from .prep_workspace import prep_workspace
from .run_agents import run_agent_loop
from .run_evaluation import run_evaluation


QUIXBUGS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = QUIXBUGS_DIR / "Results"


class QuixBugsObjective:
    """
    Callable objective adapter for the QuixBugs MAS example.

    The FLIWBO optimizer calls objective(x_vector). This adapter owns the
    iteration counter only so it can create unique workspace and result names.
    """

    def __init__(
        self,
        *,
        limit: int | None = OBJECTIVE_EVALUATION_LIMIT,
        system_name_prefix: str = "MAS_iteration",
    ):
        self.limit = limit
        self.system_name_prefix = system_name_prefix
        self._iteration = 0

    def __call__(self, x_vector) -> float:
        self._iteration += 1
        return evaluate_quixbugs_objective(
            x_vector,
            system_name=f"{self.system_name_prefix}_{self._iteration}",
            limit=self.limit,
        )


def evaluate_quixbugs_objective(
    x_vector,
    *,
    system_name: str,
    limit: int | None = OBJECTIVE_EVALUATION_LIMIT,
) -> float:
    """
    Build a MAS from x_vector, evaluate it on QuixBugs, and return a scalar score.
    """
    features_to_spec(x_vector, QUIXBUGS_DIR / "mas_spec.json")
    prep_workspace(system_name)

    _successful, _failed, result_file, total_tokens, elapsed_time = run_agent_loop(
        system_name=system_name,
        limit=limit,
    )

    run_evaluation(system_name, result_file, limit=limit)

    results = _find_and_load_results(system_name)
    resolved_files = results["resolved_files"]

    return resolved_files - TOKEN_WEIGHT * total_tokens - TIME_WEIGHT * elapsed_time


def _find_and_load_results(system_name: str) -> dict:
    matching_files = list(RESULTS_DIR.glob(f"{system_name}*.json"))
    if not matching_files:
        raise FileNotFoundError(f"No evaluation results found for system {system_name}")

    latest_file = max(matching_files, key=lambda path: path.stat().st_mtime)
    with latest_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    return {
        "resolved_files": results["resolved_files"],
        "total_files": results["total_files"],
    }
