"""Run FLIWBO on the QuixBugs MAS design-space example."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from fliwbo_core.BO_config import N_ITERS
from fliwbo_core import Discrete, FLIWBOConfig, FLIWBOOptimizer, SearchSpace
from examples.quixbugs.objective import QuixBugsObjective
from examples.quixbugs.search_space import get_default_search_space
from examples.quixbugs.resource_statement import MAX_NUMBER_OF_AGENTS, NUMBER_OF_FEATURES_PER_AGENT


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ROOT_DIR / "initial_x_points.csv"
DEFAULT_METADATA_DIR = ROOT_DIR / "BO metadata"
OBJECTIVE_COLUMN = "objective_value"
LABEL_COLUMN = "label"
VECTOR_COLUMNS = [
    f"a{agent}_{feature}"
    for agent in range(1, MAX_NUMBER_OF_AGENTS + 1)
    for feature in ("llm", "toolset", "prompt", "next")
]


def main() -> None:
    search_space = get_default_search_space()
    X_init, y_init, labels = load_initial_observations(DEFAULT_CSV_PATH, search_space)
    objective = QuixBugsObjective()

    print(f"Loaded {len(y_init)} initial BO observations from {DEFAULT_CSV_PATH}")
    for label, y_value in zip(labels, y_init):
        print(f"  {label}: objective_value={y_value:.12g}")

    config = FLIWBOConfig(
        n_iters=N_ITERS,
        metadata_dir=DEFAULT_METADATA_DIR,
        log_csv=True,
        verbose=True,
    )
    optimizer = FLIWBOOptimizer(search_space, config=config)
    optimizer.run(objective, X_init, y_init)


def load_initial_observations(
    csv_path: Path,
    search_space: SearchSpace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read initial BO seed observations from CSV.

    The feature layout per agent is:
    [llm_choice, toolset_choice, prompt_choice, next_agent]
    repeated for MAX_NUMBER_OF_AGENTS.
    """
    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Initial observations CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV contains no initial observations: {csv_path}")

    missing_columns = [name for name in VECTOR_COLUMNS if name not in fieldnames]
    if missing_columns:
        raise ValueError(f"CSV is missing vector columns: {missing_columns}")
    if OBJECTIVE_COLUMN not in fieldnames:
        raise ValueError(
            f"CSV is missing {OBJECTIVE_COLUMN!r}. Run evaluate_initial_x_points.py first."
        )

    expected_length = MAX_NUMBER_OF_AGENTS * NUMBER_OF_FEATURES_PER_AGENT

    x_rows: list[list[int]] = []
    y_values: list[float] = []
    labels: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        label = (row.get(LABEL_COLUMN) or f"row_{row_index}").strip()
        x_vector = _row_to_feature_vector(row, row_index, expected_length)
        _validate_feature_bounds(x_vector, search_space, row_index)

        raw_objective = (row.get(OBJECTIVE_COLUMN) or "").strip()
        if raw_objective == "":
            raise ValueError(
                f"Row {row_index} ({label}) is missing {OBJECTIVE_COLUMN!r}. "
                "Run evaluate_initial_x_points.py before starting BO."
            )
        try:
            y_value = float(raw_objective)
        except ValueError as exc:
            raise ValueError(
                f"Row {row_index} ({label}) has invalid objective value {raw_objective!r}"
            ) from exc

        x_rows.append(x_vector)
        y_values.append(y_value)
        labels.append(label)

    return (
        np.asarray(x_rows, dtype=int),
        np.asarray(y_values, dtype=float),
        labels,
    )


def _row_to_feature_vector(
    row: dict[str, str],
    row_index: int,
    expected_length: int,
) -> list[int]:
    values: list[int] = []
    for column in VECTOR_COLUMNS:
        raw_value = (row.get(column) or "").strip()
        if raw_value == "":
            raise ValueError(f"Row {row_index} has a blank X-vector value in {column}")
        try:
            values.append(int(raw_value))
        except ValueError as exc:
            raise ValueError(
                f"Row {row_index} has non-integer X-vector value {raw_value!r} in {column}"
            ) from exc

    if len(values) != expected_length:
        raise ValueError(
            f"Row {row_index} produced {len(values)} feature values, expected {expected_length}"
        )
    return values


def _validate_feature_bounds(
    x_vector: list[int],
    search_space: SearchSpace,
    row_index: int,
) -> None:
    if len(x_vector) != search_space.dimension:
        raise ValueError(
            f"Row {row_index} has {len(x_vector)} values, expected {search_space.dimension}"
        )

    out_of_bounds = [
        (column, value, variable.num_choices)
        for column, value, variable in zip(VECTOR_COLUMNS, x_vector, search_space.variables)
        if isinstance(variable, Discrete) and (value < 0 or value >= variable.num_choices)
    ]
    if out_of_bounds:
        details = ", ".join(
            f"{column}={value} outside 0..{size - 1}"
            for column, value, size in out_of_bounds
        )
        raise ValueError(f"Row {row_index} has out-of-range choices: {details}")


if __name__ == "__main__":
    main()
