"""Evaluate seed BO vectors and write their objective values into the CSV."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from examples.quixbugs.prep_workspace import prep_workspace
from examples.quixbugs.run_agents import quixbugs_target_files, run_agent_loop
from examples.quixbugs.run_evaluation import run_evaluation
from fliwbo_core.BO_config import TIME_WEIGHT, TOKEN_WEIGHT
from examples.quixbugs.features_to_spec import features_to_spec
from examples.quixbugs.resource_statement import MAX_NUMBER_OF_AGENTS, NUMBER_OF_FEATURES_PER_AGENT


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ROOT_DIR / "initial_x_points.csv"
OBJECTIVE_COLUMN = "objective_value"
LABEL_COLUMN = "label"
VECTOR_COLUMNS = [
    f"a{agent}_{feature}"
    for agent in range(1, MAX_NUMBER_OF_AGENTS + 1)
    for feature in ("llm", "toolset", "prompt", "next")
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each initial X-vector on the full QuixBugs benchmark and "
            "write the measured objective value back into the CSV."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV containing X-vector columns and an objective_value column.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Re-evaluate rows that already have objective_value filled in.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=40,
        help="Expected number of QuixBugs target files for a full evaluation.",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    rows, fieldnames = _read_rows(csv_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for row_index, row in enumerate(rows, start=1):
        label = (row.get(LABEL_COLUMN) or f"row_{row_index}").strip()
        existing_value = (row.get(OBJECTIVE_COLUMN) or "").strip()
        if existing_value and not args.rerun_completed:
            print(f"[{row_index}/{len(rows)}] Skipping {label}: objective_value already set.")
            continue

        x_vector = _row_to_feature_vector(row, row_index)
        system_name = _system_name_for_row(label, row_index, run_id)

        print(f"[{row_index}/{len(rows)}] Evaluating {label} as {system_name}")
        objective_value = evaluate_full_objective(
            x_vector,
            system_name,
            expected_total=args.expected_total,
        )

        row[OBJECTIVE_COLUMN] = f"{objective_value:.12g}"
        _write_rows(csv_path, rows, fieldnames)
        print(f"[{row_index}/{len(rows)}] Wrote objective_value={row[OBJECTIVE_COLUMN]}")


def evaluate_full_objective(
    feature_vector: list[int],
    system_name: str,
    *,
    expected_total: int | None = 40,
) -> float:
    """
    Build the MAS spec, run all available QuixBugs targets, and return the raw objective.
    """
    features_to_spec(feature_vector)
    workspace = prep_workspace(system_name)

    target_files = quixbugs_target_files(workspace, limit=None)
    if expected_total is not None and len(target_files) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} QuixBugs target files, found {len(target_files)} in {workspace}"
        )

    _successful, _failed, result_file, total_tokens, elapsed_time = run_agent_loop(
        system_name=system_name,
        limit=None,
    )
    evaluation_results = run_evaluation(
        system_name,
        result_file,
        limit=None,
    )

    resolved_files = int(evaluation_results.get("resolved_files", 0) or 0)
    total_files = int(evaluation_results.get("total_files", 0) or 0)
    if expected_total is not None and total_files != expected_total:
        raise RuntimeError(
            f"Expected evaluation over {expected_total} files, got {total_files}"
        )
    return resolved_files - TOKEN_WEIGHT * total_tokens - TIME_WEIGHT * elapsed_time


def _read_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    missing_columns = [name for name in VECTOR_COLUMNS if name not in fieldnames]
    if missing_columns:
        raise ValueError(f"CSV is missing vector columns: {missing_columns}")

    if OBJECTIVE_COLUMN not in fieldnames:
        fieldnames.append(OBJECTIVE_COLUMN)
        for row in rows:
            row[OBJECTIVE_COLUMN] = ""

    return rows, fieldnames


def _write_rows(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(csv_path)


def _row_to_feature_vector(row: dict[str, str], row_index: int) -> list[int]:
    expected_length = MAX_NUMBER_OF_AGENTS * NUMBER_OF_FEATURES_PER_AGENT
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


def _system_name_for_row(label: str, row_index: int, run_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
    if not slug:
        slug = f"row_{row_index}"
    return f"initial_x_{row_index:02d}_{slug}_{run_id}"


if __name__ == "__main__":
    main()
