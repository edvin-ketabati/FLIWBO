"""
Estimate run-to-run noise for the full agent evaluation pipeline.

Runs the same benchmark multiple times and records only the objective value
from each completed full evaluation.
"""

import argparse
import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from examples.quixbugs.prep_workspace import prep_workspace
from examples.quixbugs.run_agents import run_agent_loop
from examples.quixbugs.run_evaluation import run_evaluation
from fliwbo_core.BO_config import TIME_WEIGHT, TOKEN_WEIGHT
from examples.quixbugs.features_to_spec import SPEC_PATH, features_to_spec
from examples.quixbugs.main import update_agent_result_summary


QUIXBUGS_DIR = Path(__file__).resolve().parent
BEST_X_VECTOR = [0, 0, 13, 0, 2, 7, 47, 0, 1, 31, 28, 4, 1, 13, 5, 2, 1, 9, 47, 5]

CSV_FIELDS = [
    "timestamp_utc",
    "objective_value",
    "n_completed",
    "mean_objective_value",
    "sample_std_objective_value",
    "standard_error_objective_value",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_row() -> dict:
    return {field: "" for field in CSV_FIELDS}


def append_csv_row(csv_path: Path, row: dict) -> None:
    """
    Append one row and force it to disk.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        clean_row = empty_row()
        clean_row.update(row)

        writer.writerow(clean_row)
        f.flush()
        os.fsync(f.fileno())


def read_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_completed_objective_values(csv_path: Path) -> list[float]:
    """
    Reads all rows with a valid objective_value.

    The final summary row is ignored because its objective_value is blank.
    """
    values = []

    for row in read_existing_rows(csv_path):
        objective_value = row.get("objective_value", "")

        if objective_value in ("", None):
            continue

        try:
            values.append(float(objective_value))
        except ValueError:
            continue

    return values


def compute_stats(values: list[float]) -> dict:
    n = len(values)

    if n == 0:
        return {
            "n": 0,
            "mean": "",
            "sample_std": "",
            "standard_error": "",
        }

    mean_value = statistics.mean(values)

    if n >= 2:
        sample_std = statistics.stdev(values)
        standard_error = sample_std / math.sqrt(n)
    else:
        sample_std = ""
        standard_error = ""

    return {
        "n": n,
        "mean": mean_value,
        "sample_std": sample_std,
        "standard_error": standard_error,
    }


def parse_x_vector(raw_vector: str) -> list[int]:
    """
    Parse a JSON-style X-vector, for example: "[2, 27, 2, 4, ...]".
    """
    parsed = json.loads(raw_vector)
    if not isinstance(parsed, list):
        raise ValueError("--x-vector must be a JSON list of integers")
    return [int(value) for value in parsed]


def build_mas_spec_from_x_vector(x_vector: list[int]) -> dict:
    """
    Build examples/quixbugs/mas_spec.json from a BO feature vector.
    """
    spec = features_to_spec(x_vector)
    print(f"Built MAS spec from X-vector: {SPEC_PATH}")
    return spec


def append_summary_row(csv_path: Path) -> None:
    """
    Append a summary row without modifying any existing data.
    """
    values = get_completed_objective_values(csv_path)
    stats = compute_stats(values)

    summary_row = empty_row()
    summary_row.update({
        "timestamp_utc": utc_now_iso(),
        "n_completed": stats["n"],
        "mean_objective_value": stats["mean"],
        "sample_std_objective_value": stats["sample_std"],
        "standard_error_objective_value": stats["standard_error"],
    })

    append_csv_row(csv_path, summary_row)


def run_single_full_evaluation(system_name: str) -> float:
    """
    Mirrors main.py's pipeline and returns only the objective value.
    """
    print("\nPhase 1: Prep workspace...")
    prep_workspace(system_name)

    print("\nPhase 2: Running agents...")
    successful, failed, result_file, total_tokens, elapsed_time = run_agent_loop(
        system_name=system_name,
    )

    print("\nPhase 3: Running evaluation...")
    evaluation_results = run_evaluation(system_name, result_file)

    resolved_files = evaluation_results.get("resolved_files", 0)
    total_files = evaluation_results.get("total_files", successful + failed)
    unresolved_files = evaluation_results.get(
        "unresolved_files",
        total_files - resolved_files,
    )
    passed_tests = evaluation_results.get("passed", 0)
    failed_tests = evaluation_results.get("failed", 0)

    objective_value = (
        resolved_files
        - TOKEN_WEIGHT * total_tokens
        - TIME_WEIGHT * elapsed_time
    )

    update_agent_result_summary(
        result_file,
        total_tokens=total_tokens,
        elapsed_time=elapsed_time,
        successful=successful,
        failed=failed,
        resolved_files=resolved_files,
        total_files=total_files,
        unresolved_files=unresolved_files,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        objective_value=objective_value,
    )

    return objective_value


def main():
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

    parser = argparse.ArgumentParser(
        description="Estimate objective-value noise by repeating full evaluations."
    )
    parser.add_argument(
        "--system-name",
        default="mock_agent_v1",
        help="System name used under examples/quixbugs/outputs and examples/quixbugs/workdirs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        required=True,
        help="Number of new full benchmark repeats to append in this invocation.",
    )
    parser.add_argument(
        "--x-vector",
        default=None,
        help=(
            "Optional JSON list feature vector to write to examples/quixbugs/mas_spec.json "
            "before running noise estimation. Defaults to the best BO vector."
        ),
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    x_vector = BEST_X_VECTOR if args.x_vector is None else parse_x_vector(args.x_vector)
    build_mas_spec_from_x_vector(x_vector)

    csv_path = QUIXBUGS_DIR / "noise_estimates" / f"{args.system_name}_noise.csv"

    completed_values = get_completed_objective_values(csv_path)
    completed_so_far = len(completed_values)
    target_completed = completed_so_far + args.repeats

    print("\n" + "=" * 80)
    print("NOISE ESTIMATION")
    print("=" * 80)
    print(f"System name: {args.system_name}")
    print(f"New repeats requested: {args.repeats}")
    print(f"Already completed repeats in CSV: {completed_so_far}")
    print(f"Target completed repeats after this run: {target_completed}")
    print(f"CSV path: {csv_path}")
    print("=" * 80 + "\n")

    while len(get_completed_objective_values(csv_path)) < target_completed:
        current_completed = len(get_completed_objective_values(csv_path))
        run_number = current_completed - completed_so_far + 1

        print("\n" + "-" * 80)
        print(f"Starting new full repeat {run_number}/{args.repeats}")
        print(f"Overall completed repeats after this one: {current_completed + 1}/{target_completed}")
        print("-" * 80)

        objective_value = run_single_full_evaluation(
            system_name=args.system_name,
        )

        append_csv_row(
            csv_path,
            {
                "timestamp_utc": utc_now_iso(),
                "objective_value": objective_value,
            },
        )

        print(f"\nCompleted new repeat {run_number}/{args.repeats}")
        print(f"Objective value: {objective_value:.6f}")
        print(f"CSV updated: {csv_path}")

    append_summary_row(csv_path)

    stats = compute_stats(get_completed_objective_values(csv_path))
    print_final_stats(stats)
    print(f"Final CSV written to: {csv_path}")


def print_final_stats(stats: dict) -> None:
    print("\n" + "=" * 80)
    print("NOISE ESTIMATION COMPLETE")
    print("=" * 80)
    print(f"Completed full repeats: {stats['n']}")

    if stats["n"] == 0:
        print("No completed runs available.")
    elif stats["n"] == 1:
        print(f"Mean objective value: {stats['mean']:.6f}")
        print("Sample standard deviation: unavailable with n=1")
        print("Standard error: unavailable with n=1")
    else:
        print(f"Mean objective value: {stats['mean']:.6f}")
        print(f"Sample standard deviation: {stats['sample_std']:.6f}")
        print(f"Standard error: {stats['standard_error']:.6f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
