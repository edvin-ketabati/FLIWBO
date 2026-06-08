"""Evaluate MAS-edited QuixBugs files with the canonical tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from examples.quixbugs.mcp_tools import run_quixbugs_tests
from examples.quixbugs.run_agents import quixbugs_target_files


QUIXBUGS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = QUIXBUGS_DIR / "Results"


def _workspace_for_system(system_name: str) -> Path:
    return (QUIXBUGS_DIR / "workdirs" / system_name).resolve()


def run_evaluation(
    system_name: str,
    result_file: Path | None = None,
    *,
    limit: int | None = None,
    target_files: Sequence[str] | None = None,
) -> dict:
    """
    Run canonical QuixBugs pytest tests against the MAS-edited workspace.

    The test files are copied from examples/quixbugs/evaluation_repos/QuixBugs/python_testcases
    into a temporary test root, while python_programs is copied from the edited
    workspace. This keeps the executed tests untouched by the MAS.
    """
    workspace = _workspace_for_system(system_name)
    if target_files is not None:
        targets = list(target_files)
    elif result_file is not None and result_file.exists():
        result_payload = json.loads(result_file.read_text(encoding="utf-8"))
        targets = list(result_payload.get("processed_files") or [])
    else:
        targets = quixbugs_target_files(workspace, limit=limit)

    if limit is not None:
        targets = targets[:limit]
    if not targets:
        raise FileNotFoundError(f"No QuixBugs target files found for evaluation in {workspace}")

    print(f"\n{'=' * 80}")
    print("Running QuixBugs evaluation...")
    print(f"Testing {len(targets)} target file(s)")
    print(f"{'=' * 80}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"{system_name}_quixbugs_{run_suffix}.json"

    per_file_results = []
    passed = 0
    failed = 0
    skipped = 0
    resolved_files = 0
    for target in targets:
        print(f"Testing {target}...")
        file_result = run_quixbugs_tests(workspace, target_file=target)
        file_passed = int(file_result.get("passed", 0) or 0)
        file_failed = int(file_result.get("failed", 0) or 0)
        file_skipped = int(file_result.get("skipped", 0) or 0)
        resolved = file_failed == 0
        per_file_results.append({"target_file": target, "resolved": resolved, **file_result})
        passed += file_passed
        failed += file_failed
        skipped += file_skipped
        if resolved:
            resolved_files += 1

    payload = {
        "system_name": system_name,
        "workspace": str(workspace),
        "target_files": targets,
        "total_files": len(targets),
        "resolved_files": resolved_files,
        "unresolved_files": len(targets) - resolved_files,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "per_file_results": per_file_results,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Resolved files: {payload['resolved_files']}/{payload['total_files']}")
    print(f"Unresolved files: {payload['unresolved_files']}/{payload['total_files']}")
    print(f"Results written to: {results_path}")
    return payload
