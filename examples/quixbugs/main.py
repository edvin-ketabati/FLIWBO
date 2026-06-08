"""
Outer loop orchestrating the full agent evaluation pipeline.
"""

import argparse
import json
import os

from examples.quixbugs.prep_workspace import prep_workspace
from examples.quixbugs.run_agents import run_agent_loop
from examples.quixbugs.run_evaluation import run_evaluation
from fliwbo_core.BO_config import TIME_WEIGHT, TOKEN_WEIGHT


def update_agent_result_summary(
    result_file,
    *,
    total_tokens,
    elapsed_time,
    successful,
    failed,
    resolved_files,
    total_files,
    unresolved_files,
    passed_tests,
    failed_tests,
    objective_value,
):
    """
    Append the final pipeline summary to the existing agent result payload.
    """
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    total_agent_runs = successful + failed
    summary = {
        "status": "PIPELINE COMPLETE",
        "total_tokens_consumed_k": total_tokens / 1000,
        "total_execution_time_seconds": elapsed_time,
        "successful_agent_runs": successful,
        "failed_agent_runs": failed,
        "total_agent_runs": total_agent_runs,
        "resolved_files": resolved_files,
        "unresolved_files": unresolved_files,
        "total_files": total_files,
        "passed_test_cases": passed_tests,
        "failed_test_cases": failed_tests,
        "agent_result_file": str(result_file),
        "system_objective_value": objective_value,
    }
    payload["pipeline_summary"] = summary
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    """
    Main entry point: orchestrate the full pipeline.
    """
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

    parser = argparse.ArgumentParser(description="Run agent system and evaluation.")
    parser.add_argument(
        "--system-name",
        default="mock_agent_v1",
        help="System name used under examples/quixbugs/outputs and examples/quixbugs/workdirs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many QuixBugs program files to process, useful for smoke tests.",
    )
    args = parser.parse_args()

    system_name = args.system_name

    print("\n" + "=" * 80)
    print("AGENT SYSTEM EVALUATION PIPELINE")
    print("=" * 80)

    print("\nPhase 1: Prep workspace...")
    prep_workspace(system_name)

    print("\nPhase 2: Running agents...")
    successful, failed, result_file, total_tokens, elapsed_time = run_agent_loop(
        system_name=system_name,
        limit=args.limit,
    )

    print("\nPhase 3: Running evaluation...")
    evaluation_results = run_evaluation(system_name, result_file, limit=args.limit)

    resolved_files = evaluation_results.get("resolved_files", 0)
    total_files = evaluation_results.get("total_files", successful + failed)
    unresolved_files = evaluation_results.get("unresolved_files", total_files - resolved_files)
    passed_tests = evaluation_results.get("passed", 0)
    failed_tests = evaluation_results.get("failed", 0)
    objective_value = resolved_files - TOKEN_WEIGHT * total_tokens - TIME_WEIGHT * elapsed_time
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

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print(f"Total tokens consumed (k): {total_tokens/1000:,}")
    print(f"Total execution time: {elapsed_time:.2f}s")
    print(f"Successful agent runs: {successful}/{successful + failed}")
    print(f"Failed agent runs: {failed}/{successful + failed}")
    print(f"Resolved files: {resolved_files}/{total_files}")
    print(f"Unresolved files: {unresolved_files}/{total_files}")
    print(f"Passed test cases: {passed_tests}")
    print(f"Failed test cases: {failed_tests}")
    print(f"Agent result file: {result_file}")
    print(f"System objective value: {objective_value:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
