"""Prepare a fresh editable QuixBugs workspace for one MAS evaluation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


QUIXBUGS_DIR = Path(__file__).resolve().parent
QUIXBUGS_PROGRAMS = QUIXBUGS_DIR / "evaluation_repos" / "QuixBugs" / "python_programs"


def prep_workspace(system_name: str) -> Path:
    """Copy fresh QuixBugs Python program files to the editable workdir."""
    workdirs_root = (QUIXBUGS_DIR / "workdirs").resolve()
    dest = (workdirs_root / system_name).resolve()
    if not dest.is_relative_to(workdirs_root):
        raise ValueError(f"Workspace escaped the expected workdirs root: {dest}")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for source_file in QUIXBUGS_PROGRAMS.glob("*.py"):
        if source_file.name.endswith("_test.py"):
            continue
        shutil.copy2(source_file, dest / source_file.name)

    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a QuixBugs Python workspace.")
    parser.add_argument("--system-name", default="mock_agent_v1", help="System name used under workdirs/")
    args = parser.parse_args()

    workspace = prep_workspace(args.system_name)
    print(f"Workspace ready at: {workspace}")


if __name__ == "__main__":
    main()
