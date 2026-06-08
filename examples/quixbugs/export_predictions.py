"""Export edited workspace diffs as benchmark prediction records."""

import json
import subprocess
from pathlib import Path


def _git(repo_path, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )


def _list_untracked_files(repo_path):
    result = _git(repo_path, "ls-files", "--others", "--exclude-standard", "-z")
    return [p for p in result.stdout.split("\0") if p]


def git_diff(repo_path):
    """Return a git diff including staged, unstaged, and untracked files."""

    # Compare working tree+index against HEAD so staged and unstaged edits are both included.
    untracked_files = _list_untracked_files(repo_path)

    if untracked_files:
        _git(repo_path, "add", "--intent-to-add", "--", *untracked_files)

    try:
        result = _git(repo_path, "diff", "--binary", "HEAD", "--")
        return result.stdout
    finally:
        if untracked_files:
            _git(repo_path, "reset", "-q", "--", *untracked_files)


def append_prediction(system_name, instance_id, repo_path, out_path):
    """Append one JSONL prediction record for an edited benchmark instance."""

    patch = git_diff(repo_path)
    if not patch.strip():
        raise ValueError(
            f"No code changes found for {instance_id}; refusing to export empty prediction."
        )

    record = {
        "instance_id": instance_id,
        "model_name_or_path": system_name,
        "model_patch": patch,
    }
    # Ensure the output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
