from __future__ import annotations

from pathlib import Path

from examples.quixbugs.mcp_tools import MCPRuntime, start_mcp


def quixbugs_dir() -> Path:
    return Path(__file__).resolve().parent


def workspace_for_instance(system_name: str, instance_id: str) -> Path:
    return quixbugs_dir() / "workdirs" / system_name / instance_id / "repo"


def start_instance_mcp(system_name: str, instance_id: str) -> MCPRuntime:
    workspace = workspace_for_instance(system_name, instance_id)
    return start_mcp(workspace)
