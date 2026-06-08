"""Run the spec-driven MAS over prepared QuixBugs target files.

The optimizer never calls this file directly. The QuixBugs objective adapter
uses it after a vector has been decoded into mas_spec.json and a fresh workspace
has been prepared.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import anyio
from dotenv import load_dotenv

from examples.quixbugs.mas_builder import build_mas_system
from examples.quixbugs.mcp_tools import DEFAULT_ALLOWED_TOOLS, mcp_tools_from_mcp, start_mcp


load_dotenv()

DEFAULT_TEMPERATURE = 0.0
QUIXBUGS_TASK_ID = "quixbugs-python"
QUIXBUGS_TASK_PROMPT = """
You have been given a workspace containing Python programs from the QuixBugs benchmark.
The target program has a bug on exactly one line. Your job is to identify the buggy line
in that Python program and make the smallest code change needed to fix it.

Work only inside the prepared workspace and use the provided tools. Do not edit files
outside the workspace. Focus on the target file named below. When you change code, keep the fix minimal and avoid unrelated
refactoring.
""".strip()

# Try to import centralized recursion limit; fall back to 80 if unavailable.
try:
    from .resource_statement import AGENT_RECURSION_LIMIT  # type: ignore
except Exception:
    try:
        from examples.quixbugs.resource_statement import AGENT_RECURSION_LIMIT  # type: ignore
    except Exception:
        AGENT_RECURSION_LIMIT = 80


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue

            if isinstance(item, dict):
                dict_text = item.get("text") or item.get("content")
                if isinstance(dict_text, str):
                    parts.append(dict_text)
                    continue

            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                parts.append(json.dumps(model_dump(mode="json"), ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if hasattr(message, "__dict__"):
        return dict(message.__dict__)
    return {"content": str(message)}


def _short_text(value: Any, limit: int = 1_000_000) -> str:
    text = _content_to_text(value)
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
    return text


def _message_key(message: Any) -> str:
    msg = _message_to_dict(message)
    stable = {
        "id": msg.get("id"),
        "role": msg.get("role") or msg.get("type"),
        "name": msg.get("name"),
        "tool_call_id": msg.get("tool_call_id"),
        "content": msg.get("content"),
        "tool_calls": msg.get("tool_calls"),
    }
    return json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)


def _print_message(message: Any) -> None:
    msg = _message_to_dict(message)
    message_type = msg.get("type") or msg.get("role") or type(message).__name__
    name = msg.get("name")
    label = f"{message_type}"
    if name:
        label += f":{name}"

    print(f"\n[Message] {label}", flush=True)
    content = msg.get("content")
    if content:
        print(_short_text(content), flush=True)

    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        print("[Tool calls requested]", flush=True)
        print(_short_text(json.dumps(tool_calls, indent=2, ensure_ascii=False, default=str)), flush=True)

    invalid_tool_calls = msg.get("invalid_tool_calls") or []
    if invalid_tool_calls:
        print("[Invalid tool calls]", flush=True)
        print(_short_text(json.dumps(invalid_tool_calls, indent=2, ensure_ascii=False, default=str)), flush=True)

    if msg.get("tool_call_id") and not content:
        print(f"tool_call_id: {msg['tool_call_id']}", flush=True)


def _build_task_prompt(target_file: str) -> str:
    return "\n".join([
        QUIXBUGS_TASK_PROMPT,
        "",
        f"Target file: {target_file}",
        "The available tools are already bound to this file and do not accept a path argument.",
    ])


def _workspace_for_system(system_name: str) -> Path:
    return (Path(__file__).resolve().parent / "workdirs" / system_name).resolve()


def quixbugs_target_files(workspace: Path, limit: int | None = None) -> list[str]:
    targets = sorted(
        path.name
        for path in workspace.glob("*.py")
        if path.name != "node.py" and not path.name.endswith("_test.py")
    )
    if limit is not None:
        return targets[:limit]
    return targets


def extract_final_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        message_type = getattr(message, "type", None) or getattr(message, "role", None)
        if message_type in {"ai", "assistant"}:
            return _content_to_text(getattr(message, "content", ""))
        if isinstance(message, dict) and message.get("role") == "assistant":
            return _content_to_text(message.get("content", ""))
    return ""


async def _run_task_async(
    agent: Any,
    task: str,
    thread_id: Optional[str],
    recursion_limit: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    last_state: dict[str, Any] = {}
    transition_count = 0
    hit_agent_recursion_limit = False
    hit_max_transitions = False
    max_transitions = None
    printed_message_keys: set[str] = set()
    active_agent_id: int | None = None

    def _print_new_messages(state: dict[str, Any]) -> None:
        messages = state.get("new_agent_messages")
        if messages is None:
            messages = state.get("agent_messages") or state.get("messages", [])
        if not isinstance(messages, list):
            return

        for message in messages:
            key = _message_key(message)
            if key in printed_message_keys:
                continue
            printed_message_keys.add(key)
            _print_message(message)

    try:
        async for stream_item in agent.astream(
            {"messages": [{"role": "user", "content": task}]},
            config=config,
        ):
            if isinstance(stream_item, dict):
                last_state = stream_item
                transition_count = int(stream_item.get("transition_count") or transition_count)
                hit_agent_recursion_limit = hit_agent_recursion_limit or bool(stream_item.get("hit_agent_recursion_limit"))
                hit_max_transitions = hit_max_transitions or bool(stream_item.get("hit_max_transitions"))
                max_transitions = stream_item.get("max_transitions") or max_transitions
                current_agent_id = stream_item.get("current_agent_id")
                if isinstance(current_agent_id, int) and current_agent_id != active_agent_id:
                    active_agent_id = current_agent_id
                    print(f"\nEntering Agent {current_agent_id}", flush=True)
                if stream_item.get("hit_agent_recursion_limit"):
                    completed_agent_id = stream_item.get("completed_agent_id")
                    suffix = f" for agent {completed_agent_id}" if completed_agent_id is not None else ""
                    print(f"\n[Agent Control] Hit recursion limit{suffix}; moving to next agent.")
                if stream_item.get("hit_max_transitions"):
                    print(f"\n[Agent Control] Hit max transitions ({max_transitions}); stopping MAS execution.")
                _print_new_messages(stream_item)
    except Exception as exc:
        exc_text = str(exc or "").lower()
        final_ai_text = extract_final_text(last_state).strip()
        if "recursion limit" in exc_text or "recursion_limit" in exc_text or "recursionlimit" in exc_text:
            return {
                "agent_success": True,
                "agent_error": None,
                "result": last_state,
                "final_text": final_ai_text,
                "final_ai_text": final_ai_text,
                "transition_count": transition_count,
                "hit_recursion_limit": True,
                "hit_agent_recursion_limit": True,
                "hit_max_transitions": hit_max_transitions,
                "max_transitions": max_transitions,
            }

        return {
            "agent_success": False,
            "agent_error": str(exc),
            "result": None,
            "final_text": final_ai_text,
            "final_ai_text": final_ai_text,
            "transition_count": transition_count,
            "hit_agent_recursion_limit": hit_agent_recursion_limit,
            "hit_max_transitions": hit_max_transitions,
            "max_transitions": max_transitions,
        }

    final_ai_text = extract_final_text(last_state).strip()
    return {
        "agent_success": True,
        "agent_error": None,
        "result": last_state,
        "final_text": final_ai_text,
        "final_ai_text": final_ai_text,
        "transition_count": transition_count,
        "hit_agent_recursion_limit": hit_agent_recursion_limit,
        "hit_max_transitions": hit_max_transitions,
        "max_transitions": max_transitions,
    }


def run_task(
    agent: Any,
    task: str,
    *,
    thread_id: Optional[str] = None,
    recursion_limit: int = AGENT_RECURSION_LIMIT,
) -> dict[str, Any]:
    return anyio.run(_run_task_async, agent, task, thread_id, recursion_limit)


def run_agent_loop(
    instances: list[dict[str, Any]] | None = None,
    system_name: str = "mock_agent",
    limit: int | None = None,
) -> tuple[int, int, Path, int, float]:
    """
    Process the prepared QuixBugs workspace through the spec-driven MAS.

    The instances argument is kept temporarily so older callers do not break
    while the workflow moves away from per-issue benchmark dictionaries.
    """
    del instances

    return anyio.run(_run_agent_loop_async, system_name, limit)


async def _run_agent_loop_async(
    system_name: str,
    limit: int | None,
) -> tuple[int, int, Path, int, float]:
    print(f"\n{'=' * 80}")
    print(f"Starting evaluation for system: {system_name}")
    print("Processing QuixBugs Python workspace")
    print(f"{'=' * 80}\n")

    workspace = _workspace_for_system(system_name)
    target_files = quixbugs_target_files(workspace, limit=limit)
    if not target_files:
        raise FileNotFoundError(f"No QuixBugs target files found in {workspace}")

    output_dir = Path(__file__).resolve().parent / "outputs" / system_name
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / "agent_result.json"
    if result_file.exists():
        result_file.unlink()

    successful = 0
    failed = 0
    total_tokens = 0
    loop_start_time = time.time()

    file_results: list[dict[str, Any]] = []

    async with start_mcp(workspace) as mcp_runtime:
        tools = await mcp_tools_from_mcp(mcp_runtime, DEFAULT_ALLOWED_TOOLS)
        if not tools:
            raise RuntimeError("No MCP tools were loaded from the filesystem server.")

        agent = build_mas_system(
            tools,
            model_override=None,
            temperature=DEFAULT_TEMPERATURE,
        )

        previous_total_tokens = 0
        for index, target_file in enumerate(target_files, start=1):
            print(f"\n{'=' * 80}")
            print(f"[{index}/{len(target_files)}] Processing: {target_file}")
            print(f"{'=' * 80}")

            try:
                print("\n-> Starting agent execution...")
                mcp_runtime.target_file = target_file
                run_result = await _run_task_async(
                    agent,
                    _build_task_prompt(target_file),
                    f"{system_name}-{Path(target_file).stem}",
                    AGENT_RECURSION_LIMIT,
                )

                cumulative_usage = agent.get_cumulative_token_usage()
                cumulative_total_tokens = int(cumulative_usage.get("total_tokens") or 0)
                file_tokens = max(0, cumulative_total_tokens - previous_total_tokens)
                previous_total_tokens = cumulative_total_tokens
                token_usage = {"total_tokens": file_tokens}
                total_tokens = cumulative_total_tokens

                final_ai_text = (run_result.get("final_ai_text") or run_result.get("final_text") or "").strip()

                if not run_result["agent_success"]:
                    print(f"\nAgent failed: {run_result['agent_error']}")
                    failed += 1
                else:
                    successful += 1

                file_results.append(
                    {
                        "target_file": target_file,
                        "agent_success": run_result["agent_success"],
                        "agent_error": run_result["agent_error"],
                        "final_text": final_ai_text,
                        "transition_count": run_result.get("transition_count", 0),
                        "hit_agent_recursion_limit": bool(
                            run_result.get("hit_agent_recursion_limit") or run_result.get("hit_recursion_limit")
                        ),
                        "hit_max_transitions": bool(run_result.get("hit_max_transitions")),
                        "max_transitions": run_result.get("max_transitions"),
                        "token_usage": token_usage,
                    }
                )
            except Exception as exc:
                print(f"Failed to process {target_file}: {exc}")
                import traceback

                traceback.print_exc()
                failed += 1
                file_results.append(
                    {
                        "target_file": target_file,
                        "agent_success": False,
                        "agent_error": str(exc),
                        "final_text": "",
                        "transition_count": 0,
                        "hit_agent_recursion_limit": False,
                        "hit_max_transitions": False,
                        "max_transitions": None,
                        "token_usage": {},
                    }
                )

    try:
        repo_path = _workspace_for_system(system_name)
        result_payload = {
            "task_id": QUIXBUGS_TASK_ID,
            "repo_path": str(repo_path),
            "processed_files": target_files,
            "file_results": file_results,
            "successful": successful,
            "failed": failed,
            "token_usage": {"total_tokens": total_tokens},
        }
        result_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        print(f"\nResult written to: {result_file}")
    except Exception as exc:
        print(f"Could not write agent result file: {exc}")

    elapsed_time = time.time() - loop_start_time

    print(f"\n{'=' * 80}")
    print("Agent loop completed:")
    print(f"  Successful: {successful}/{len(target_files)}")
    print(f"  Failed: {failed}/{len(target_files)}")
    print(f"  Total tokens consumed (k): {total_tokens / 1000:,}")
    print(f"  Execution time: {elapsed_time:.2f}s")
    print(f"  Result written to: {result_file}")
    print(f"{'=' * 80}\n")

    return successful, failed, result_file, total_tokens, elapsed_time
