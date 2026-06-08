"""Tool and test helpers for the QuixBugs MAS runtime.

This file adapts filesystem/MCP tools into LangChain tools that agents can use.
It also provides the canonical QuixBugs test runner used by the evaluation step.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
import types
import builtins
from contextlib import AsyncExitStack
from pathlib import Path
from pathlib import Path as _PathType
from pathlib import PurePosixPath
from typing import Any, Sequence, cast

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model, ConfigDict

QUIXBUGS_DIR = Path(__file__).resolve().parent
QUIXBUGS_ROOT = QUIXBUGS_DIR / "evaluation_repos" / "QuixBugs"
QUIXBUGS_TESTCASES = QUIXBUGS_ROOT / "python_testcases"
QUIXBUGS_JSON_TESTCASES = QUIXBUGS_ROOT / "json_testcases"
QUIXBUGS_CONFTEST = QUIXBUGS_ROOT / "conftest.py"
MCP_IMAGE = "mcp/filesystem"
MCP_WORKSPACE_ROOT = "/projects/workspace"
MAX_RAW_OUTPUT_CHARS = 1_000_000
MAX_TOOL_OUTPUT_CHARS = 1_000_000
MAX_GREP_OUTPUT_CHARS = 1_000_000
MAX_GREP_FILE_BYTES = 1_000_000
DEFAULT_QUIXBUGS_TEST_TIMEOUT = 10
MAX_CONSOLE_TOOL_OUTPUT_CHARS = 1_000_000
DEFAULT_GREP_INCLUDE_GLOBS: tuple[str, ...] = (
    "*.py", "*.pyi", "*.js", "*.jsx", "*.ts", "*.tsx",
    "*.java", "*.go", "*.rs", "*.c", "*.cc", "*.cpp", "*.h", "*.hpp",
    "*.rb", "*.php", "*.scala", "*.kt", "*.swift",
    "*.sql", "*.yml", "*.yaml", "*.json", "*.toml", "*.ini", "*.cfg",
    "*.md", "*.rst", "*.txt",
)
DEFAULT_GREP_EXCLUDE_GLOBS: tuple[str, ...] = (
    ".git/*", ".hg/*", ".svn/*",
    "__pycache__/*", ".pytest_cache/*", ".mypy_cache/*", ".ruff_cache/*",
    ".tox/*", ".nox/*", ".venv/*", "venv/*", "env/*",
    "node_modules/*", "dist/*", "build/*", "target/*", ".eggs/*",
)
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "read_file",
    "edit_file",
    "write_file",
    "search_current_file",
    "run_tests",
)


def _normalize_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Workspace path does not exist or is not a directory: {resolved}")
    return resolved


def _to_container_path(path: str) -> str:
    clean = (path or ".").strip()
    if clean in {"", ".", "./"}:
        return MCP_WORKSPACE_ROOT
    if clean.startswith("/projects/"):
        return clean
    return f"{MCP_WORKSPACE_ROOT}/{clean.lstrip('/')}"


def _to_workspace_path(workspace: Path, path: str | Path) -> Path:
    """Resolve an agent-supplied path to a real path inside the host workspace."""
    clean = str(path or ".").strip() or "."

    # Agents often see MCP/container paths in tool output. Map those back to the
    # host workspace so custom Python tools and MCP tools can share path strings.
    if clean == MCP_WORKSPACE_ROOT or clean.startswith(f"{MCP_WORKSPACE_ROOT}/"):
        rel = clean[len(MCP_WORKSPACE_ROOT):].lstrip("/")
        candidate = workspace / Path(*PurePosixPath(rel).parts)
    else:
        raw = Path(clean)
        candidate = raw if raw.is_absolute() else workspace / raw

    workspace_resolved = workspace.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError as exc:
        raise ValueError(f"Path is outside workspace: {path}") from exc
    return resolved


def _rel_posix(workspace: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _matches_any_glob(rel_path: str, patterns: Sequence[str]) -> bool:
    normalized = rel_path.replace(os.sep, "/")
    name = normalized.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in patterns
    )


def _iter_searchable_files(
    workspace: Path,
    root: Path,
    include: Sequence[str],
    exclude: Sequence[str],
) -> list[Path]:
    if root.is_file():
        rel = _rel_posix(workspace, root)
        if _matches_any_glob(rel, exclude):
            return []
        if include and not _matches_any_glob(rel, include):
            return []
        return [root]

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)

        kept_dirnames: list[str] = []
        for dirname in dirnames:
            dir_rel = _rel_posix(workspace, current_dir / dirname)
            if _matches_any_glob(f"{dir_rel}/", exclude):
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in filenames:
            file_path = current_dir / filename
            rel = _rel_posix(workspace, file_path)
            if _matches_any_glob(rel, exclude):
                continue
            if include and not _matches_any_glob(rel, include):
                continue
            files.append(file_path)
    return files


def _compile_grep_pattern(pattern: str, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    expression = pattern if regex else re.escape(pattern)
    try:
        return re.compile(expression, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc


def _search_files_impl(
    workspace: Path,
    path: str = ".",
    pattern: str = "",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    context: int = 2,
    max_matches: int = 50,
    search_filenames: bool = True,
    tool_name: str = "search_current_file",
) -> str:
    """Search file contents and filenames inside the workspace."""
    if not pattern:
        return f"Tool error ({tool_name}): pattern must not be empty."

    context = max(0, min(int(context), 10))
    max_matches = max(1, min(int(max_matches), 200))
    include_patterns = tuple(include or DEFAULT_GREP_INCLUDE_GLOBS)
    exclude_patterns = tuple(DEFAULT_GREP_EXCLUDE_GLOBS) + tuple(exclude or [])

    try:
        root = _to_workspace_path(workspace, path)
    except Exception as exc:
        return f"Tool error ({tool_name}): {exc}"

    if not root.exists():
        return f"Tool error ({tool_name}): Path does not exist: {path}"

    try:
        compiled = _compile_grep_pattern(pattern, regex=regex, case_sensitive=case_sensitive)
    except ValueError as exc:
        return f"Tool error ({tool_name}): {exc}"

    files = _iter_searchable_files(workspace, root, include_patterns, exclude_patterns)
    results: list[str] = []
    content_match_count = 0
    filename_match_count = 0
    truncated = False

    for file_path in files:
        rel = _rel_posix(workspace, file_path)

        # When enabled, this helper can also act as a lightweight filename
        # search. Current-file MAS wrappers disable this and search content only.
        if search_filenames and compiled.search(rel):
            filename_match_count += 1
            results.append(f"[filename] {rel}")
            if len(results) >= max_matches:
                truncated = True
                break

        try:
            if file_path.stat().st_size > MAX_GREP_FILE_BYTES:
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            continue

        if "\x00" in text[:4096]:
            continue

        lines = text.splitlines()
        matching_line_indexes = [idx for idx, line in enumerate(lines) if compiled.search(line)]
        for idx in matching_line_indexes:
            content_match_count += 1
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            results.append(f"[content] {rel}:{idx + 1}")
            for line_idx in range(start, end):
                marker = ">" if line_idx == idx else " "
                # Trim very long lines. Agents need location and surrounding text,
                # not megabytes of minified content.
                line = lines[line_idx]
                if len(line) > 300:
                    line = line[:300] + " ...[line truncated]"
                results.append(f"{marker} {line_idx + 1}: {line}")
            results.append("")
            if len(results) >= max_matches:
                truncated = True
                break

        if truncated:
            break

    if not results:
        searched = _rel_posix(workspace, root) if root != workspace.resolve() else "."
        return f"No matches found for {pattern!r} under {searched}."

    if search_filenames:
        header = (
            f"Found {content_match_count} content match(es) and "
            f"{filename_match_count} filename match(es) for {pattern!r}."
        )
    else:
        header = f"Found {content_match_count} content match(es) for {pattern!r}."
    if truncated:
        header += f" Showing first {max_matches} result entries. Narrow path/include or increase max_matches."

    output = header + "\n\n" + "\n".join(results).rstrip()
    return _truncate_raw_output(output, MAX_GREP_OUTPUT_CHARS)


class SearchCurrentFileArgs(BaseModel):
    pattern: str = Field(
        ...,
        description="Text or regex to search for within the current target file.",
    )
    regex: bool = Field(False, description="Treat pattern as a Python regular expression instead of literal text.")
    case_sensitive: bool = Field(False, description="Whether matching should be case-sensitive.")
    context: int = Field(2, description="Number of context lines before and after each content match. Clamped to 0..10.")
    max_matches: int = Field(50, description="Maximum result entries to return. Clamped to 1..200.")


class ReadCurrentFileArgs(BaseModel):
    pass


class EditReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oldText: str = Field(
        ...,
        description=(
            "The exact text currently present in the current target file. "
            "This must match the file contents character-for-character, including whitespace and indentation."
        ),
    )
    newText: str = Field(
        ...,
        description="The replacement text to write in place of oldText.",
    )


class EditCurrentFileArgs(BaseModel):
    edits: list[EditReplacement] = Field(
        ...,
        description=(
            "Exact replacements to apply immediately to the current target file. "
            "Each edit must contain oldText and newText. oldText must be an exact substring of the current file."
        ),
    )


class WriteCurrentFileArgs(BaseModel):
    content: str = Field(
        ...,
        description="Complete replacement contents for the current target file.",
    )


def _render_content(content: Any) -> str:
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

            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                parts.append(json.dumps(model_dump(mode="json"), ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _json_type_to_python(field_schema: dict[str, Any]) -> Any:
    field_type = field_schema.get("type")
    if field_type == "string":
        return str
    if field_type == "integer":
        return int
    if field_type == "number":
        return float
    if field_type == "boolean":
        return bool
    if field_type == "array":
        return list[Any]
    if field_type == "object":
        return dict[str, Any]
    return Any


def _schema_to_annotation(name: str, schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")

    if schema_type == "object":
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        nested_fields: dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            field_description = field_schema.get("description") or ""
            field_annotation = _schema_to_annotation(
                f"{name}_{field_name.title().replace('_', '')}",
                field_schema,
            )

            if field_name in required:
                nested_fields[field_name] = (
                    field_annotation,
                    Field(..., description=field_description),
                )
            else:
                default_value = field_schema.get("default", None)
                nested_fields[field_name] = (
                    field_annotation,
                    Field(default_value, description=field_description),
                )

        model_name = f"{name.title().replace('_', '')}Args"
        return create_model(model_name, **nested_fields)

    if schema_type == "array":
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            item_annotation = _schema_to_annotation(f"{name}Item", items_schema)
        else:
            item_annotation = Any
        return list[item_annotation]

    return _json_type_to_python(schema)


def _build_args_schema(name: str, input_schema: dict[str, Any] | None) -> type[BaseModel]:
    schema = input_schema or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    field_definitions: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = _schema_to_annotation(f"{name}_{field_name.title().replace('_', '')}", field_schema)
        description = field_schema.get("description") or ""

        if field_name in required:
            default = Field(..., description=description)
            annotation = field_type
        else:
            default_value = field_schema.get("default", None)
            default = Field(default_value, description=description)
            annotation = field_type

        field_definitions[field_name] = (annotation, default)

    model_name = f"{name.title().replace('_', '')}Args"
    model = create_model(model_name, __config__=ConfigDict(extra="forbid"), **field_definitions,)
    return cast(type[BaseModel], model)


def _format_tool_description(tool_name: str, description: str | None, input_schema: dict[str, Any] | None) -> str:
    summary = description or f"MCP tool: {tool_name}"
    if tool_name == "edit_file":
        summary += " This edits the current target file by applying exact replacements."
    if tool_name == "write_file":
        summary += " This overwrites the current target file with complete replacement content."
    if not input_schema:
        return summary

    schema_text = json.dumps(input_schema, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{summary}\n\nInput schema:\n{schema_text}"


def _truncate_raw_output(output: str, max_chars: int = MAX_RAW_OUTPUT_CHARS) -> str:
    if len(output) <= max_chars:
        return output

    truncated_chars = len(output) - max_chars
    return output[:max_chars] + f"\n...[truncated {truncated_chars} chars]"


def _log_tool_call(tool_name: str, arguments: dict[str, Any]) -> None:
    print(f"\n[Tool call] {tool_name}", flush=True)
    if arguments:
        print(json.dumps(arguments, indent=2, ensure_ascii=False), flush=True)


def _log_tool_result(tool_name: str, result: Any) -> None:
    if isinstance(result, str):
        rendered = result
    else:
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(f"[Tool result] {tool_name}", flush=True)
    print(_truncate_raw_output(rendered, MAX_CONSOLE_TOOL_OUTPUT_CHARS), flush=True)


def _ensure_resource_module_for_windows() -> None:
    if os.name == "nt" and "resource" not in sys.modules:
        sys.modules["resource"] = types.ModuleType("resource")


def _enable_utf8_text_io_for_windows() -> tuple[Any, Any] | None:
    if os.name != "nt":
        return None

    original_open = builtins.open
    original_read_text = _PathType.read_text

    def _open_utf8(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
            if errors is None:
                errors = "replace"
        return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    def _read_text_utf8(self, encoding=None, errors=None):
        if encoding is None:
            encoding = "utf-8"
            if errors is None:
                errors = "replace"
        return original_read_text(self, encoding=encoding, errors=errors)

    builtins.open = _open_utf8
    _PathType.read_text = _read_text_utf8
    return original_open, original_read_text


def _restore_text_io_patches(originals: tuple[Any, Any] | None) -> None:
    if os.name != "nt" or originals is None:
        return
    original_open, original_read_text = originals
    builtins.open = original_open
    _PathType.read_text = original_read_text


def _count_test_status_entries(tests_status: dict[str, Any], bucket: str) -> int:
    total = 0
    for status_name in ["FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL"]:
        status_data = tests_status.get(status_name, {})
        values = status_data.get(bucket, [])
        if isinstance(values, list):
            total += len(values)
    return total


def _quixbugs_test_name_for_program(program_file: str) -> str:
    return f"test_{Path(program_file).stem}.py"


def _parse_pytest_count(output: str, label: str) -> int:
    matches = re.findall(rf"(\d+)\s+{re.escape(label)}\b", output)
    return sum(int(value) for value in matches)


PYTEST_REPORTER = r'''
from __future__ import annotations

import json
import os


_reports = {}


def _safe_text(value):
    text = str(value or "")
    max_chars = 3000
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


def pytest_runtest_logreport(report):
    if report.when != "call":
        return

    entry = {
        "test": report.nodeid,
        "outcome": report.outcome,
        "duration": report.duration,
    }
    if report.failed:
        entry["failure"] = _safe_text(report.longrepr)
    _reports[report.nodeid] = entry


def pytest_sessionfinish(session, exitstatus):
    report_path = os.environ.get("QUIXBUGS_PYTEST_REPORT")
    if not report_path:
        return

    tests = list(_reports.values())
    payload = {
        "exitstatus": int(exitstatus),
        "passed": [item["test"] for item in tests if item["outcome"] == "passed"],
        "failed": [
            {
                "test": item["test"],
                "failure": item.get("failure", ""),
                "duration": item.get("duration", 0.0),
            }
            for item in tests
            if item["outcome"] == "failed"
        ],
        "skipped": [item["test"] for item in tests if item["outcome"] == "skipped"],
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
'''


def _build_test_summary(
    report: dict[str, Any],
    *,
    raw_output: str,
    returncode: int | None,
    tests: list[str],
) -> dict[str, Any]:
    passed_tests = list(report.get("passed") or [])
    failed_tests = list(report.get("failed") or [])
    skipped_tests = list(report.get("skipped") or [])

    if returncode != 0 and not failed_tests:
        failed_tests = [
            {
                "test": "pytest",
                "failure": _truncate_raw_output(raw_output, MAX_TOOL_OUTPUT_CHARS),
                "duration": 0.0,
            }
        ]

    summary: dict[str, Any] = {
        "passed": len(passed_tests),
        "failed": len(failed_tests),
        "skipped": len(skipped_tests),
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "skipped_tests": skipped_tests,
        "returncode": returncode,
        "tests": tests or ["all"],
    }
    if failed_tests:
        summary["raw_output"] = _truncate_raw_output(raw_output, MAX_TOOL_OUTPUT_CHARS)
    return summary


def _prepare_quixbugs_test_root(workspace: Path, temp_root: Path) -> None:
    shutil.copytree(workspace, temp_root / "python_programs")
    shutil.copytree(QUIXBUGS_TESTCASES, temp_root / "python_testcases")
    shutil.copytree(QUIXBUGS_JSON_TESTCASES, temp_root / "json_testcases")
    shutil.copy2(QUIXBUGS_CONFTEST, temp_root / "conftest.py")


def run_quixbugs_tests(
    workspace: Path,
    *,
    target_file: str | None = None,
    target_files: Sequence[str] | None = None,
    timeout: int = DEFAULT_QUIXBUGS_TEST_TIMEOUT,
) -> dict[str, Any]:
    """Run QuixBugs pytest tests against one prepared workspace."""

    tests: list[str]
    if target_files:
        tests = [_quixbugs_test_name_for_program(file_name) for file_name in target_files]
    elif target_file:
        tests = [_quixbugs_test_name_for_program(target_file)]
    else:
        tests = []

    with tempfile.TemporaryDirectory(prefix="quixbugs_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        _prepare_quixbugs_test_root(workspace, temp_root)
        reporter_path = temp_root / "quixbugs_pytest_reporter.py"
        report_path = temp_root / "quixbugs_pytest_report.json"
        reporter_path.write_text(PYTEST_REPORTER, encoding="utf-8")

        test_args = [str(Path("python_testcases") / test_name) for test_name in tests]
        if not test_args:
            test_args = ["python_testcases"]

        missing_tests = [arg for arg in test_args if not (temp_root / arg).exists()]
        if missing_tests:
            raw_output = f"Missing QuixBugs test file(s): {', '.join(missing_tests)}"
            return {
                "passed": 0,
                "failed": len(missing_tests),
                "failed_tests": [{"test": test_name, "error": raw_output} for test_name in missing_tests],
                "raw_output": raw_output,
            }

        env = os.environ.copy()
        pythonpath_entries = [str(temp_root), str(temp_root / "python_testcases")]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env["QUIXBUGS_PYTEST_REPORT"] = str(report_path)

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "quixbugs_pytest_reporter", *test_args],
                cwd=temp_root,
                timeout=timeout,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            raw_output = stdout + stderr + f"\nTimed out after {timeout} seconds."
            return {
                "passed": _parse_pytest_count(raw_output, "passed"),
                "failed": 1,
                "failed_tests": [{"test": "pytest", "error": f"Timed out after {timeout} seconds."}],
                "raw_output": _truncate_raw_output(raw_output, MAX_TOOL_OUTPUT_CHARS),
                "returncode": None,
                "tests": tests or ["all"],
            }

        raw_output = (completed.stdout or "") + (completed.stderr or "")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return _build_test_summary(
                report,
                raw_output=raw_output,
                returncode=completed.returncode,
                tests=tests,
            )

    failed = _parse_pytest_count(raw_output, "failed")
    errors = _parse_pytest_count(raw_output, "error") + _parse_pytest_count(raw_output, "errors")
    passed = _parse_pytest_count(raw_output, "passed")
    if completed.returncode != 0 and failed == 0 and errors == 0:
        failed = 1

    return {
        "passed": passed,
        "failed": failed + errors,
        "failed_tests": [],
        "passed_tests": [],
        "skipped_tests": [],
        "raw_output": _truncate_raw_output(raw_output, MAX_TOOL_OUTPUT_CHARS),
        "returncode": completed.returncode,
        "tests": tests or ["all"],
    }


class MCPRuntime:
    """Single MCP stdio runtime bound to one workspace mount."""

    def __init__(
        self,
        workspace: str | Path,
        instance: dict[str, Any] | None = None,
        target_file: str | None = None,
    ):
        self.workspace = _normalize_path(workspace)
        self.instance = instance or {}
        self.target_file = target_file
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP runtime is not started.")
        return self._session

    async def __aenter__(self) -> "MCPRuntime":
        mount = f"type=bind,src={self.workspace.as_posix()},dst={MCP_WORKSPACE_ROOT}"
        server = StdioServerParameters(
            command="docker",
            args=["run", "-i", "--rm", "--mount", mount, MCP_IMAGE, "/projects"],
        )

        stack = AsyncExitStack()
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        self._stack = stack
        self._session = session

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self.session.call_tool(name, arguments)
        rendered = _render_content(result.content)
        if result.isError:
            return f"Tool error ({name}): {rendered or 'unknown error'}"
        rendered = _truncate_raw_output(rendered or "[No tool output returned]", MAX_TOOL_OUTPUT_CHARS)
        return rendered

    def run_tests(self, target_file: str | None = None) -> dict[str, Any]:
        return run_quixbugs_tests(self.workspace, target_file=target_file or self.target_file)

    def _target_container_path(self) -> str:
        if not self.target_file:
            raise RuntimeError("No current target file is bound to this MCP runtime.")
        return _to_container_path(self.target_file)


async def mcp_tools_from_mcp(
    runtime: MCPRuntime,
    allowed_tool_names: Sequence[str] | None = None,
) -> list[StructuredTool]:
    """Create LangChain tools by lifting MCP tool metadata (name/schema/description)."""
    allowed_set = set(allowed_tool_names or DEFAULT_ALLOWED_TOOLS)
    listed_tools = (await runtime.session.list_tools()).tools

    langchain_tools: list[StructuredTool] = []

    if "read_file" in allowed_set:
        async def _read_current_file() -> str:
            arguments = {"path": runtime._target_container_path()}
            _log_tool_call("read_file", {})
            result = await runtime.call_tool("read_file", arguments)
            _log_tool_result("read_file", result)
            return result

        langchain_tools.append(
            StructuredTool.from_function(
                coroutine=_read_current_file,
                name="read_file",
                description="Read the current QuixBugs target file. Takes no arguments.",
                args_schema=ReadCurrentFileArgs,
                infer_schema=False,
            )
        )

    if "edit_file" in allowed_set:
        async def _edit_current_file(edits: list[EditReplacement]) -> str:
            edit_payload = [
                edit.model_dump(mode="json") if isinstance(edit, BaseModel) else edit
                for edit in edits
            ]
            public_arguments = {"edits": edit_payload}
            _log_tool_call("edit_file", public_arguments)
            result = await runtime.call_tool(
                "edit_file",
                {
                    "path": runtime._target_container_path(),
                    "edits": edit_payload,
                    "dryRun": False,
                },
            )
            _log_tool_result("edit_file", result)
            return result

        langchain_tools.append(
            StructuredTool.from_function(
                coroutine=_edit_current_file,
                name="edit_file",
                description=(
                    "Apply exact text replacements to the current QuixBugs target file immediately. "
                    "Do not provide a path. The edits argument is a list of objects with oldText and newText. "
                    "oldText must be an exact character-for-character match already present in the file, "
                    "including whitespace and indentation."
                ),
                args_schema=EditCurrentFileArgs,
                infer_schema=False,
            )
        )

    if "write_file" in allowed_set:
        async def _write_current_file(content: str) -> str:
            public_arguments = {"content": content}
            _log_tool_call("write_file", public_arguments)
            result = await runtime.call_tool(
                "write_file",
                {
                    "path": runtime._target_container_path(),
                    "content": content,
                },
            )
            _log_tool_result("write_file", result)
            return result

        langchain_tools.append(
            StructuredTool.from_function(
                coroutine=_write_current_file,
                name="write_file",
                description=(
                    "Overwrite the current QuixBugs target file with complete replacement content. "
                    "Prefer edit_file for small localized changes; use write_file "
                    "when replacing the whole file is clearer or simpler."
                ),
                args_schema=WriteCurrentFileArgs,
                infer_schema=False,
            )
        )

    for tool_def in listed_tools:
        if tool_def.name not in allowed_set:
            continue
        if tool_def.name in {"read_file", "edit_file", "write_file", "run_tests"}:
            continue

        args_schema = _build_args_schema(tool_def.name, tool_def.inputSchema)
        description = _format_tool_description(tool_def.name, tool_def.description, tool_def.inputSchema)

        async def _tool_coroutine(_tool_name: str = tool_def.name, **kwargs: Any) -> str:
            if "path" in kwargs and isinstance(kwargs["path"], str):
                kwargs["path"] = _to_container_path(kwargs["path"])
            _log_tool_call(_tool_name, kwargs)
            result = await runtime.call_tool(_tool_name, kwargs)
            _log_tool_result(_tool_name, result)
            return result

        langchain_tools.append(
            StructuredTool.from_function(
                coroutine=_tool_coroutine,
                name=tool_def.name,
                description=description,
                args_schema=args_schema,
                infer_schema=False,
            )
        )

    if "search_current_file" in allowed_set:
        def _search_current_file_tool(
            pattern: str = "",
            regex: bool = False,
            case_sensitive: bool = False,
            context: int = 2,
            max_matches: int = 50,
        ) -> str:
            return _search_files_impl(
                runtime.workspace,
                path=runtime.target_file or runtime._target_container_path(),
                pattern=pattern,
                regex=regex,
                case_sensitive=case_sensitive,
                context=context,
                max_matches=max_matches,
                search_filenames=False,
                tool_name="search_current_file",
            )

        langchain_tools.append(
            StructuredTool.from_function(
                func=_search_current_file_tool,
                name="search_current_file",
                description=(
                    "Search only inside the current QuixBugs target file and return matching line numbers "
                    "with surrounding context. This tool does not search for files and does not accept a path. "
                    "The pattern is literal text by default; set regex=true for a Python regex. "
                    "Use this when read_file would be too broad and you need to jump to a specific symbol, "
                    "condition, variable, or error-message fragment in the current file."
                ),
                args_schema=SearchCurrentFileArgs,
                infer_schema=False,
            )
        )

    if "run_tests" in allowed_set:
        def _run_tests_tool() -> dict[str, Any]:
            _log_tool_call("run_tests", {})
            result = runtime.run_tests()
            _log_tool_result("run_tests", result)
            return result

        langchain_tools.append(
            StructuredTool.from_function(
                func=_run_tests_tool,
                name="run_tests",
                description=(
                    "Run canonical QuixBugs pytest tests against a temporary copy of the edited workspace. "
                    "Takes no arguments and only runs the current target file's test."
                ),
                infer_schema=True,
            )
        )

    return langchain_tools


def start_mcp(
    workspace: str | Path,
    instance: dict[str, Any] | None = None,
    target_file: str | None = None,
) -> MCPRuntime:
    """Create an MCP runtime manager for a mounted workspace."""
    return MCPRuntime(workspace, instance=instance, target_file=target_file)
