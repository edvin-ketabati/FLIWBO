"""Build a runnable multi-agent system from examples/quixbugs/mas_spec.json.

This file is example plumbing. It reads the decoded MAS spec, creates LangChain
agents with the selected models/tools/prompts, and runs them in the configured
handoff order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

try:
    from .resource_statement import MAX_AGENT_TRANSITIONS  # type: ignore
except ImportError:
    try:
        from examples.quixbugs.resource_statement import MAX_AGENT_TRANSITIONS  # type: ignore
    except ImportError:
        MAX_AGENT_TRANSITIONS = 6

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAS_SPEC_PATH = Path(__file__).resolve().parent / "mas_spec.json"


@dataclass(frozen=True)
class AgentSpec:
    """One decoded agent entry from mas_spec.json."""

    id: int
    active: bool
    model: str
    system_prompt: str
    tools: tuple[str, ...]
    next1: int | None


@dataclass(frozen=True)
class MASSpec:
    """Complete decoded MAS specification."""

    agents: tuple[AgentSpec, ...]


def _get_together_api_key() -> str:
    return os.getenv("together_api_key") or os.getenv("TOGETHER_API_KEY") or ""


def _build_chat_model(model: str, temperature: float) -> ChatOpenAI:
    api_key = _get_together_api_key()
    if not api_key:
        raise ValueError("Missing Together API key. Set together_api_key or TOGETHER_API_KEY.")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key="dummy-key",
        base_url=BASE_URL + model + "/v1",
        max_completion_tokens=5000,
    )


def _load_agent_spec(raw_agent: dict[str, Any]) -> AgentSpec:
    agent_id = raw_agent.get("id")
    if not isinstance(agent_id, int):
        raise ValueError(f"Agent id must be an integer, got: {agent_id!r}")

    active = raw_agent.get("active", True)
    if not isinstance(active, bool):
        raise ValueError(f"Agent {agent_id} active flag must be boolean, got: {active!r}")

    model = raw_agent.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Agent {agent_id} must define a non-empty model")

    system_prompt = raw_agent.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(f"Agent {agent_id} must define a non-empty system_prompt")

    tools = raw_agent.get("tools", [])
    if not isinstance(tools, list) or not all(isinstance(tool_name, str) and tool_name for tool_name in tools):
        raise ValueError(f"Agent {agent_id} tools must be a list of tool names")

    def _load_next(value: Any) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int):
            raise ValueError(f"Agent {agent_id} next1 must be an integer or null")
        return value

    return AgentSpec(
        id=agent_id,
        active=active,
        model=model,
        system_prompt=system_prompt,
        tools=tuple(tools),
        next1=_load_next(raw_agent.get("next1")),
    )


def load_mas_spec(spec_path: Path | None = None) -> MASSpec:
    """Load and validate the MAS spec JSON used by the runtime builder."""

    resolved_path = spec_path or DEFAULT_MAS_SPEC_PATH
    raw_spec = json.loads(resolved_path.read_text(encoding="utf-8"))
    raw_agents = raw_spec.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError("MAS spec must define a non-empty 'agents' list")

    agents = tuple(_load_agent_spec(raw_agent) for raw_agent in raw_agents)
    agent_ids = [agent.id for agent in agents]
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("MAS spec agent ids must be unique")

    return MASSpec(agents=agents)


def _build_tool_map(tools: list[Any]) -> dict[str, Any]:
    tool_map: dict[str, Any] = {}
    for tool in tools:
        tool_name = getattr(tool, "name", None)
        if isinstance(tool_name, str) and tool_name:
            tool_map[tool_name] = tool
    return tool_map


def _infer_start_agent_id(spec: MASSpec) -> int:
    active_agents = {agent.id: agent for agent in spec.agents if agent.active}
    if not active_agents:
        raise ValueError("MAS spec does not contain any active agents")

    inbound_counts = {agent_id: 0 for agent_id in active_agents}
    for agent in active_agents.values():
        if agent.next1 in inbound_counts:
            inbound_counts[agent.next1] += 1

    start_candidates = [agent_id for agent_id, count in inbound_counts.items() if count == 0]
    if start_candidates:
        return min(start_candidates)
    return min(active_agents)


def _next_agent_id(spec: MASSpec, current_agent_id: int) -> int | None:
    active_agents = {agent.id: agent for agent in spec.agents if agent.active}
    current_agent = active_agents.get(current_agent_id)
    if current_agent is None:
        return None

    if current_agent.next1 in active_agents:
        return current_agent.next1
    return None


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
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
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


def _last_message_content_as_text(messages: list[Any]) -> str:
    if not messages:
        return ""

    last_message = _message_to_dict(messages[-1])
    return _content_to_text(last_message.get("content", "")).strip()


def _append_agent_handoff(
    state_messages: list[Any],
    agent_messages: list[Any],
) -> list[Any]:
    produced_messages = agent_messages[len(state_messages):]
    handoff_text = _last_message_content_as_text(produced_messages)
    return [*state_messages, AIMessage(content=handoff_text)]


class SpecDrivenMultiAgentSystem:
    """Small sequential handoff runner for the decoded active agents."""

    def __init__(
        self,
        spec: MASSpec,
        agents: dict[int, Any],
        *,
        max_transitions: int = MAX_AGENT_TRANSITIONS,
    ):
        self._spec = spec
        self._agents = agents
        self._max_transitions = max_transitions
        self._start_agent_id = _infer_start_agent_id(spec)
        self._cumulative_token_usage: dict[str, int] = {"total_tokens": 0}

    def get_cumulative_token_usage(self) -> dict[str, int]:
        """Return the cumulative token usage across all astream runs on this system."""
        return dict(self._cumulative_token_usage)

    async def astream(
        self,
        inputs: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        stream_mode: str = "updates",
    ):
        messages = list(inputs.get("messages", []))
        current_agent_id = self._start_agent_id
        transition_count = 0

        while current_agent_id is not None and transition_count < self._max_transitions:
            agent = self._agents[current_agent_id]
            agent_messages = list(messages)
            try:
                # Keep the agent's internal transcript private to this node. The MAS
                # state only receives one AIMessage handoff after the node finishes.
                async for chunk in agent.astream({"messages": messages}, config=config, stream_mode=stream_mode):
                    if isinstance(chunk, dict) and "messages" in chunk:
                        agent_messages = chunk["messages"]
                        yield {
                            "messages": messages,
                            "agent_messages": agent_messages,
                            "new_agent_messages": agent_messages,
                            "current_agent_id": current_agent_id,
                            "transition_count": transition_count,
                            "agent_chunk": chunk,
                        }
                        continue

                    if isinstance(chunk, dict):
                        for node_name, node_state in chunk.items():
                            if isinstance(node_state, dict) and "messages" in node_state:
                                new_agent_messages = node_state["messages"]
                                agent_messages = [*agent_messages, *new_agent_messages]
                                yield {
                                    "messages": messages,
                                    "agent_messages": agent_messages,
                                    "new_agent_messages": new_agent_messages,
                                    "current_agent_id": current_agent_id,
                                    "transition_count": transition_count,
                                    "inner_node": node_name,
                                    "agent_chunk": chunk,
                                }
                                break
                
                # If we reach here, agent completed successfully
                # Extract token usage from the final message if available
                if agent_messages:
                    last_message = agent_messages[-1]
                    response_metadata = last_message.get("response_metadata") if isinstance(last_message, dict) else getattr(last_message, "response_metadata", None)
                    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}
                    self._cumulative_token_usage["total_tokens"] += int(token_usage.get("total_tokens") or 0)
            except Exception as exc:
                # If agent hit recursion limit, gracefully move to next agent instead of failing
                exc_text = str(exc or "").lower()
                if "recursion limit" in exc_text or "recursion_limit" in exc_text or "recursionlimit" in exc_text:
                    completed_agent_id = current_agent_id
                    messages = _append_agent_handoff(messages, agent_messages)
                    transition_count += 1
                    current_agent_id = _next_agent_id(self._spec, current_agent_id)
                    yield {
                        "messages": messages,
                        "agent_messages": agent_messages,
                        "new_agent_messages": [],
                        "current_agent_id": current_agent_id,
                        "transition_count": transition_count,
                        "hit_agent_recursion_limit": True,
                        "completed_agent_id": completed_agent_id,
                    }
                    continue
                else:
                    # Re-raise non-recursion-limit exceptions
                    raise

            messages = _append_agent_handoff(messages, agent_messages)
            transition_count += 1
            yield {
                "messages": messages,
                "agent_messages": agent_messages,
                "new_agent_messages": [],
                "current_agent_id": current_agent_id,
                "transition_count": transition_count,
            }

            current_agent_id = _next_agent_id(self._spec, current_agent_id)

        if current_agent_id is not None:
            yield {
                "messages": messages,
                "current_agent_id": current_agent_id,
                "transition_count": transition_count,
                "hit_max_transitions": True,
                "max_transitions": self._max_transitions,
            }


def build_mas_system(
    tools: list[Any],
    *,
    spec_path: Path | None = None,
    model_override: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> SpecDrivenMultiAgentSystem:
    """Create the runnable MAS from a spec path and loaded tool objects."""

    spec = load_mas_spec(spec_path)
    tool_map = _build_tool_map(tools)
    agents: dict[int, Any] = {}

    for agent_spec in spec.agents:
        if not agent_spec.active:
            continue

        selected_tools = []
        for tool_name in agent_spec.tools:
            tool = tool_map.get(tool_name)
            if tool is None:
                raise ValueError(f"Tool {tool_name!r} for agent {agent_spec.id} was not loaded")
            selected_tools.append(tool)

        llm = _build_chat_model(model_override or agent_spec.model, temperature)
        agents[agent_spec.id] = create_agent(
            model=llm,
            tools=selected_tools,
            system_prompt=agent_spec.system_prompt,
        )

    if not agents:
        raise ValueError("MAS spec does not define any active agents")

    return SpecDrivenMultiAgentSystem(spec, agents)
