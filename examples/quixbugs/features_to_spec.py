"""Convert QuixBugs MAS feature vectors into executable MAS specs.

The optimizer proposes compact integer vectors. This module decodes those
integers into model names, toolsets, prompts, and agent handoff edges, then
writes examples/quixbugs/mas_spec.json for the runtime builder.
"""

import json
from pathlib import Path

from fliwbo_core.decoder import ResourceDecoder

from .resource_statement import (
    LLMS,
    MAX_NUMBER_OF_AGENTS,
    NUMBER_OF_FEATURES_PER_AGENT,
    TOOLS,
    SYSTEM_PROMPTS
)

QUIXBUGS_DIR = Path(__file__).resolve().parent
ENCODING_MAP_PATH = QUIXBUGS_DIR / "encoding_map.json"
SPEC_PATH = QUIXBUGS_DIR / "mas_spec.json"

decoder = ResourceDecoder.from_json(
    ENCODING_MAP_PATH,
    LLMS,
    TOOLS,
    SYSTEM_PROMPTS,
)

bounds = {
    "llm_choice": (0, decoder.n_llm_choices - 1),
    "toolset_choice": (0, decoder.n_toolset_choices - 1),
    "prompt_choice": (0, decoder.n_prompt_choices - 1),
}



def features_to_spec(feature_vector, spec_path: str | Path = SPEC_PATH):
    """
    Convert a flattened feature vector to a mas_spec.json structure and write it to disk.
    
    Args:
        feature_vector: list of NUMBER_OF_FEATURES_PER_AGENT*MAX_NUMBER_OF_AGENTS numeric values
                       Each NUMBER_OF_FEATURES_PER_AGENT-value chunk defines one agent:
                       [llm_choice, toolset_choice, prompt_choice, next_agent_idx, ...]
    """
    if len(feature_vector) != NUMBER_OF_FEATURES_PER_AGENT * MAX_NUMBER_OF_AGENTS:
        raise ValueError(
            f"Feature vector must have {NUMBER_OF_FEATURES_PER_AGENT * MAX_NUMBER_OF_AGENTS} values, got {len(feature_vector)}"
        )
    
    # Parse agents from feature vector
    agents = []
    for agent_idx in range(MAX_NUMBER_OF_AGENTS):
        base = agent_idx * NUMBER_OF_FEATURES_PER_AGENT
        llm_choice = _clamp_choice(
            feature_vector[base + 0],
            decoder.n_llm_choices,
            "llm_choice",
        )
        toolset_choice = _clamp_choice(
            feature_vector[base + 1],
            decoder.n_toolset_choices,
            "toolset_choice",
        )
        prompt_choice = _clamp_choice(
            feature_vector[base + 2],
            decoder.n_prompt_choices,
            "prompt_choice",
        )
        next_agent = int(feature_vector[base + 3])
        
        agent_id = agent_idx + 1

        decoded_agent = decoder.decode_agent(
            llm_choice=llm_choice,
            toolset_choice=toolset_choice,
            prompt_choice=prompt_choice,
        )
        
        # Decode next agent: 0 means no next agent (null)
        next1 = None if next_agent == 0 else max(1, min(next_agent, MAX_NUMBER_OF_AGENTS))
        
        agent = {
            "id": agent_id,
            "active": True,  # Will be updated below
            "model": decoded_agent["llm"],
            "system_prompt": decoded_agent["system_prompt"],
            "tools": decoded_agent["tools"],
            "next1": next1,
        }
        agents.append(agent)
    
    active_agent_ids = _reachable_agent_ids_from_start(agents)
    for agent in agents:
        agent["active"] = agent["id"] in active_agent_ids
    
    spec = {"agents": agents}

    resolved_spec_path = Path(spec_path)
    resolved_spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    return spec


def _clamp_choice(value, n_choices: int, name: str) -> int:
    if n_choices <= 0:
        raise ValueError(f"{name} has no available choices")

    choice = int(value)
    return max(0, min(choice, n_choices - 1))


def _reachable_agent_ids_from_start(agents: list[dict]) -> set[int]:
    agent_by_id = {agent["id"]: agent for agent in agents}
    connected_ids = _connected_agent_ids(agents)
    if not connected_ids:
        connected_ids = {min(agent_by_id)}

    start_id = _infer_start_agent_id(agents, connected_ids)

    reachable_ids = set()
    current_id = start_id
    while current_id is not None and current_id in agent_by_id and current_id not in reachable_ids:
        reachable_ids.add(current_id)
        current_id = agent_by_id[current_id]["next1"]

    return reachable_ids


def _connected_agent_ids(agents: list[dict]) -> set[int]:
    connected_ids = set()
    agent_ids = {agent["id"] for agent in agents}

    for agent in agents:
        next_id = agent["next1"]
        if next_id is None:
            continue

        connected_ids.add(agent["id"])
        if next_id in agent_ids:
            connected_ids.add(next_id)

    return connected_ids


def _infer_start_agent_id(agents: list[dict], active_ids: set[int]) -> int:
    inbound_counts = {agent_id: 0 for agent_id in active_ids}

    for agent in agents:
        agent_id = agent["id"]
        next_id = agent["next1"]
        if agent_id in active_ids and next_id in inbound_counts:
            inbound_counts[next_id] += 1

    start_candidates = [
        agent_id for agent_id, count in inbound_counts.items()
        if count == 0
    ]
    if start_candidates:
        return min(start_candidates)
    return min(active_ids)
