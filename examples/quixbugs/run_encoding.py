"""Build the QuixBugs resource encoding map.

The QuixBugs SearchSpace uses discrete coordinates. This script orders LLMs,
toolsets, and prompts so nearby choices have a useful neighborhood structure
for BO.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


OUTPUT_PATH = Path(__file__).resolve().parent / "encoding_map.json"
MAX_TOOLS_PER_AGENT = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class CompiledResourceMap:
    """Ordered resource indices saved to encoding_map.json."""

    llm_order: list[int]
    toolset_order: list[list[int]]
    prompt_order: list[int]

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def compile_resource_map(
    llms: list[str],
    tools: list[str],
    system_prompts: list[str],
    max_tools_per_agent: int = 5,
    embedding_model: str = EMBEDDING_MODEL,
) -> CompiledResourceMap:
    """Compile ordered LLM, toolset, and prompt choices for the example."""

    llm_order = order_llms(llms)
    toolset_order = order_toolsets(tools, max_tools_per_agent=max_tools_per_agent)
    prompt_order = order_prompts(system_prompts, embedding_model=embedding_model)

    return CompiledResourceMap(
        llm_order=llm_order,
        toolset_order=toolset_order,
        prompt_order=prompt_order,
    )


# ---------------------------------------------------------------------
# LLM ordering
# ---------------------------------------------------------------------
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    total_params_b: float
    active_params_b: float
    context_tokens: int
    reasoning: float
    tool_calling: float


KNOWN_MODEL_SPECS: dict[str, ModelSpec] = {
    # Dense Qwen3-based Hermes model.
    # Hermes 4 14B supports hybrid reasoning and tool calls.
    "hermes-4-14b": ModelSpec(
        total_params_b=14.0,
        active_params_b=14.0,
        context_tokens=40_960,
        reasoning=0.85,
        tool_calling=0.8,
    ),

    # MoE: 30.5B total, 3.3B activated, 262K native context.
    # Coder-specialized, tool-call capable, but "non-thinking" mode only.
    "qwen/qwen3-coder-30b-a3b-instruct-fp8": ModelSpec(
        total_params_b=30.5,
        active_params_b=3.3,
        context_tokens=262_144,
        reasoning=0.70,
        tool_calling=0.9,
    ),

    # MoE: 117B total, 5.1B active, 131K context, strong reasoning/tool support.
    "openai/gpt-oss-120b": ModelSpec(
        total_params_b=117.0,
        active_params_b=5.1,
        context_tokens=131_072,
        reasoning=1.0,
        tool_calling=1.0,
    ),

    "openai/gpt-oss-20b": ModelSpec(
        total_params_b=21.0,
        active_params_b=3.6,
        context_tokens=131_072,
        reasoning=1.0,
        tool_calling=1.0,
    ),

    "google/gemma-3n-e4b-it": ModelSpec(
        total_params_b=8.0,
        active_params_b=4.0,
        context_tokens=32_000,
        reasoning=0.4,
        tool_calling=0.0,
    ),
}


def canonical_model_key(model_name: str) -> str:
    key = model_name.lower().strip()

    # Allow either "Hermes-4-14B" or "NousResearch/Hermes-4-14B".
    if key.endswith("hermes-4-14b") or key.endswith("hermes-4-14b-fp8"):
        return "hermes-4-14b"

    return key


def order_llms(llms: list[str], strongest_first: bool = False) -> list[int]:
    scored = []

    for i, llm in enumerate(llms):
        score = llm_bugsolving_score(llm)
        scored.append((score, i))

    scored.sort(reverse=strongest_first)
    return [i for _score, i in scored]


def llm_bugsolving_score(model_name: str) -> float:
    key = canonical_model_key(model_name)
    spec = KNOWN_MODEL_SPECS.get(key)

    if spec is None:
        total_params_b = parse_param_count_b(model_name)
        active_params_b = parse_active_param_count_b(model_name) or total_params_b
        context_tokens = 32_000
        reasoning = 0.5
        tool_calling = 0.0
    else:
        total_params_b = spec.total_params_b
        active_params_b = spec.active_params_b
        context_tokens = spec.context_tokens
        reasoning = spec.reasoning
        tool_calling = spec.tool_calling

    size_score = (
        0.5 * math.log1p(total_params_b)
        + 0.5 * math.log1p(active_params_b)
    )

    context_score = math.log1p(context_tokens / 1000.0)

    return (
        0.45 * size_score
        + 0.30 * context_score
        + 0.20 * reasoning
        + 0.05 * tool_calling
    )


def parse_param_count_b(model_name: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    if match:
        return float(match.group(1))
    return 7.0


def parse_active_param_count_b(model_name: str) -> float | None:
    # Handles names like "30B-A3B" as a fallback for MoE models.
    match = re.search(r"[-_/]a(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------
# Toolset ordering
# ---------------------------------------------------------------------

def order_toolsets(
    tools: list[str],
    max_tools_per_agent: int,
) -> list[list[int]]:
    n_tools = len(tools)
    max_size = min(max_tools_per_agent, n_tools)

    toolsets: list[tuple[int, ...]] = []

    for r in range(max_size + 1):
        for combo in combinations(range(n_tools), r):
            toolsets.append(combo)

    weights = np.array([tool_weight(t) for t in tools], dtype=float)
    D = toolset_distance_matrix(toolsets, weights)

    order = nearest_neighbor_2opt_order(D, start_index=0)

    return [list(toolsets[i]) for i in order]


def tool_weight(tool_name: str) -> float:
    t = tool_name.lower()

    if "test" in t or "run" in t or "execute" in t:
        return 1.4

    if "edit" in t or "write" in t or "patch" in t:
        return 1.3

    if "read" in t or "search" in t or "grep" in t:
        return 1.0

    return 1.0


def toolset_distance_matrix(
    toolsets: list[tuple[int, ...]],
    weights: np.ndarray,
) -> np.ndarray:
    n = len(toolsets)
    D = np.zeros((n, n), dtype=float)

    total_weight = float(weights.sum())

    sets = [set(s) for s in toolsets]

    for i in range(n):
        for j in range(i + 1, n):
            symdiff = sets[i].symmetric_difference(sets[j])
            d = sum(weights[k] for k in symdiff) / total_weight
            D[i, j] = d
            D[j, i] = d

    return D


# ---------------------------------------------------------------------
# Prompt ordering
# ---------------------------------------------------------------------

def order_prompts(
    prompts: list[str],
    embedding_model: str,
) -> list[int]:
    model = SentenceTransformer(embedding_model)

    embeddings = model.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=float)

    D = cosine_distance_matrix(embeddings)

    order = nearest_neighbor_2opt_order(D, start_index=0)

    return [int(i) for i in order]


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


# ---------------------------------------------------------------------
# Generic ordering helper
# ---------------------------------------------------------------------

def nearest_neighbor_2opt_order(
    D: np.ndarray,
    start_index: int = 0,
) -> list[int]:
    n = D.shape[0]

    unvisited = set(range(n))
    order = [start_index]
    unvisited.remove(start_index)

    while unvisited:
        last = order[-1]
        next_i = min(unvisited, key=lambda j: (D[last, j], j))
        order.append(next_i)
        unvisited.remove(next_i)

    order = two_opt(order, D)
    return order


def two_opt(order: list[int], D: np.ndarray) -> list[int]:
    order = list(order)
    n = len(order)

    improved = True

    while improved:
        improved = False

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                old = (
                    D[order[i - 1], order[i]]
                    + D[order[j], order[j + 1]]
                )

                new = (
                    D[order[i - 1], order[j]]
                    + D[order[i], order[j + 1]]
                )

                if new < old:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True

    return order


if __name__ == "__main__":
    from .resource_statement import LLMS, SYSTEM_PROMPTS, TOOLS

    compiled = compile_resource_map(
        llms=LLMS,
        tools=TOOLS,
        system_prompts=SYSTEM_PROMPTS,
        max_tools_per_agent=MAX_TOOLS_PER_AGENT,
        embedding_model=EMBEDDING_MODEL,
    )

    compiled.save(OUTPUT_PATH)

    print(f"Saved compiled resource map to {OUTPUT_PATH}")

    print("\nLLM choices:")
    for choice, original_idx in enumerate(compiled.llm_order):
        print(f"{choice:>3} -> LLMS[{original_idx}] = {LLMS[original_idx]}")

    print("\nToolset choices:")
    for choice, tool_indices in enumerate(compiled.toolset_order):
        names = [TOOLS[i] for i in tool_indices]
        print(f"{choice:>3} -> {names}")

    print("\nPrompt choices:")
    for choice, original_idx in enumerate(compiled.prompt_order):
        preview = SYSTEM_PROMPTS[original_idx].replace("\n", " ")[:80]
        print(f"{choice:>3} -> SYSTEM_PROMPTS[{original_idx}] = {preview!r}...")
