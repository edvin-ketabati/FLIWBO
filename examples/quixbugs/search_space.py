"""Choice-size helper for the QuixBugs MAS design vector."""

from __future__ import annotations

from pathlib import Path

from fliwbo_core.decoder import ResourceDecoder

from .resource_statement import (
    LLMS,
    MAX_NUMBER_OF_AGENTS,
    NUMBER_OF_FEATURES_PER_AGENT,
    SYSTEM_PROMPTS,
    TOOLS,
)


QUIXBUGS_DIR = Path(__file__).resolve().parent
ENCODING_MAP_PATH = QUIXBUGS_DIR / "encoding_map.json"


def get_default_choice_sizes() -> list[int]:
    """Return the per-coordinate number of choices for the 20-value MAS vector."""

    decoder = ResourceDecoder.from_json(
        ENCODING_MAP_PATH,
        LLMS,
        TOOLS,
        SYSTEM_PROMPTS,
    )

    per_agent_choice_sizes = [
        decoder.n_llm_choices,
        decoder.n_toolset_choices,
        decoder.n_prompt_choices,
        MAX_NUMBER_OF_AGENTS + 1,
    ]

    if len(per_agent_choice_sizes) != NUMBER_OF_FEATURES_PER_AGENT:
        raise ValueError(
            "NUMBER_OF_FEATURES_PER_AGENT must match the configured MAS feature layout"
        )

    return per_agent_choice_sizes * MAX_NUMBER_OF_AGENTS
