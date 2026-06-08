"""Decode ordered resource maps into user-facing choices.

This helper is generic: it does not know about QuixBugs. The QuixBugs example
uses it to map compact integer choices back to LLM names, toolsets, and prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceDecoder:
    """Decode integer choices through an encoding_map.json resource ordering."""

    llms: list[str]
    tools: list[str]
    system_prompts: list[str]

    llm_order: list[int]
    toolset_order: list[list[int]]
    prompt_order: list[int]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        llms: list[str],
        tools: list[str],
        system_prompts: list[str],
    ) -> "ResourceDecoder":
        """Load an encoding map and attach it to concrete resource lists."""

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            llms=llms,
            tools=tools,
            system_prompts=system_prompts,
            llm_order=data["llm_order"],
            toolset_order=data["toolset_order"],
            prompt_order=data["prompt_order"],
        )

    @property
    def n_llm_choices(self) -> int:
        return len(self.llm_order)

    @property
    def n_toolset_choices(self) -> int:
        return len(self.toolset_order)

    @property
    def n_prompt_choices(self) -> int:
        return len(self.prompt_order)

    def decode_llm(self, choice: int) -> str:
        original_idx = self.llm_order[int(choice)]
        return self.llms[original_idx]

    def decode_toolset(self, choice: int) -> list[str]:
        tool_indices = self.toolset_order[int(choice)]
        return [self.tools[i] for i in tool_indices]

    def decode_prompt(self, choice: int) -> str:
        original_idx = self.prompt_order[int(choice)]
        return self.system_prompts[original_idx]

    def decode_agent(
        self,
        llm_choice: int,
        toolset_choice: int,
        prompt_choice: int,
    ) -> dict:
        return {
            "llm": self.decode_llm(llm_choice),
            "tools": self.decode_toolset(toolset_choice),
            "system_prompt": self.decode_prompt(prompt_choice),
        }
