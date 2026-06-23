"""agent_llm.py — injectable LLM abstraction for the matching agent.

A tiny Protocol (`LLMClient`) decouples the LangGraph nodes from any specific
SDK. `AnthropicLLM` wraps the existing `anthropic` SDK (already a project
dependency); `StubLLM` makes every node testable offline with no API key.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, Union


class LLMClient(Protocol):
    """Anything that can turn a (system, prompt) pair into text."""

    def complete(self, system: str, prompt: str) -> str: ...


class StubLLM:
    """Deterministic offline LLM for tests.

    Args:
        responses: either a list of strings (returned FIFO) or a callable
            ``handler(system, prompt) -> str``.
    """

    name = "stub"

    def __init__(self, responses: Union[List[str], Callable[[str, str], str]]) -> None:
        self._responses = responses
        self._index = 0
        self.calls: List[Tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if callable(self._responses):
            return self._responses(system, prompt)
        if self._index >= len(self._responses):
            raise AssertionError("StubLLM ran out of scripted responses")
        out = self._responses[self._index]
        self._index += 1
        return out


class AnthropicLLM:
    """Adapter over the existing `anthropic` SDK (reused, not langchain)."""

    def __init__(self, client: Optional[object] = None, model: Optional[str] = None) -> None:
        self._client = client
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.name = f"anthropic:{self.model}"

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            from anthropic import Anthropic
            self._client = Anthropic()
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        client = self._ensure_client()
        resp = client.messages.create(  # type: ignore[union-attr]
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", None) == "text"
        ).strip()


_INTENT_SYSTEM = (
    "You are an intent router for a recruiting assistant. Reply with EXACTLY ONE "
    "label from the allowed list and nothing else."
)


def classify_intent(
    llm: LLMClient,
    user_message: str,
    allowed: Sequence[str],
    default: str = "done",
) -> str:
    """Map a free-text follow-up to one of *allowed* labels (default on miss)."""
    prompt = (
        f"Allowed labels: {', '.join(allowed)}\n"
        f"User message: {user_message!r}\n"
        "Label:"
    )
    raw = llm.complete(_INTENT_SYSTEM, prompt).strip().lower()
    for label in allowed:
        # Word-boundary match so 'redefine' doesn't match 'refine', etc.
        if re.search(rf"\b{re.escape(label.lower())}\b", raw):
            return label
    return default


def narrate(llm: LLMClient, system: str, prompt: str) -> str:
    """Thin pass-through for prose generation (keeps call sites uniform)."""
    return llm.complete(system, prompt)
