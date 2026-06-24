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


# ---------------------------------------------------------------------------
# Conversational turn classifier
# ---------------------------------------------------------------------------
#
# The previous router matched keywords on the raw sentence, which mislabels real
# conversation: "not deep screen, find a web dev" matched "deep screen" and ran a
# screen; "what was my first message" had no home and ran a resume search. This
# classifier asks the LLM to read the message IN CONTEXT and return one label,
# with deterministic guardrails for cases that can't be wrong (greetings, done)
# and a keyword fallback if the LLM output is unusable.

TURN_INTENTS: Tuple[str, ...] = (
    "search", "refine", "compare", "interview", "screen", "explain", "chat", "done",
)

_CHITCHAT = {
    "hi", "hello", "hey", "yo", "sup", "hiya", "heya", "hi there", "hello there",
    "thanks", "thank you", "ty", "thx", "cheers", "ok", "okay", "cool", "nice",
    "great", "test", "ping", "help", "?", "hey there", "good morning",
    "good evening", "gm",
}
_DONE_WORDS = {"done", "stop", "end", "quit", "exit", "bye", "goodbye"}
_DONE_PHRASES = (
    "that's all", "thats all", "that is all", "no thanks", "no, thanks",
    "i'm done", "im done", "we're done", "were done", "nothing else",
    "all done", "that's it", "thats it",
)

_TURN_SYSTEM = (
    "You route one turn of a recruiting assistant conversation. Read the latest "
    "user message in the context of the conversation, then reply with a JSON "
    'object {"intent": "<label>"} using EXACTLY ONE allowed label.\n'
    "Definitions:\n"
    "- search: the user describes a NEW role or candidate type to find — e.g. "
    "'i need ai developers', 'now find me a web developer', 'someone with a "
    "masters in AI'. Phrases that REJECT a prior action still count as search: "
    "'not a deep screen, I want a web developer' is search.\n"
    "- refine: adjust the CURRENT ranking — re-weight factors or tweak the same "
    "role, e.g. 'weight experience higher', 'prioritise React'.\n"
    "- compare: compare/contrast specific already-shortlisted candidates.\n"
    "- interview: generate interview or screening questions for a candidate.\n"
    "- screen: run a deep multi-round screen / hire-vs-no-hire analysis of the "
    "shortlist. ONLY when the user explicitly asks to screen/deep-dive — never "
    "when they are asking for a different kind of candidate.\n"
    "- explain: explain WHY the ranking is as it is, or why one candidate ranks "
    "above another.\n"
    "- chat: greetings, thanks, small talk, or questions about the conversation "
    "itself or your capabilities — e.g. 'what was my first message', 'what can "
    "you do'.\n"
    "- done: the user wants to finish.\n"
    "Reply with ONLY the JSON object, nothing else."
)


def _history_text(history: Optional[Sequence[object]], limit: int = 12) -> str:
    """Render recent messages (dicts or LangChain message objects) as a transcript."""
    lines: List[str] = []
    for m in list(history or [])[-limit:]:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role = getattr(m, "type", "user")
            content = getattr(m, "content", "")
        text = str(content).replace("\n", " ").strip()
        if text:
            lines.append(f"{role}: {text[:200]}")
    return "\n".join(lines)


def _parse_intent(raw: str, allowed: Sequence[str]) -> Optional[str]:
    """Accept a JSON object {"intent": ...} OR a bare label word."""
    import json

    txt = (raw or "").strip()
    if not txt:
        return None
    match = re.search(r"\{.*\}", txt, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            cand = str(obj.get("intent", "")).strip().lower()
            if cand in allowed:
                return cand
        except (ValueError, AttributeError):
            pass
    low = txt.lower()
    for label in allowed:
        if re.search(rf"\b{re.escape(label)}\b", low):
            return label
    return None


def _keyword_fallback(low: str) -> str:
    """Last-resort routing if the LLM output can't be parsed. Negation-aware."""
    negated = any(n in low for n in ("not ", "no ", "instead", "rather than", "don't"))
    if any(p in low for p in ("compare", " vs ", "versus", "side by side", "side-by-side")):
        return "compare"
    if "interview" in low or "questions for" in low:
        return "interview"
    if (("deep" in low and "screen" in low) or "hire/no" in low
            or "hire or no" in low) and not negated:
        return "screen"
    if any(p in low for p in ("why did", "why is", "why does", "why ", "explain")):
        return "explain"
    if any(p in low for p in ("what ", "who ", "when ", "how ", "can you",
                              "first message", "you say", "did you", "your name")):
        return "chat"
    return "refine"


def classify_turn(
    llm: LLMClient,
    user_message: str,
    *,
    has_shortlist: bool = True,
    history: Optional[Sequence[object]] = None,
) -> str:
    """Classify a conversational turn into one routing intent.

    'search' is folded into 'refine' for routing (the extract node decides whether
    a turn is a brand-new search or a re-weight of the current one). Commands that
    need a shortlist degrade to 'chat' when none exists yet.
    """
    msg = (user_message or "").strip()
    low = msg.lower()
    if not low:
        return "done"
    norm = low.strip(" .!?,")
    if norm in _CHITCHAT:
        return "chat"
    if norm in _DONE_WORDS or any(p in low for p in _DONE_PHRASES):
        return "done"

    convo = _history_text(history)
    prompt = (
        (f"Conversation so far:\n{convo}\n\n" if convo else "")
        + f"Allowed labels: {', '.join(TURN_INTENTS)}\n"
        + f"Shortlist exists: {'yes' if has_shortlist else 'no'}\n"
        + f"Latest user message: {msg!r}\n"
        + 'Reply as JSON: {"intent": "<label>"}'
    )
    try:
        raw = llm.complete(_TURN_SYSTEM, prompt)
    except Exception:  # noqa: BLE001 — never let a routing call crash the graph
        raw = ""
    intent = _parse_intent(raw, TURN_INTENTS) or _keyword_fallback(low)

    if intent == "search":
        intent = "refine"
    if not has_shortlist and intent in ("compare", "interview", "screen", "explain"):
        intent = "chat"
    return intent
