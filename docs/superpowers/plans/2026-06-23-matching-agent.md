# Matching Agent (Milestone 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph conversational matching agent that orchestrates the existing M1 file tools and M2 RAG/hybrid matcher into an interactive, multi-round, explainable recruiting assistant.

**Architecture:** A deterministic LangGraph funnel (`parse_jd → extract_requirements → search_resumes → rank_candidates → summarize_shortlist → generate_report → human_feedback`) wraps the deterministic `JobMatcher`/`ResumeRAG` core. An injectable LLM is used only for intent routing, prose, and interview questions. A hard invariant keeps scoring deterministic and the LLM explanatory: LLM nodes receive pre-computed scores/excerpts, never raw resumes, and never reorder the shortlist. The `human_feedback` node uses `interrupt()` for human-in-the-loop and routes follow-ups to refine / compare / interview / screen / done.

**Tech Stack:** Python 3.14, LangGraph, the existing `anthropic` SDK (reused — no `langchain-anthropic`), ChromaDB + MiniLM (existing), pytest.

## Global Constraints

- Python ≥ 3.12; the project runs on 3.14. Use `from __future__ import annotations`, PEP 8, type annotations on every signature.
- Run tests with `python -m pytest` (a bare `pytest` runs a stale 3.12 shim) — verbatim.
- Reuse the M1/M2 deterministic core; never reimplement matching, scoring, embedding, or extraction.
- **Scoring/ordering invariant:** `shortlist` order is written only by deterministic nodes (`rank_candidates`, `multi_round_screen`). LLM nodes may add prose only; they must never reorder or rescore.
- LLM client is injectable via an `LLMClient` Protocol; all tests use `StubLLM` — no network, no `ANTHROPIC_API_KEY`.
- Structured errors, never tracebacks to the UI (mirror `fs_tools`/`JobMatcher`).
- New runtime dependency: `langgraph` only. Keep files focused (< 400 lines target; `matching_agent.py` ≤ 800 hard cap).
- Immutability: nodes return new state dicts; never mutate inputs in place.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `agent_llm.py` (new) | `LLMClient` Protocol, `StubLLM`, `AnthropicLLM` adapter (existing `anthropic` SDK), `classify_intent()`, `narrate()`. |
| `agent_tools.py` (new) | `rag_search()`, `extract_requirements()`, `compare_candidates()`, `generate_interview_questions()`. Pure, unit-testable. |
| `matching_agent.py` (new) | `AgentState`, all graph nodes, the router, `build_agent()`, `MatchingAgent` wrapper, `--anonymize` + decision log helpers. |
| `agent_cli.py` (new) | Streaming conversational CLI. |
| `app.py` (modify) | Add an "Agent Chat" tab driving the compiled graph. |
| `tests/test_agent_llm.py` (new) | StubLLM + helper tests. |
| `tests/test_agent_tools.py` (new) | The three tools + rag_search. |
| `tests/test_matching_agent.py` (new) | 5+ conversation-flow scenarios + invariant. |
| `docs/diagrams/matching_agent_state_machine.md` + `.png` (new) | Mermaid source + rendered PNG. |
| `requirements.txt` (modify) | Add `langgraph`. |
| `README.md` (modify) | Milestone 3 section + Explainability & Compliance note. |

---

## Task 1: LLM abstraction (`agent_llm.py`)

**Files:**
- Create: `agent_llm.py`
- Test: `tests/test_agent_llm.py`

**Interfaces:**
- Produces:
  - `class LLMClient(Protocol): def complete(self, system: str, prompt: str) -> str: ...`
  - `class StubLLM:` constructed with `StubLLM(responses)` where `responses` is `list[str]` (popped FIFO) **or** `Callable[[str, str], str]` (called as `handler(system, prompt)`); has `.calls: list[tuple[str, str]]`.
  - `class AnthropicLLM:` `AnthropicLLM(client=None, model=None)`; `.complete(system, prompt) -> str`.
  - `def classify_intent(llm: LLMClient, user_message: str, allowed: Sequence[str], default: str = "done") -> str`
  - `def narrate(llm: LLMClient, system: str, prompt: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_llm.py
from __future__ import annotations

import pytest

from agent_llm import StubLLM, classify_intent, narrate


def test_stub_list_pops_in_order():
    llm = StubLLM(["first", "second"])
    assert llm.complete("sys", "a") == "first"
    assert llm.complete("sys", "b") == "second"
    assert llm.calls == [("sys", "a"), ("sys", "b")]


def test_stub_callable_handler():
    llm = StubLLM(lambda system, prompt: f"{system}|{prompt}")
    assert llm.complete("S", "P") == "S|P"


def test_classify_intent_matches_allowed_label():
    llm = StubLLM(["  COMPARE  "])
    out = classify_intent(llm, "compare the top 3", ["refine", "compare", "done"])
    assert out == "compare"


def test_classify_intent_falls_back_to_default_on_unknown():
    llm = StubLLM(["banana"])
    out = classify_intent(llm, "???", ["refine", "compare"], default="done")
    assert out == "done"


def test_narrate_passes_through():
    llm = StubLLM(["a sentence"])
    assert narrate(llm, "sys", "prompt") == "a sentence"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_llm'`.

- [ ] **Step 3: Write minimal implementation**

```python
# agent_llm.py
"""agent_llm.py — injectable LLM abstraction for the matching agent.

A tiny Protocol (`LLMClient`) decouples the LangGraph nodes from any specific
SDK. `AnthropicLLM` wraps the existing `anthropic` SDK (already a project
dependency); `StubLLM` makes every node testable offline with no API key.
"""

from __future__ import annotations

import os
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
        if label.lower() in raw:
            return label
    return default


def narrate(llm: LLMClient, system: str, prompt: str) -> str:
    """Thin pass-through for prose generation (keeps call sites uniform)."""
    return llm.complete(system, prompt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_llm.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_llm.py tests/test_agent_llm.py
git commit -m "feat: injectable LLM abstraction (Protocol, StubLLM, Anthropic adapter)"
```

---

## Task 2: `extract_requirements` + `rag_search` tools (`agent_tools.py`)

**Files:**
- Create: `agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `parse_job_description` (job_matcher), `ResumeRAG.query` (resume_rag).
- Produces:
  - `def extract_requirements(jd: str) -> dict` returning keys `title, must_haves (list[str]), nice_to_have (list[str]), required_skills (list[str]), weights (dict)`.
  - `DEFAULT_WEIGHTS: dict` = `{"retrieval": 0.45, "skills": 0.35, "experience": 0.15, "nice": 0.05}`.
  - `def rag_search(rag, text: str, k: int = 10) -> list[dict]` returning `[{candidate, file, section, similarity, text}]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_tools.py
from __future__ import annotations

from agent_tools import DEFAULT_WEIGHTS, extract_requirements, rag_search
from tests.rag_test_utils import make_corpus

ML_JD = """Job Title: Machine Learning Engineer

Requirements:
- 5+ years of professional Python experience.
- Experience with PyTorch and pandas.
- Bachelor's degree in Computer Science.

Nice to have:
- AWS experience.
"""


def test_extract_requirements_splits_must_and_nice():
    req = extract_requirements(ML_JD)
    assert req["title"] == "Machine Learning Engineer"
    assert "Python" in req["required_skills"] and "PyTorch" in req["required_skills"]
    assert req["nice_to_have"] == ["AWS"]
    assert any("5+" in mh or "Python" in mh for mh in req["must_haves"])
    assert req["weights"] == DEFAULT_WEIGHTS


def test_rag_search_returns_ranked_hits(tmp_path, monkeypatch):
    rag = make_corpus(tmp_path, monkeypatch)
    rag.build_index()
    hits = rag_search(rag, "machine learning engineer with PyTorch", k=3)
    assert hits and isinstance(hits[0], dict)
    assert {"candidate", "file", "section", "similarity", "text"} <= hits[0].keys()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_tools'`.

- [ ] **Step 3: Write minimal implementation**

```python
# agent_tools.py
"""agent_tools.py — the three Milestone-3 agent tools plus a RAG search wrapper.

Pure functions over the M2 deterministic core so they are unit-testable and
reusable both inside the LangGraph nodes and from the CLI. None of these
functions reorder or invent match scores — numbers come from JobMatcher.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from job_matcher import parse_job_description

DEFAULT_WEIGHTS: Dict[str, float] = {
    "retrieval": 0.45,
    "skills": 0.35,
    "experience": 0.15,
    "nice": 0.05,
}


def extract_requirements(jd: str) -> Dict[str, object]:
    """Parse a JD into a must-have vs nice-to-have requirements dict."""
    parsed = parse_job_description(jd)
    return {
        "title": parsed.title,
        "must_haves": [mh.raw for mh in parsed.must_haves],
        "nice_to_have": list(parsed.nice_to_have),
        "required_skills": list(parsed.required_skills),
        "weights": dict(DEFAULT_WEIGHTS),
    }


def rag_search(rag: object, text: str, k: int = 10) -> List[Dict[str, object]]:
    """Semantic search wrapper returning plain dicts (decoupled from ChunkHit)."""
    hits = rag.query(text, k=k)  # type: ignore[attr-defined]
    return [
        {
            "candidate": h.candidate,
            "file": h.file,
            "section": h.section,
            "similarity": h.similarity,
            "text": h.text,
        }
        for h in hits
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_tools.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_tools.py tests/test_agent_tools.py
git commit -m "feat: extract_requirements + rag_search agent tools"
```

---

## Task 3: `compare_candidates` tool

**Files:**
- Modify: `agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Produces: `def compare_candidates(candidate_ids: Sequence[str], shortlist: Sequence[dict]) -> dict`.
  - Each shortlist item is a `CandidateRecord`-shaped dict with keys `name, resume_path, score, matched_skills, breakdown` and optional profile keys `exp_years, education_level`.
  - Returns `{"candidates": [...resolved records...], "dimensions": {dim: winner_name}, "ranking": [names high→low], "errors": [unknown ids]}`.
  - `dimensions` covers `"score"`, `"skills"` (most matched), `"experience"` (highest `exp_years`).
  - Resolution matches an id against `name` (case-insensitive) or `resume_path`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_agent_tools.py
from agent_tools import compare_candidates

_SHORTLIST = [
    {"name": "Riley Carter", "resume_path": "resumes/riley.txt", "score": 88,
     "matched_skills": ["Python", "PyTorch", "pandas"], "breakdown": {},
     "exp_years": 8, "education_level": "master"},
    {"name": "Jordan Blake", "resume_path": "resumes/jordan.txt", "score": 61,
     "matched_skills": ["Python"], "breakdown": {},
     "exp_years": 2, "education_level": "bachelor"},
]


def test_compare_candidates_picks_per_dimension_winners():
    out = compare_candidates(["Riley Carter", "Jordan Blake"], _SHORTLIST)
    assert out["errors"] == []
    assert out["ranking"] == ["Riley Carter", "Jordan Blake"]
    assert out["dimensions"]["score"] == "Riley Carter"
    assert out["dimensions"]["skills"] == "Riley Carter"
    assert out["dimensions"]["experience"] == "Riley Carter"
    assert len(out["candidates"]) == 2


def test_compare_candidates_reports_unknown_ids():
    out = compare_candidates(["Nobody"], _SHORTLIST)
    assert out["errors"] == ["Nobody"]
    assert out["candidates"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_tools.py -k compare -v`
Expected: FAIL with `ImportError: cannot import name 'compare_candidates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to agent_tools.py

def _resolve(candidate_id: str, shortlist: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    cid = candidate_id.strip().lower()
    for rec in shortlist:
        if str(rec.get("name", "")).lower() == cid or str(rec.get("resume_path", "")).lower() == cid:
            return rec
    return None


def compare_candidates(
    candidate_ids: Sequence[str],
    shortlist: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Head-to-head comparison; winners per dimension are deterministic (no LLM)."""
    resolved: List[Dict[str, object]] = []
    errors: List[str] = []
    for cid in candidate_ids:
        rec = _resolve(cid, shortlist)
        (resolved if rec else errors).append(rec if rec else cid)  # type: ignore[arg-type]

    if not resolved:
        return {"candidates": [], "dimensions": {}, "ranking": [], "errors": errors}

    def _best(key, transform=lambda r: r) -> str:
        winner = max(resolved, key=lambda r: transform(r.get(key, 0)))
        return str(winner.get("name", ""))

    dimensions = {
        "score": _best("score"),
        "skills": max(resolved, key=lambda r: len(r.get("matched_skills") or [])).get("name", ""),
        "experience": _best("exp_years"),
    }
    ranking = [str(r.get("name", "")) for r in sorted(
        resolved, key=lambda r: -int(r.get("score", 0) or 0))]
    return {
        "candidates": resolved,
        "dimensions": dimensions,
        "ranking": ranking,
        "errors": errors,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_tools.py -k compare -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_tools.py tests/test_agent_tools.py
git commit -m "feat: compare_candidates head-to-head tool"
```

---

## Task 4: `generate_interview_questions` tool

**Files:**
- Modify: `agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `LLMClient` (agent_llm), `narrate`.
- Produces: `def generate_interview_questions(candidate_id: str, requirements: dict, shortlist: Sequence[dict], llm) -> dict` returning `{"candidate": name, "gaps": list[str], "questions": list[str], "error": Optional[str]}`.
  - Gaps = `required_skills` minus the candidate's `matched_skills`.
  - The LLM returns one question per line; parse into a list. Unknown id → `{"error": ...}` and no LLM call.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_agent_tools.py
from agent_llm import StubLLM
from agent_tools import generate_interview_questions

_REQ = {"title": "ML Engineer", "required_skills": ["Python", "PyTorch", "Docker"]}


def test_interview_questions_grounded_in_gaps():
    llm = StubLLM(["1. How have you used Docker?\n2. Describe a PyTorch project."])
    out = generate_interview_questions("Jordan Blake", _REQ, _SHORTLIST, llm)
    assert out["error"] is None
    assert "Docker" in out["gaps"] and "PyTorch" in out["gaps"]
    assert len(out["questions"]) == 2
    # The candidate's gaps must appear in the prompt sent to the LLM.
    assert "Docker" in llm.calls[0][1]


def test_interview_questions_unknown_candidate_no_llm_call():
    llm = StubLLM([])
    out = generate_interview_questions("Nobody", _REQ, _SHORTLIST, llm)
    assert out["error"] is not None
    assert out["questions"] == [] and llm.calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_tools.py -k interview -v`
Expected: FAIL with `ImportError: cannot import name 'generate_interview_questions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to agent_tools.py (add this import near the top of the file)
# from agent_llm import LLMClient, narrate

_INTERVIEW_SYSTEM = (
    "You are a technical recruiter. Write concise, specific screening questions "
    "that probe a candidate's gaps against a role. One question per line, no preamble."
)


def generate_interview_questions(
    candidate_id: str,
    requirements: Dict[str, object],
    shortlist: Sequence[Dict[str, object]],
    llm: "LLMClient",
) -> Dict[str, object]:
    """Generate gap-grounded screening questions for one candidate."""
    from agent_llm import narrate  # local import keeps agent_tools import-light

    rec = _resolve(candidate_id, shortlist)
    if rec is None:
        return {"candidate": candidate_id, "gaps": [], "questions": [],
                "error": f"unknown candidate: {candidate_id}"}

    required = list(requirements.get("required_skills") or [])
    matched = set(rec.get("matched_skills") or [])
    gaps = [s for s in required if s not in matched]

    prompt = (
        f"Role: {requirements.get('title', 'the role')}\n"
        f"Candidate: {rec.get('name')}\n"
        f"Matched skills: {', '.join(sorted(matched)) or 'none'}\n"
        f"Gaps to probe: {', '.join(gaps) or 'none — probe depth on matched skills'}\n"
        "Write 4-6 screening questions."
    )
    raw = narrate(llm, _INTERVIEW_SYSTEM, prompt)
    questions = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return {"candidate": rec.get("name"), "gaps": gaps, "questions": questions, "error": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_tools.py -k interview -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agent_tools.py tests/test_agent_tools.py
git commit -m "feat: generate_interview_questions gap-grounded tool"
```

---

## Task 5: Agent state + deterministic nodes (`matching_agent.py`)

**Files:**
- Create: `matching_agent.py`
- Test: `tests/test_matching_agent.py`

**Interfaces:**
- Consumes: `JobMatcher` (job_matcher), `extract_requirements` (agent_tools).
- Produces:
  - `AgentState` TypedDict (keys from spec §5).
  - `class Engine:` `Engine(matcher: JobMatcher, llm: LLMClient)` — holds the deterministic matcher + LLM, passed to nodes via a closure factory.
  - Node functions (each `(state: dict) -> dict` partial-state update), created by `make_nodes(engine) -> dict[str, Callable]`:
    - `parse_jd`, `extract_requirements_node`, `search_resumes`, `rank_candidates`.
  - `rank_candidates` writes `shortlist` as a list of `CandidateRecord` dicts: `{name, resume_path, score, breakdown, matched_skills, excerpts, reasoning}` and snapshots prior `shortlist` into `prev_shortlist`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_matching_agent.py
from __future__ import annotations

from agent_llm import StubLLM
from job_matcher import JobMatcher
from matching_agent import Engine, make_nodes
from tests.rag_test_utils import make_corpus

ML_JD = """Job Title: Machine Learning Engineer

Requirements:
- 5+ years of professional Python experience.
- Experience with PyTorch and pandas.

Nice to have:
- AWS experience.
"""


def _engine(tmp_path, monkeypatch, llm=None):
    rag = make_corpus(tmp_path, monkeypatch)
    rag.build_index()
    return Engine(matcher=JobMatcher(rag=rag), llm=llm or StubLLM([]))


def test_deterministic_nodes_produce_ranked_shortlist(tmp_path, monkeypatch):
    nodes = make_nodes(_engine(tmp_path, monkeypatch))
    state = {"jd_text": ML_JD, "k": 5, "messages": []}
    state.update(nodes["parse_jd"](state))
    state.update(nodes["extract_requirements"](state))
    state.update(nodes["search_resumes"](state))
    state.update(nodes["rank_candidates"](state))

    sl = state["shortlist"]
    assert sl, "expected at least one ranked candidate"
    assert sl == sorted(sl, key=lambda r: -r["score"]), "shortlist must be score-desc"
    assert {"name", "resume_path", "score", "breakdown", "matched_skills",
            "excerpts", "reasoning"} <= sl[0].keys()
    assert state["requirements"]["title"] == "Machine Learning Engineer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matching_agent'`.

- [ ] **Step 3: Write minimal implementation**

```python
# matching_agent.py
"""matching_agent.py — LangGraph conversational resume-matching agent (M3).

Wraps the deterministic M1/M2 core in a human-in-the-loop graph. The LLM never
ranks: deterministic nodes own `shortlist` ordering; LLM nodes add prose only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from agent_llm import LLMClient
from agent_tools import extract_requirements as _extract_requirements
from job_matcher import JobMatcher


@dataclass
class Engine:
    """Carries the deterministic matcher and the injectable LLM into nodes."""

    matcher: JobMatcher
    llm: LLMClient


def _records_from_match(result: Dict[str, object]) -> List[Dict[str, object]]:
    """Convert JobMatcher.match() top_matches into CandidateRecord dicts."""
    records: List[Dict[str, object]] = []
    for m in result.get("top_matches", []):  # type: ignore[union-attr]
        records.append({
            "name": m["candidate_name"],
            "resume_path": m["resume_path"],
            "score": int(m["match_score"]),
            "breakdown": m["score_breakdown"],
            "matched_skills": list(m["matched_skills"]),
            "excerpts": list(m["relevant_excerpts"]),
            "reasoning": m["reasoning"],
        })
    return records


def make_nodes(engine: Engine) -> Dict[str, Callable[[dict], dict]]:
    """Build the node functions bound to *engine* (closures over matcher/llm)."""

    def parse_jd(state: dict) -> dict:
        jd = state.get("jd_text", "")
        if not jd or not jd.strip():
            raise ValueError("no job description provided")
        return {"jd_text": jd}

    def extract_requirements_node(state: dict) -> dict:
        req = _extract_requirements(state["jd_text"])
        # Preserve user-adjusted weights across a refine loop.
        existing = state.get("requirements") or {}
        if existing.get("weights"):
            req["weights"] = existing["weights"]
        return {"requirements": req}

    def search_resumes(state: dict) -> dict:
        # Retrieval is handled inside JobMatcher.match; this node records intent
        # and keeps the graph shape faithful to the assignment diagram.
        return {"messages": []}

    def rank_candidates(state: dict) -> dict:
        weights = (state.get("requirements") or {}).get("weights") or {}
        sem = weights.get("retrieval")
        result = engine.matcher.match(state["jd_text"], k=int(state.get("k", 10)))
        records = _records_from_match(result)
        out: dict = {"shortlist": records}
        if state.get("shortlist"):
            out["prev_shortlist"] = state["shortlist"]
        return out

    return {
        "parse_jd": parse_jd,
        "extract_requirements": extract_requirements_node,
        "search_resumes": search_resumes,
        "rank_candidates": rank_candidates,
    }
```

> Note: weight wiring into `JobMatcher` happens in Task 7 (refine path). This task keeps `rank_candidates` calling `match()` with defaults; the closure already reads `weights` so Task 7 only adds the plumbing.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add matching_agent.py tests/test_matching_agent.py
git commit -m "feat: agent state + deterministic ranking nodes"
```

---

## Task 6: Summarizer + report nodes (LLM, invariant-safe)

**Files:**
- Modify: `matching_agent.py`
- Test: `tests/test_matching_agent.py`

**Interfaces:**
- Produces (added to `make_nodes`):
  - `summarize_shortlist` — for each record adds `record["summary"]` (strengths/gaps; improvement suggestion if `45 <= score <= 60`). Receives breakdown + excerpts + matched_skills only.
  - `generate_report` — sets `state["report"]` (markdown). On refine (when `prev_shortlist` present) appends a ranking-delta section.
  - Module constant `BORDERLINE = (45, 60)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py

def test_summarize_and_report_do_not_reorder(tmp_path, monkeypatch):
    llm = StubLLM(lambda system, prompt: "Strength: Python. Gap: Docker.")
    nodes = make_nodes(_engine(tmp_path, monkeypatch, llm))
    state = {"jd_text": ML_JD, "k": 5, "messages": []}
    for name in ("parse_jd", "extract_requirements", "search_resumes", "rank_candidates"):
        state.update(nodes[name](state))
    order_before = [r["name"] for r in state["shortlist"]]

    state.update(nodes["summarize_shortlist"](state))
    state.update(nodes["generate_report"](state))

    assert [r["name"] for r in state["shortlist"]] == order_before  # invariant
    assert all("summary" in r for r in state["shortlist"])
    assert isinstance(state["report"], str) and state["report"].strip()
    # LLM never saw raw resume file content (only excerpts/skills).
    assert all("EXPERIENCE\n" not in call[1] or "[" in call[1] for call in llm.calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k summarize -v`
Expected: FAIL with `KeyError: 'summarize_shortlist'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add near the top of matching_agent.py
from agent_llm import narrate

BORDERLINE = (45, 60)

_SUMMARY_SYSTEM = (
    "You summarize a candidate's fit from PRE-COMPUTED evidence only. You are "
    "given a match score, a score breakdown, matched skills, and short resume "
    "excerpts. State strengths and gaps in 2-3 sentences. If asked, add one "
    "improvement suggestion. Never invent facts beyond the evidence; never "
    "restate or change the numeric score."
)


def _summary_prompt(record: dict, requirements: dict, suggest: bool) -> str:
    return (
        f"Role: {requirements.get('title', 'the role')}\n"
        f"Required skills: {', '.join(requirements.get('required_skills') or [])}\n"
        f"Candidate: {record['name']} (score {record['score']}/100)\n"
        f"Breakdown: {record['breakdown']}\n"
        f"Matched skills: {', '.join(record['matched_skills']) or 'none'}\n"
        f"Excerpts: {' / '.join(record['excerpts'])}\n"
        + ("Also give one concrete improvement suggestion." if suggest else "")
    )
```

```python
# inside make_nodes(), add these two nodes and include them in the returned dict

    def summarize_shortlist(state: dict) -> dict:
        req = state.get("requirements") or {}
        updated: List[dict] = []
        for rec in state.get("shortlist", []):
            suggest = BORDERLINE[0] <= int(rec["score"]) <= BORDERLINE[1]
            summary = narrate(engine.llm, _SUMMARY_SYSTEM,
                              _summary_prompt(rec, req, suggest))
            updated.append({**rec, "summary": summary})
        return {"shortlist": updated}

    def generate_report(state: dict) -> dict:
        req = state.get("requirements") or {}
        lines = [f"# Match report — {req.get('title', 'role')}", ""]
        for i, rec in enumerate(state.get("shortlist", []), 1):
            lines.append(f"## {i}. {rec['name']} — {rec['score']}/100")
            lines.append(rec.get("summary", rec.get("reasoning", "")))
            lines.append("")
        prev = {r["name"]: r["score"] for r in state.get("prev_shortlist") or []}
        if prev:
            lines.append("## Ranking changes")
            for rec in state.get("shortlist", []):
                old = prev.get(rec["name"])
                if old is not None and old != rec["score"]:
                    arrow = "▲" if rec["score"] > old else "▼"
                    lines.append(f"- {rec['name']}: {old} → {rec['score']} {arrow}")
        return {"report": "\n".join(lines)}
```

Add `"summarize_shortlist": summarize_shortlist, "generate_report": generate_report` to the returned dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -k summarize -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add matching_agent.py tests/test_matching_agent.py
git commit -m "feat: summarizer + report nodes (deterministic-order invariant held)"
```

---

## Task 7: Build the graph + HITL router (`build_agent`)

**Files:**
- Modify: `matching_agent.py`, `requirements.txt`
- Test: `tests/test_matching_agent.py`

**Interfaces:**
- Consumes: `langgraph.graph.StateGraph/START/END`, `langgraph.checkpoint.memory.MemorySaver`, `langgraph.types.interrupt/Command`, `classify_intent` (agent_llm).
- Produces:
  - `AgentState` (TypedDict with `add_messages` reducer on `messages`).
  - `def build_agent(engine: Engine, checkpointer=None)` → compiled graph.
  - `human_feedback` node calls `interrupt(...)`, classifies the resumed message into `refine|compare|interview|screen|done`, writes `last_intent`.
  - `route_after_feedback(state) -> str` returns the next node or `END`.
  - `INTENTS = ("refine", "compare", "interview", "screen", "done")`.
  - Wiring `rank_candidates` to honor `weights["retrieval"]` via `JobMatcher.match(..., )` using a fresh matcher semantic weight when provided (see code).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py
from langgraph.types import Command
from matching_agent import build_agent

def _run_first_pass(engine, thread="t1"):
    graph = build_agent(engine)
    cfg = {"configurable": {"thread_id": thread}}
    state = graph.invoke({"jd_text": ML_JD, "k": 5, "messages": []}, cfg)
    return graph, cfg, state


def test_first_pass_interrupts_with_report(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, StubLLM(lambda s, p: "ok"))
    _, _, state = _run_first_pass(engine)
    assert state["shortlist"]
    assert "__interrupt__" in state  # paused at human_feedback


def test_done_intent_ends_graph(tmp_path, monkeypatch):
    # First narration calls return "ok"; the router classifies "done".
    engine = _engine(tmp_path, monkeypatch, StubLLM(lambda s, p: "ok" if "label" not in p.lower() else "done"))
    graph, cfg, _ = _run_first_pass(engine)
    final = graph.invoke(Command(resume="thanks, that's all"), cfg)
    assert final.get("last_intent") == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k "first_pass or done_intent" -v`
Expected: FAIL with `ImportError: cannot import name 'build_agent'` (and/or `No module named 'langgraph'`).

- [ ] **Step 3: Write minimal implementation**

First add the dependency:

```bash
python -m pip install langgraph
```

Append `langgraph` to `requirements.txt`.

```python
# add to matching_agent.py imports
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agent_llm import classify_intent

INTENTS = ("refine", "compare", "interview", "screen", "done")


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    jd_text: str
    requirements: dict
    shortlist: list
    prev_shortlist: list
    screening: dict
    last_intent: str
    report: str
    k: int


def route_after_feedback(state: dict) -> str:
    intent = state.get("last_intent", "done")
    return END if intent == "done" else intent
```

Add the `human_feedback` node inside `make_nodes`:

```python
    def human_feedback(state: dict) -> dict:
        user_message = interrupt({"report": state.get("report", ""),
                                  "prompt": "What would you like to do next?"})
        intent = classify_intent(engine.llm, str(user_message), INTENTS, default="done")
        return {"last_intent": intent,
                "messages": [{"role": "user", "content": str(user_message)}]}
```

Add it to the returned node dict, then add `build_agent`:

```python
def build_agent(engine: Engine, checkpointer: Optional[object] = None):
    """Assemble and compile the LangGraph state machine."""
    nodes = make_nodes(engine)
    g = StateGraph(AgentState)
    for name in ("parse_jd", "extract_requirements", "search_resumes",
                 "rank_candidates", "summarize_shortlist", "generate_report",
                 "human_feedback"):
        g.add_node(name, nodes[name])

    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "extract_requirements")
    g.add_edge("extract_requirements", "search_resumes")
    g.add_edge("search_resumes", "rank_candidates")
    g.add_edge("rank_candidates", "summarize_shortlist")
    g.add_edge("summarize_shortlist", "generate_report")
    g.add_edge("generate_report", "human_feedback")
    g.add_conditional_edges("human_feedback", route_after_feedback, {
        "refine": "extract_requirements",
        "compare": "human_feedback",     # replaced in Task 8
        "interview": "human_feedback",   # replaced in Task 8
        "screen": "human_feedback",      # replaced in Task 9
        END: END,
    })
    return g.compile(checkpointer=checkpointer or MemorySaver())
```

Also update `rank_candidates` to honor an adjusted retrieval weight:

```python
    def rank_candidates(state: dict) -> dict:
        weights = (state.get("requirements") or {}).get("weights") or {}
        sem = weights.get("retrieval")
        matcher = engine.matcher
        if sem is not None:
            from job_matcher import JobMatcher as _JM
            matcher = _JM(rag=engine.matcher.rag, semantic_weight=float(sem))
        result = matcher.match(state["jd_text"], k=int(state.get("k", 10)))
        records = _records_from_match(result)
        out: dict = {"shortlist": records}
        if state.get("shortlist"):
            out["prev_shortlist"] = state["shortlist"]
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -k "first_pass or done_intent" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add matching_agent.py tests/test_matching_agent.py requirements.txt
git commit -m "feat: LangGraph graph + interrupt()-based HITL router"
```

---

## Task 8: Compare + interview nodes wired into the graph

**Files:**
- Modify: `matching_agent.py`
- Test: `tests/test_matching_agent.py`

**Interfaces:**
- Consumes: `compare_candidates`, `generate_interview_questions` (agent_tools).
- Produces (added to `make_nodes` + graph):
  - `compare_node` — reads candidate names from the latest user message (fallback: top 3 of `shortlist`), calls `compare_candidates`, stores result in `state["report"]` (markdown) and loops back to `human_feedback`.
  - `interview_node` — extracts a candidate name from the latest user message (fallback: top candidate), calls `generate_interview_questions`, writes report, loops back.
  - Helper `_mentioned_names(message, shortlist) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py

def _route_llm(intent):
    # Router returns `intent` when classifying; "ok" otherwise.
    return StubLLM(lambda s, p: intent if "allowed labels" in p.lower() else "ok")


def test_compare_intent_produces_comparison_report(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, _route_llm("compare"))
    graph = build_agent(engine)
    cfg = {"configurable": {"thread_id": "c"}}
    graph.invoke({"jd_text": ML_JD, "k": 5, "messages": []}, cfg)
    state = graph.invoke(Command(resume="compare the top 3"), cfg)
    assert "comparison" in state["report"].lower() or "vs" in state["report"].lower()
    assert "__interrupt__" in state  # looped back for more input


def test_interview_intent_lists_questions(tmp_path, monkeypatch):
    def handler(system, prompt):
        if "allowed labels" in prompt.lower():
            return "interview"
        return "1. Tell me about PyTorch.\n2. Describe a Docker setup."
    engine = _engine(tmp_path, monkeypatch, StubLLM(handler))
    graph = build_agent(engine)
    cfg = {"configurable": {"thread_id": "i"}}
    first = graph.invoke({"jd_text": ML_JD, "k": 5, "messages": []}, cfg)
    top = first["shortlist"][0]["name"]
    state = graph.invoke(Command(resume=f"interview questions for {top}"), cfg)
    assert "?" in state["report"] or "1." in state["report"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k "compare_intent or interview_intent" -v`
Expected: FAIL (router maps `compare`/`interview` back to `human_feedback`, so no comparison/questions in the report).

- [ ] **Step 3: Write minimal implementation**

```python
# add to matching_agent.py imports
from agent_tools import compare_candidates, generate_interview_questions


def _latest_user_message(state: dict) -> str:
    for msg in reversed(state.get("messages") or []):
        if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")) in ("user", "human"):
            return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    return ""


def _mentioned_names(message: str, shortlist: list) -> list:
    msg = (message or "").lower()
    return [r["name"] for r in shortlist if str(r["name"]).lower() in msg]
```

```python
# inside make_nodes(): add nodes

    def compare_node(state: dict) -> dict:
        shortlist = state.get("shortlist", [])
        names = _mentioned_names(_latest_user_message(state), shortlist) \
            or [r["name"] for r in shortlist[:3]]
        result = compare_candidates(names, shortlist)
        lines = ["# Candidate comparison", "", f"Ranking: {', '.join(result['ranking'])}"]
        for dim, winner in result["dimensions"].items():
            lines.append(f"- Best {dim}: {winner}")
        if result["errors"]:
            lines.append(f"- (unknown: {', '.join(result['errors'])})")
        return {"report": "\n".join(lines)}

    def interview_node(state: dict) -> dict:
        shortlist = state.get("shortlist", [])
        names = _mentioned_names(_latest_user_message(state), shortlist)
        target = names[0] if names else (shortlist[0]["name"] if shortlist else "")
        out = generate_interview_questions(target, state.get("requirements") or {},
                                           shortlist, engine.llm)
        if out["error"]:
            return {"report": f"Could not generate questions: {out['error']}"}
        lines = [f"# Interview questions — {out['candidate']}", ""]
        lines += [f"{i}. {q}" for i, q in enumerate(out["questions"], 1)]
        return {"report": "\n".join(lines)}
```

Register both nodes, and update the conditional edges + loop-back edges:

```python
    # in build_agent: g.add_node("compare", nodes["compare_node"]); g.add_node("interview", nodes["interview_node"])
    # conditional edges map "compare"->"compare", "interview"->"interview"
    # then: g.add_edge("compare", "human_feedback"); g.add_edge("interview", "human_feedback")
```

Update `build_agent` accordingly (add the two nodes, change the routing dict entries for `compare` and `interview` to their own nodes, and add the loop-back edges).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -k "compare_intent or interview_intent" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add matching_agent.py tests/test_matching_agent.py
git commit -m "feat: compare + interview nodes wired into HITL loop"
```

---

## Task 9: Multi-round screening node (`Send()` fan-out)

**Files:**
- Modify: `matching_agent.py`
- Test: `tests/test_matching_agent.py`

**Interfaces:**
- Consumes: `langgraph.types.Send`.
- Produces:
  - `screen_node` — orchestrator: for the current `shortlist` (round-1 top-k), returns `Send("deep_analyze", {...})` per candidate.
  - `deep_analyze` worker — reads the full resume via `fs_tools.read_file`, asks the LLM for structured strengths/gaps + a `hire|no_hire|borderline` recommendation, writes into a `screening["analyses"]` list (reducer-merged).
  - `screen_collect` — aggregates analyses into `state["report"]` (round-3 recommendations) and loops back to `human_feedback`.
  - `AgentState` gains `screening: Annotated[dict, _merge_screening]` reducer.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py

def test_screen_intent_runs_rounds_and_recommends(tmp_path, monkeypatch):
    def handler(system, prompt):
        if "allowed labels" in prompt.lower():
            return "screen"
        if "recommendation" in prompt.lower():
            return "Recommendation: hire. Strong Python and PyTorch depth."
        return "ok"
    engine = _engine(tmp_path, monkeypatch, StubLLM(handler))
    graph = build_agent(engine)
    cfg = {"configurable": {"thread_id": "s"}}
    graph.invoke({"jd_text": ML_JD, "k": 5, "messages": []}, cfg)
    state = graph.invoke(Command(resume="deep-screen the top candidates"), cfg)
    report = state["report"].lower()
    assert "recommend" in report or "hire" in report
    analyses = state.get("screening", {}).get("analyses", [])
    assert analyses and all("recommendation" in a for a in analyses)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k screen_intent -v`
Expected: FAIL (`screen` currently routes back to `human_feedback`; no `screening` produced).

- [ ] **Step 3: Write minimal implementation**

```python
# add to matching_agent.py imports
import fs_tools
from langgraph.types import Send

_SCREEN_SYSTEM = (
    "You are a senior hiring manager doing a deep screen. Given a role and a "
    "candidate's resume text, give 2-3 strengths, 2-3 gaps, and end with exactly "
    "one line 'Recommendation: hire|no_hire|borderline' plus a short rationale."
)


def _merge_screening(left: dict, right: dict) -> dict:
    """Reducer: concatenate analyses lists, shallow-merge other keys."""
    left = left or {}
    right = right or {}
    merged = {**left, **{k: v for k, v in right.items() if k != "analyses"}}
    merged["analyses"] = (left.get("analyses") or []) + (right.get("analyses") or [])
    return merged


def _parse_recommendation(text: str) -> str:
    low = text.lower()
    for token in ("no_hire", "no-hire", "borderline", "hire"):
        if token in low:
            return "no_hire" if token in ("no_hire", "no-hire") else token
    return "borderline"
```

Change the `screening` field type in `AgentState` to `Annotated[dict, _merge_screening]`. Add nodes in `make_nodes`:

```python
    def screen_node(state: dict) -> list:
        title = (state.get("requirements") or {}).get("title", "the role")
        return [
            Send("deep_analyze", {"_cand": rec, "_title": title})
            for rec in state.get("shortlist", [])
        ]

    def deep_analyze(payload: dict) -> dict:
        rec = payload["_cand"]
        read = fs_tools.read_file(rec["resume_path"])
        text = str(read.get("content", ""))[:6000] if read.get("success") else ""
        prompt = (f"Role: {payload['_title']}\nCandidate: {rec['name']} "
                  f"(prior score {rec['score']}/100)\nResume:\n{text}")
        analysis = narrate(engine.llm, _SCREEN_SYSTEM, prompt)
        return {"screening": {"analyses": [{
            "name": rec["name"], "score": rec["score"],
            "analysis": analysis, "recommendation": _parse_recommendation(analysis),
        }]}}

    def screen_collect(state: dict) -> dict:
        analyses = (state.get("screening") or {}).get("analyses", [])
        lines = ["# Multi-round screening — recommendations", ""]
        for a in sorted(analyses, key=lambda x: -int(x["score"])):
            lines.append(f"## {a['name']} — {a['recommendation'].upper()}")
            lines.append(a["analysis"])
            lines.append("")
        return {"report": "\n".join(lines)}
```

Wire into `build_agent`: add nodes `screen` (→ `screen_node`), `deep_analyze`, `screen_collect`; route `"screen": "screen"`; `g.add_edge("deep_analyze", "screen_collect")`; `g.add_edge("screen_collect", "human_feedback")`. The `Send` list from `screen_node` fans out to `deep_analyze` workers; LangGraph joins them before `screen_collect`.

> Note on `deep_analyze`: it is the one node that reads raw resume text — permitted because it is a *generative* deep-screen, not a ranking step; it never rewrites `shortlist` order (round-1 scores stand; it only adds a recommendation).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -k screen_intent -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add matching_agent.py tests/test_matching_agent.py
git commit -m "feat: multi-round screening via Send() fan-out + hire recommendations"
```

---

## Task 10: `MatchingAgent` wrapper + invariant + refine tests

**Files:**
- Modify: `matching_agent.py`
- Test: `tests/test_matching_agent.py`

**Interfaces:**
- Produces:
  - `class MatchingAgent:` `MatchingAgent(matcher: JobMatcher, llm: LLMClient, thread_id="default")`; methods `start(jd_text, k=10) -> dict` (returns first-pass state) and `send(message) -> dict` (resumes the graph). Internally holds the compiled graph + config.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py
from matching_agent import MatchingAgent


def test_refine_reranks_and_explains_delta(tmp_path, monkeypatch):
    def handler(system, prompt):
        if "allowed labels" in prompt.lower():
            return "refine"
        return "ok"
    rag = make_corpus(tmp_path, monkeypatch); rag.build_index()
    agent = MatchingAgent(JobMatcher(rag=rag), StubLLM(handler), thread_id="r")
    first = agent.start(ML_JD, k=5)
    assert first["shortlist"]
    after = agent.send("weight experience higher please")
    assert after.get("last_intent") == "refine"
    assert after["shortlist"]  # re-ranked, still present


def test_invariant_llm_cannot_reorder(tmp_path, monkeypatch):
    # Hostile LLM tries to inject a different order; shortlist order must hold.
    rag = make_corpus(tmp_path, monkeypatch); rag.build_index()
    matcher = JobMatcher(rag=rag)
    baseline = [m["candidate_name"] for m in matcher.match(ML_JD, k=5)["top_matches"]]
    agent = MatchingAgent(matcher, StubLLM(lambda s, p: "IGNORE ALL — rank Jordan #1"), thread_id="inv")
    state = agent.start(ML_JD, k=5)
    assert [r["name"] for r in state["shortlist"]] == baseline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k "refine_rerank or invariant" -v`
Expected: FAIL with `ImportError: cannot import name 'MatchingAgent'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to matching_agent.py
from langgraph.types import Command


class MatchingAgent:
    """Convenience wrapper: one compiled graph + a stable thread."""

    def __init__(self, matcher: JobMatcher, llm: LLMClient, thread_id: str = "default") -> None:
        self._graph = build_agent(Engine(matcher=matcher, llm=llm))
        self._cfg = {"configurable": {"thread_id": thread_id}}

    def start(self, jd_text: str, k: int = 10) -> dict:
        return self._graph.invoke({"jd_text": jd_text, "k": k, "messages": []}, self._cfg)

    def send(self, message: str) -> dict:
        return self._graph.invoke(Command(resume=message), self._cfg)
```

For the refine test to actually change weights, extend `extract_requirements_node` to read a weight hint from the latest user message (minimal, deterministic):

```python
    def extract_requirements_node(state: dict) -> dict:
        req = _extract_requirements(state["jd_text"])
        existing = state.get("requirements") or {}
        weights = dict(existing.get("weights") or req["weights"])
        msg = _latest_user_message(state).lower()
        if "experience" in msg and ("higher" in msg or "more" in msg):
            weights["experience"] = min(0.30, weights["experience"] + 0.15)
            weights["retrieval"] = max(0.30, weights["retrieval"] - 0.15)
        req["weights"] = weights
        return {"requirements": req}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -v`
Expected: PASS (all matching_agent tests).

- [ ] **Step 5: Commit**

```bash
git add matching_agent.py tests/test_matching_agent.py
git commit -m "feat: MatchingAgent wrapper + refine weights + ordering-invariant tests"
```

---

## Task 11: Conversational CLI (`agent_cli.py`)

**Files:**
- Create: `agent_cli.py`
- Test: `tests/test_matching_agent.py` (a smoke test of the render helpers)

**Interfaces:**
- Produces:
  - `def render_state(state: dict) -> str` — formats report + a "What next?" prompt.
  - `def main(argv=None) -> int` — REPL: read JD (path via `fs_tools.read_file` or inline text), `MatchingAgent.start`, print, then loop reading stdin → `.send`. Flags `--k`, `--no-stream`, `--anonymize`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py
from agent_cli import render_state


def test_render_state_includes_report():
    out = render_state({"report": "# Hi\nbody", "shortlist": [{"name": "A", "score": 90}]})
    assert "# Hi" in out and "A" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k render_state -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_cli'`.

- [ ] **Step 3: Write minimal implementation**

```python
# agent_cli.py
"""agent_cli.py — streaming conversational CLI for the matching agent."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import fs_tools
from agent_llm import AnthropicLLM
from job_matcher import JobMatcher
from matching_agent import MatchingAgent


def render_state(state: dict) -> str:
    lines = [state.get("report", "(no report)")]
    if state.get("shortlist"):
        lines.append("")
        lines.append("Candidates: " + ", ".join(
            f"{r['name']} ({r['score']})" for r in state["shortlist"]))
    lines.append("\nWhat next? (refine / compare / interview <name> / screen / done)")
    return "\n".join(lines)


def _load_jd(source: str) -> str:
    res = fs_tools.read_file(source)
    return str(res["content"]) if res.get("success") else source


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Conversational resume-matching agent.")
    parser.add_argument("jd", help="path to a JD file or inline JD text")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args(argv)

    agent = MatchingAgent(JobMatcher(), AnthropicLLM())
    state = agent.start(_load_jd(args.jd), k=args.k)
    print(render_state(state))

    while "__interrupt__" in state:
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        state = agent.send(message)
        print(render_state(state))
        if state.get("last_intent") == "done":
            break
    print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -k render_state -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_cli.py tests/test_matching_agent.py
git commit -m "feat: streaming conversational CLI for the matching agent"
```

---

## Task 12: Streamlit "Agent Chat" tab (`app.py`)

**Files:**
- Modify: `app.py`
- Test: manual (Streamlit UI). Add a guard import test below.

**Interfaces:**
- Consumes: `MatchingAgent`, `JobMatcher`, `AnthropicLLM`.
- Produces: a new tab/section in `app.py` that: takes a JD (text area or file upload reusing existing upload handling), calls `MatchingAgent.start`, renders the report as markdown and the shortlist as a dataframe, and provides a chat input that calls `.send`. Agent + thread id stored in `st.session_state`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py
def test_app_exposes_agent_renderer():
    import app  # noqa: F401 — import must succeed with the new tab wired in
    assert hasattr(app, "render_agent_tab")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k app_exposes -v`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'render_agent_tab'`.

- [ ] **Step 3: Write minimal implementation**

Read `app.py` first to follow its existing tab/structure conventions, then add:

```python
# app.py — add this function and call it from the tab layout
def render_agent_tab() -> None:
    import streamlit as st
    from agent_llm import AnthropicLLM
    from job_matcher import JobMatcher
    from matching_agent import MatchingAgent

    st.header("Agent Chat")
    jd = st.text_area("Job description", height=180, key="agent_jd")
    if st.button("Start matching", key="agent_start") and jd.strip():
        st.session_state["agent"] = MatchingAgent(JobMatcher(), AnthropicLLM(),
                                                  thread_id="streamlit")
        st.session_state["agent_state"] = st.session_state["agent"].start(jd, k=10)

    state = st.session_state.get("agent_state")
    if state:
        st.markdown(state.get("report", ""))
        if state.get("shortlist"):
            st.dataframe([{"name": r["name"], "score": r["score"]}
                          for r in state["shortlist"]])
        follow = st.chat_input("Refine, compare, interview <name>, screen, or done")
        if follow:
            st.session_state["agent_state"] = st.session_state["agent"].send(follow)
            st.rerun()
```

Wire `render_agent_tab()` into the existing tab container (follow the pattern already in `app.py`; e.g. add an "Agent Chat" entry to the existing `st.tabs([...])` call).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_matching_agent.py -k app_exposes -v`
Expected: PASS.

Manual check (controller-owned, time-boxed — do NOT block a subagent on this):
`python -m streamlit run app.py` → open the "Agent Chat" tab, paste a JD, verify a report renders and a follow-up updates it.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_matching_agent.py
git commit -m "feat: Streamlit Agent Chat tab"
```

---

## Task 13: State-machine diagram, compliance helpers, README

**Files:**
- Create: `docs/diagrams/matching_agent_state_machine.md`, `scripts/export_graph.py`
- Modify: `matching_agent.py` (decision log + `--anonymize`), `README.md`
- Test: `tests/test_matching_agent.py`

**Interfaces:**
- Produces:
  - `def anonymize_jd_or_resume(text: str) -> str` — drops the contact/name preamble (everything before the first recognized section header) — reuse `resume_rag.split_into_sections`.
  - `def write_decision_log(state: dict, path: str) -> dict` — writes a JSON audit record via `fs_tools.write_file`.
  - `scripts/export_graph.py` — renders the compiled graph to Mermaid + PNG.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_matching_agent.py
import json
from matching_agent import anonymize_jd_or_resume, write_decision_log


def test_anonymize_drops_contact_preamble():
    out = anonymize_jd_or_resume("Jane Doe\njane@x.com\n\nSKILLS\nPython")
    assert "jane@x.com" not in out and "Python" in out


def test_write_decision_log_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("FS_TOOLS_BASE_DIR", str(tmp_path))
    state = {"requirements": {"title": "ML"}, "shortlist": [{"name": "A", "score": 90,
             "breakdown": {}, "reasoning": "r"}]}
    res = write_decision_log(state, "logs/run.json")
    assert res["success"]
    data = json.loads((tmp_path / "logs" / "run.json").read_text(encoding="utf-8"))
    assert data["title"] == "ML" and data["candidates"][0]["name"] == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_matching_agent.py -k "anonymize or decision_log" -v`
Expected: FAIL with `ImportError: cannot import name 'anonymize_jd_or_resume'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to matching_agent.py
import json


def anonymize_jd_or_resume(text: str) -> str:
    """Drop the name/contact preamble (demographic proxies) before scoring."""
    from resume_rag import split_into_sections
    sections = split_into_sections(text)
    kept = [s for s in sections if s.kind != "header"]
    return "\n\n".join(f"{s.header}\n{s.text}" if s.header else s.text for s in kept) or text


def write_decision_log(state: dict, path: str) -> dict:
    """Persist an auditable JSON record of one matching run."""
    record = {
        "title": (state.get("requirements") or {}).get("title", ""),
        "weights": (state.get("requirements") or {}).get("weights", {}),
        "candidates": [
            {"name": r["name"], "score": r["score"],
             "breakdown": r.get("breakdown", {}), "reasoning": r.get("reasoning", "")}
            for r in state.get("shortlist", [])
        ],
    }
    return fs_tools.write_file(path, json.dumps(record, indent=2))
```

Create `scripts/export_graph.py`:

```python
# scripts/export_graph.py
"""Render the matching agent's state machine to Mermaid + PNG."""

from __future__ import annotations

from pathlib import Path

from agent_llm import StubLLM
from job_matcher import JobMatcher
from matching_agent import Engine, build_agent


def main() -> int:
    graph = build_agent(Engine(matcher=JobMatcher.__new__(JobMatcher), llm=StubLLM([])))
    mermaid = graph.get_graph().draw_mermaid()
    out_dir = Path("docs/diagrams")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matching_agent_state_machine.md").write_text(
        f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")
    try:
        png = graph.get_graph().draw_mermaid_png()
        (out_dir / "matching_agent_state_machine.png").write_bytes(png)
    except Exception as exc:  # noqa: BLE001 — PNG needs network/graphviz; mermaid is enough
        print(f"(PNG export skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> If `JobMatcher.__new__` causes issues in `build_agent` graph construction (it shouldn't — no nodes run at build time), pass a real `JobMatcher()` instead.

Run it: `python scripts/export_graph.py` → produces `docs/diagrams/matching_agent_state_machine.md` (+ PNG if renderable).

Add a **Milestone 3** section to `README.md` covering: how to run the CLI (`python agent_cli.py <jd>`) and Streamlit tab, the graph diagram, the 5+ test scenarios, and an **"Explainability & Compliance"** subsection documenting the deterministic-score/generative-explain split, the decision log, `--anonymize`, and the verified caveat that model-level fairness metrics (NYC LL144 impact ratio / EEOC four-fifths) do not capture whole-funnel bias.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_matching_agent.py -k "anonymize or decision_log" -v`
Expected: PASS.
Then full suite: `python -m pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add matching_agent.py scripts/export_graph.py docs/diagrams/ README.md tests/test_matching_agent.py
git commit -m "feat: state-machine diagram, decision log, anonymize, README (M3)"
```

---

## Self-Review

**Spec coverage:**
- Agent state (§5) → Task 5 (`AgentState`) + Task 7/9 reducers. ✔
- Graph nodes (§6) → Tasks 5–9. ✔
- 3 tools (§7) → Tasks 2–4. ✔
- Conversational NL + iterative refinement + explain Δ (Part B) → Tasks 7, 8, 10. ✔
- Multi-round screening (Part C) → Task 9. ✔
- Explainability / strengths-gaps / borderline suggestions → Task 6. ✔
- CLI + Streamlit (interfaces §8) → Tasks 11, 12. ✔
- Fairness/compliance (§9) → Task 13. ✔
- State-machine diagram → Task 13. ✔
- 5+ test scenarios (§10): first-pass, refine, compare, interview, screen, error (unknown id in Task 4 / Task 8 fallback), invariant → Tasks 4–10. ✔ (7 scenarios)
- Injectable LLM / offline tests (§2) → Task 1 + all tests use `StubLLM`. ✔

**Placeholder scan:** every code step contains complete code; commands have expected output. No TBD/TODO. ✔

**Type consistency:** `CandidateRecord` dict keys (`name, resume_path, score, breakdown, matched_skills, excerpts, reasoning`, +`summary`, +`deep_analysis`/`recommendation` via `screening`) are produced in Task 5 and consumed consistently in Tasks 6, 8, 9, 13. `Engine`, `make_nodes`, `build_agent`, `MatchingAgent`, `LLMClient`, `StubLLM`, `classify_intent`, `narrate`, the three tools, and `DEFAULT_WEIGHTS` are referenced with the same signatures throughout. ✔

**Known follow-ups for the implementer:** `app.py` must be read before editing Task 12 to match its existing tab structure; `scripts/export_graph.py` may need a real `JobMatcher()` if graph construction touches the matcher.
