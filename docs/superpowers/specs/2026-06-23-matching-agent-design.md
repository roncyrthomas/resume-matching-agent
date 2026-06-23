# Matching Agent (Milestone 3) — Design Spec

**Date:** 2026-06-23
**Status:** Approved (brainstorming) — pending spec review
**Author:** roncyrthomas
**Component:** `matching_agent.py` + supporting modules

---

## 1. Purpose & Scope

Milestone 3 adds a **LangGraph-based conversational matching agent** on top of the
existing Milestone 1 (file-system tools) and Milestone 2 (RAG + hybrid matcher).
The agent turns the deterministic matcher into an interactive, multi-round
recruiting assistant: it parses a job description, extracts must-have vs
nice-to-have requirements, retrieves and ranks candidates, generates explainable
match reports, and then enters a human-feedback loop that supports natural-language
refinement, head-to-head comparison, interview-question generation, and a
multi-round screening funnel ending in a hire / no-hire recommendation.

**Non-goals.** No reimplementation of matching, scoring, embedding, or extraction
logic — the agent *orchestrates* the existing `JobMatcher` / `ResumeRAG` /
`fs_tools` modules. No new embedding model, no knowledge graph, no sliding-window
multi-pass LLM reranking (overkill for a ~20–40 doc corpus). No production
deployment, auth, or persistence beyond an in-process checkpointer.

---

## 2. Design Principles (informed by industry research)

These principles come from a verified multi-source review of production
resume-matching systems (Synapse, ConFit v3, JobMatchAI, Smart-Hiring,
AI-Hiring-with-LLMs) and the official LangGraph docs. See §11 for sources.

1. **Two-stage retrieve-then-rerank** is the dominant production shape and is
   already implemented in M2 (dense + BM25 + optional cross-encoder). The agent
   reuses it unchanged.
2. **White-box, multi-factor utility score over a black-box model** — keep the
   transparent weighted `score_breakdown` from `JobMatcher`; expose its weights
   for conversational adjustment.
3. **Separate the deterministic scoring layer from the generative explanation
   layer (hard invariant).** LLM nodes receive only pre-computed scores,
   `score_breakdown`, matched skills, and excerpts — **never** the raw resume
   text — so the LLM narrates a ranking but can never inflate or re-order it.
   This makes every explanation auditable.
4. **Four-agent decomposition** (`extractor → evaluator → summarizer →
   formatter`) as the node taxonomy.
5. **Deterministic funnel wrapping a tool-calling core** — LangGraph "workflow"
   (fixed order) for the first pass and the funnel; dynamic routing only at the
   human-feedback node.
6. **Idempotent nodes** — because LangGraph re-executes a node from its start on
   `Command(resume=...)`, every node must be safe to re-run.

---

## 3. Architecture Overview

```
                 ┌────────────────────── deterministic core (M1 + M2) ──────────────────────┐
                 │  fs_tools.read/list/write/search   ResumeRAG.query   JobMatcher.match     │
                 └───────────────────────────────────────────────────────────────────────────┘
                                            ▲              ▲                ▲
                                            │ (tools)      │                │
   ┌────────────────────────────────────────┴──────────────┴────────────────┴───────────────┐
   │                         matching_agent.py  (LangGraph StateGraph)                         │
   │   parse_jd → extract_requirements → search_resumes → rank_candidates                      │
   │            → summarize_shortlist → generate_report → human_feedback [interrupt()]         │
   │                 ├─ refine → extract_requirements (adjust weights, explain Δ)              │
   │                 ├─ compare → compare_candidates                                            │
   │                 ├─ interview → generate_interview_questions                                │
   │                 ├─ screen → multi_round_screen (Send() fan-out → hire/no-hire)            │
   │                 └─ done → END                                                              │
   └───────────────────────────────────────────────────────────────────────────────────────────┘
        ▲                                            ▲
        │ injectable LLM (langchain-anthropic)        │ checkpointer (MemorySaver)
   ┌────┴─────────┐                          ┌────────┴──────────┐
   │ agent_cli.py │                          │ app.py Agent tab  │
   └──────────────┘                          └───────────────────┘
```

The LLM is used **only** for: (a) intent routing at `human_feedback`, (b) prose in
`summarize_shortlist` / reports, (c) interview-question generation. All ranking and
scoring stays deterministic. The LLM client is **injectable** so tests run with a
stub and no API key.

---

## 4. Modules & Files

| File | Responsibility |
|------|----------------|
| `matching_agent.py` | `AgentState`, the StateGraph, all nodes, the conditional router, `build_agent()` factory, `MatchingAgent` convenience wrapper. |
| `agent_tools.py` | The three new tools: `extract_requirements`, `compare_candidates`, `generate_interview_questions`; plus a thin `rag_search` wrapper over `ResumeRAG.query`. Pure functions usable outside the graph and unit-testable. |
| `agent_llm.py` | LLM abstraction: a `LLMClient` Protocol + a `langchain-anthropic` adapter + a `StubLLM` for tests. Routing/JSON helpers (`classify_intent`, `narrate`). |
| `agent_cli.py` | Conversational CLI that streams node transitions and tool calls (the reasoning trace for the demo). |
| `app.py` (edit) | New "Agent Chat" tab driving the same compiled graph. |
| `tests/test_matching_agent.py` | 5+ conversation-flow scenarios with `StubLLM`. |
| `tests/test_agent_tools.py` | Unit tests for the three new tools. |
| `docs/superpowers/specs/2026-06-23-matching-agent-design.md` | This spec. |
| `docs/diagrams/matching_agent_state_machine.*` | Mermaid source + PNG (via `graph.get_graph().draw_mermaid_png()`). |

Keep each file focused (target < 400 lines); `matching_agent.py` may approach but
not exceed the 800-line cap — if it grows, extract node groups.

---

## 5. Agent State

```python
class CandidateRecord(TypedDict):
    name: str
    resume_path: str
    score: int
    breakdown: dict          # score_breakdown from JobMatcher
    matched_skills: list[str]
    excerpts: list[str]
    reasoning: str
    deep_analysis: NotRequired[dict]   # filled in round 2
    recommendation: NotRequired[str]   # "hire" | "no_hire" | "borderline" (round 3)

class Requirements(TypedDict):
    title: str
    must_haves: list[str]
    nice_to_have: list[str]
    required_skills: list[str]
    weights: dict            # adjustable W_RETRIEVAL/W_SKILLS/W_EXPERIENCE/W_NICE

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    jd_text: str
    requirements: NotRequired[Requirements]
    shortlist: NotRequired[list[CandidateRecord]]
    prev_shortlist: NotRequired[list[CandidateRecord]]   # for "explain ranking Δ"
    screening: NotRequired[dict]              # multi-round funnel state
    last_intent: NotRequired[str]
    report: NotRequired[str]
    k: int                                     # top-k (default 10)
```

**State invariants:** `shortlist` is the single source of truth for ordering and is
written only by deterministic nodes (`rank_candidates`, `multi_round_screen`).
LLM nodes may add `reasoning` / `deep_analysis` / `report` text but must not
reorder `shortlist`.

---

## 6. Graph Nodes

1. **`parse_jd`** — accept a JD as inline text or a file path; read via
   `fs_tools.read_file`; store `jd_text`. Idempotent.
2. **`extract_requirements`** — call `parse_job_description` (M2) → fill
   `requirements` (must-have / nice-to-have / required_skills / default weights).
   On the refine path, merge user-requested weight/criteria changes here.
3. **`search_resumes`** — `ResumeRAG.query` over the corpus (semantic pool).
4. **`rank_candidates`** — `JobMatcher.match(...)` with current weights →
   write `shortlist` (top-k) and `filtered_out`. Deterministic. On refine,
   snapshot the old shortlist into `prev_shortlist` first.
5. **`summarize_shortlist`** — LLM **summarizer**: per-candidate strengths/gaps and
   improvement suggestions for borderline (45–60) candidates. Receives breakdown +
   excerpts only (invariant §2.3).
6. **`generate_report`** — **formatter**: assemble the human-readable match report
   (and, on refine, the ranking-delta explanation from `prev_shortlist`).
7. **`human_feedback`** — `interrupt()`; on resume, the user's message is classified
   by the LLM router into one of: `refine | compare | interview | screen | done`.
8. **`compare_candidates`** — call the `compare_candidates` tool on named/top-N
   candidates → comparison matrix + per-dimension winner; narrate.
9. **`generate_interview_questions`** — call the tool for a named candidate →
   gap-grounded screening questions.
10. **`multi_round_screen`** — the funnel:
    - **Round 1:** already have top-10 from `rank_candidates`.
    - **Round 2:** `Send()` fan-out, one worker per shortlisted candidate, each
      doing deep analysis (full-resume read via `read_file` + structured
      strengths/gaps). Aggregated back into `shortlist[*].deep_analysis`.
    - **Round 3:** hire / no-hire / borderline recommendation per candidate with
      rationale.

### Routing

`route_after_feedback(state) -> str` reads `last_intent` and returns the next node
name (conditional edges). Unknown / ambiguous intent → re-prompt (stay at
`human_feedback`) rather than guessing.

---

## 7. The Three Required Tools (`agent_tools.py`)

```python
def extract_requirements(jd: str) -> Requirements
# Wraps parse_job_description; returns must-have vs nice-to-have split + skills.

def compare_candidates(candidate_ids: list[str], shortlist: list[CandidateRecord]) -> dict
# Head-to-head: per-candidate skills/experience/education/score matrix,
# per-dimension winner, and a deterministic overall ranking. No LLM in the
# numbers (invariant); narration is added by the caller node.

def generate_interview_questions(candidate_id: str, requirements, shortlist, llm) -> list[str]
# Gap analysis (required_skills minus matched_skills) → LLM generates targeted
# screening questions grounded in that candidate's specific gaps and strengths.
```

`candidate_ids` accept either a candidate name or a resume path; resolve against
`shortlist` and return a structured error for unknown ids.

---

## 8. Interfaces

### CLI (`agent_cli.py`)
- REPL: read a JD (path or text) → run the first pass → print the report → loop.
- Streams each node entry and tool call (`-> parse_jd`, `-> rank_candidates(...)`)
  so the demo video shows the agent's reasoning trace.
- `--no-stream`, `--k`, `--anonymize` flags.

### Streamlit (`app.py` "Agent Chat" tab)
- Chat input + message history; renders shortlist as a table and the report as
  markdown; a sidebar shows current weights/requirements and a "reset" button.
- Reuses the same compiled graph + checkpointer (one thread per session).

---

## 9. Fairness, Explainability & Compliance (proportionate)

Informed by NYC Local Law 144, the EEOC four-fifths rule, and the EU AI Act
(hiring = high-risk). Scaled to an assignment, not a production compliance stack:

- **Auditability by construction** — the deterministic-score / generative-explain
  split (§2.3) means every ranking is reproducible and every explanation traces to
  numeric evidence.
- **Decision log** — optional JSON log per run (`jd → weights → per-candidate
  scores → reasoning → recommendation`) written via `fs_tools.write_file`.
- **`--anonymize`** — redact the name/contact preamble (demographic proxies) before
  scoring; matching is skill/experience-driven regardless.
- **Documented limitation** — README notes the impact-ratio (four-fifths)
  requirement and the verified caveat that *model-level fairness metrics cannot
  capture whole-funnel ("effective") bias*; a real deployment needs an independent
  annual bias audit. We do not claim compliance.

---

## 10. Error Handling & Testing

**Error handling** (reuse the M1/M2 structured-error pattern — never leak a
traceback to the UI):
- empty corpus → actionable message ("run `python resume_rag.py --rebuild`");
- unparseable / empty JD → `ValueError` surfaced as a friendly message;
- unknown candidate id in compare/interview → structured error, agent re-prompts;
- LLM/router failure → fall back to a deterministic default route (`done`) and
  report degraded mode; never crash the loop.

**Testing** (TDD, pytest, stub LLM — no API key, follows the M2 injected-client
pattern; remember `python -m pytest`, never bare `pytest`):
1. **Full first pass** — JD in → report out with a ranked shortlist.
2. **Refine** — "weight experience higher" re-ranks and explains the delta.
3. **Compare** — "compare the top 3" → matrix with per-dimension winners.
4. **Interview** — "interview questions for <name>" → gap-grounded questions.
5. **Multi-round screen** — "deep-screen the top 10" → round-2 analyses +
   round-3 hire/no-hire recommendations.
6. **Error/edge** — unknown candidate id and empty-corpus degrade gracefully.
7. **Invariant test** — LLM stub that tries to reorder is ignored; `shortlist`
   order is byte-identical to `JobMatcher` output (proves §2.3).

Target ≥ 80% coverage on `matching_agent.py` and `agent_tools.py`.

---

## 11. Dependencies

Add to `requirements.txt`: `langgraph`, `langchain-anthropic` (and its transitive
`langchain-core`). No other new runtime deps. Pin versions at implementation time.

---

## 12. Submission Mapping

| Assignment requirement | Where satisfied |
|------------------------|-----------------|
| LangGraph agent + state design | `matching_agent.py` §5–§6 |
| Graph: Parse JD → … → Human Feedback → END | §6 graph |
| Tools (fs + RAG + 3 new) | `agent_tools.py` §7 |
| Conversational NL queries | `human_feedback` router, CLI + Streamlit §8 |
| Iterative refinement + explain changes | `refine` path, `prev_shortlist` Δ §6 |
| Multi-round screening | `multi_round_screen` §6.10 |
| Explainability / strengths-gaps / suggestions | `summarize_shortlist` §6.5 |
| State machine diagram | `docs/diagrams/...` §4 |
| Chat interface (CLI/Streamlit) | §8 |
| 5+ test scenarios | §10 |
| Demo video | recorded against the CLI trace + Streamlit tab |

**Research sources (verified):** Synapse (arXiv:2604.02539), ConFit v3
(arXiv:2605.09760), JobMatchAI (arXiv:2603.14558), Smart-Hiring
(arXiv:2511.02537), AI-Hiring-with-LLMs (arXiv:2504.02870), LangGraph official
docs (workflows-agents, interrupts), NYC LL144 / EEOC / EU AI Act compliance
review (arXiv:2501.10371 + DCWP/DLA Piper).
