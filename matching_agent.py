"""matching_agent.py — LangGraph conversational resume-matching agent (M3).

Wraps the deterministic M1/M2 core in a human-in-the-loop graph. The LLM never
ranks: deterministic nodes own `shortlist` ordering; LLM nodes add prose only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, Dict, List, Optional

from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agent_llm import LLMClient, classify_intent, narrate
from agent_tools import DEFAULT_WEIGHTS
from agent_tools import extract_requirements as _extract_requirements
from job_matcher import JobMatcher

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

BORDERLINE = (45, 60)
INTENTS = ("refine", "compare", "interview", "screen", "done")

_SUMMARY_SYSTEM = (
    "You summarize a candidate's fit from PRE-COMPUTED evidence only. You are "
    "given a match score, a score breakdown, matched skills, and short resume "
    "excerpts. State strengths and gaps in 2-3 sentences. If asked, add one "
    "improvement suggestion. Never invent facts beyond the evidence; never "
    "restate or change the numeric score."
)


# ---------------------------------------------------------------------------
# Engine (dependency carrier)
# ---------------------------------------------------------------------------


@dataclass
class Engine:
    """Carries the deterministic matcher and the injectable LLM into nodes."""

    matcher: JobMatcher
    llm: LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Graph state + routing
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Conversation + matching state threaded through the graph."""

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
    """Map the classified follow-up intent to the next node (or END)."""
    intent = state.get("last_intent", "done")
    return END if intent == "done" else intent


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


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
        matcher = engine.matcher
        # Only override the matcher when the user adjusted the retrieval weight
        # away from the default; the first pass must use the configured matcher
        # unchanged so the deterministic ranking is reproducible.
        if sem is not None and abs(float(sem) - DEFAULT_WEIGHTS["retrieval"]) > 1e-9:
            matcher = JobMatcher(rag=engine.matcher.rag, semantic_weight=float(sem))
        result = matcher.match(state["jd_text"], k=int(state.get("k", 10)))
        records = _records_from_match(result)
        out: dict = {"shortlist": records}
        if state.get("shortlist"):
            out["prev_shortlist"] = state["shortlist"]
        return out

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

    def human_feedback(state: dict) -> dict:
        # Pause and persist full state; on resume the node re-runs from the top
        # and `interrupt` returns the value passed via Command(resume=...).
        user_message = interrupt({"report": state.get("report", ""),
                                  "prompt": "What would you like to do next?"})
        intent = classify_intent(engine.llm, str(user_message), INTENTS, default="done")
        return {"last_intent": intent,
                "messages": [{"role": "user", "content": str(user_message)}]}

    return {
        "parse_jd": parse_jd,
        "extract_requirements": extract_requirements_node,
        "search_resumes": search_resumes,
        "rank_candidates": rank_candidates,
        "summarize_shortlist": summarize_shortlist,
        "generate_report": generate_report,
        "human_feedback": human_feedback,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_agent(engine: Engine, checkpointer: Optional[object] = None):
    """Assemble and compile the LangGraph state machine.

    The conditional edges for ``compare`` / ``interview`` / ``screen`` route back
    to ``human_feedback`` for now; later tasks replace them with dedicated nodes.
    """
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
        "compare": "human_feedback",     # replaced in a later task
        "interview": "human_feedback",   # replaced in a later task
        "screen": "human_feedback",      # replaced in a later task
        END: END,
    })
    return g.compile(checkpointer=checkpointer or MemorySaver())
