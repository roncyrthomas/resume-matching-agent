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
from langgraph.types import Command, interrupt, Send

import fs_tools
from agent_llm import LLMClient, classify_intent, narrate
from agent_tools import compare_candidates, generate_interview_questions
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
# Conversation + screening helpers
# ---------------------------------------------------------------------------


def _latest_user_message(state: dict) -> str:
    """Return the most recent user/human message content from the history."""
    for msg in reversed(state.get("messages") or []):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in ("user", "human"):
            return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    return ""


def _mentioned_names(message: str, shortlist: list) -> list:
    """Names from *shortlist* that appear (case-insensitively) in *message*."""
    msg = (message or "").lower()
    return [r["name"] for r in shortlist if str(r["name"]).lower() in msg]


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
    """Pull a hire/no_hire/borderline verdict from free-text analysis."""
    low = (text or "").lower()
    for token in ("no_hire", "no-hire", "borderline", "hire"):
        if token in low:
            return "no_hire" if token in ("no_hire", "no-hire") else token
    return "borderline"


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
    screening: Annotated[dict, _merge_screening]
    last_intent: str
    report: str
    k: int


def route_after_feedback(state: dict) -> str:
    """Map the classified follow-up intent to the next node (or END)."""
    intent = state.get("last_intent", "done")
    return END if intent == "done" else intent


def fan_out_candidates(state: dict) -> list:
    """Conditional-edge fan-out: one ``deep_analyze`` worker per shortlisted
    candidate. Returning Send objects from an edge (not a node) is the LangGraph
    pattern for a runtime-sized parallel map."""
    title = (state.get("requirements") or {}).get("title", "the role")
    return [
        Send("deep_analyze", {"_cand": rec, "_title": title})
        for rec in state.get("shortlist", [])
    ]


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
        # Preserve user-adjusted weights across a refine loop, and apply a simple
        # NL weight hint ("weight experience higher") from the latest message.
        existing = state.get("requirements") or {}
        weights = dict(existing.get("weights") or req["weights"])
        msg = _latest_user_message(state).lower()
        if "experience" in msg and ("higher" in msg or "more" in msg):
            weights["experience"] = min(0.30, weights["experience"] + 0.15)
            weights["retrieval"] = max(0.30, weights["retrieval"] - 0.15)
        req["weights"] = weights
        return {"requirements": req}

    def search_resumes(state: dict) -> dict:
        # Retrieval is handled inside JobMatcher.match; this node records intent
        # and keeps the graph shape faithful to the assignment diagram.
        return {"messages": []}

    def rank_candidates(state: dict) -> dict:
        # Pass the (possibly user-adjusted) top-level utility weights straight
        # into the matcher. Defaults equal the module constants, so the first
        # pass is byte-identical to a plain match() — the ordering invariant
        # holds — while a refine loop genuinely reweights the factors.
        weights = (state.get("requirements") or {}).get("weights") or None
        result = engine.matcher.match(
            state["jd_text"], k=int(state.get("k", 10)), weights=weights)
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

    def compare_node(state: dict) -> dict:
        shortlist = state.get("shortlist", [])
        names = _mentioned_names(_latest_user_message(state), shortlist) \
            or [r["name"] for r in shortlist[:3]]
        result = compare_candidates(names, shortlist)
        lines = ["# Candidate comparison", "",
                 f"Ranking: {', '.join(result['ranking'])}"]
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

    def screen_node(state: dict) -> dict:
        # Pass-through marker node; the per-candidate fan-out happens on the
        # conditional edge `fan_out_candidates` (a node may not return Sends).
        return {}

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

    return {
        "parse_jd": parse_jd,
        "extract_requirements": extract_requirements_node,
        "search_resumes": search_resumes,
        "rank_candidates": rank_candidates,
        "summarize_shortlist": summarize_shortlist,
        "generate_report": generate_report,
        "human_feedback": human_feedback,
        "compare_node": compare_node,
        "interview_node": interview_node,
        "screen_node": screen_node,
        "deep_analyze": deep_analyze,
        "screen_collect": screen_collect,
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
    g.add_node("compare", nodes["compare_node"])
    g.add_node("interview", nodes["interview_node"])
    g.add_node("screen", nodes["screen_node"])
    g.add_node("deep_analyze", nodes["deep_analyze"])
    g.add_node("screen_collect", nodes["screen_collect"])

    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "extract_requirements")
    g.add_edge("extract_requirements", "search_resumes")
    g.add_edge("search_resumes", "rank_candidates")
    g.add_edge("rank_candidates", "summarize_shortlist")
    g.add_edge("summarize_shortlist", "generate_report")
    g.add_edge("generate_report", "human_feedback")
    g.add_conditional_edges("human_feedback", route_after_feedback, {
        "refine": "extract_requirements",
        "compare": "compare",
        "interview": "interview",
        "screen": "screen",
        END: END,
    })
    # compare / interview loop back for more input; screen fans out per candidate
    # (Send) into deep_analyze, then collects before looping back.
    g.add_edge("compare", "human_feedback")
    g.add_edge("interview", "human_feedback")
    g.add_conditional_edges("screen", fan_out_candidates, ["deep_analyze"])
    g.add_edge("deep_analyze", "screen_collect")
    g.add_edge("screen_collect", "human_feedback")
    return g.compile(checkpointer=checkpointer or MemorySaver())


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class MatchingAgent:
    """One compiled graph bound to a stable conversation thread."""

    def __init__(self, matcher: JobMatcher, llm: LLMClient,
                 thread_id: str = "default") -> None:
        self._graph = build_agent(Engine(matcher=matcher, llm=llm))
        self._cfg = {"configurable": {"thread_id": thread_id}}

    def start(self, jd_text: str, k: int = 10) -> dict:
        """Run the first pass; returns state paused at the human-feedback gate."""
        return self._graph.invoke(
            {"jd_text": jd_text, "k": k, "messages": []}, self._cfg)

    def send(self, message: str) -> dict:
        """Resume the graph with a natural-language follow-up."""
        return self._graph.invoke(Command(resume=message), self._cfg)


# ---------------------------------------------------------------------------
# Fairness / auditability helpers
# ---------------------------------------------------------------------------


def anonymize_jd_or_resume(text: str) -> str:
    """Drop the name/contact preamble (demographic proxies) before scoring."""
    from resume_rag import split_into_sections
    sections = split_into_sections(text)
    kept = [s for s in sections if s.kind != "header"]
    body = "\n\n".join(
        f"{s.header}\n{s.text}" if s.header else s.text for s in kept)
    return body or text


def write_decision_log(state: dict, path: str) -> dict:
    """Persist an auditable JSON record of one matching run."""
    import json
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
