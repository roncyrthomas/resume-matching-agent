"""agent_tools.py — the three Milestone-3 agent tools plus a RAG search wrapper.

Pure functions over the M2 deterministic core so they are unit-testable and
reusable both inside the LangGraph nodes and from the CLI. None of these
functions reorder or invent match scores — numbers come from JobMatcher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from job_matcher import parse_job_description

if TYPE_CHECKING:
    from agent_llm import LLMClient

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

    def _best(key: str) -> str:
        winner = max(resolved, key=lambda r: r.get(key, 0))
        return str(winner.get("name", ""))

    dimensions = {
        "score": _best("score"),
        "skills": str(max(resolved, key=lambda r: len(r.get("matched_skills") or [])).get("name", "")),
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
