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
