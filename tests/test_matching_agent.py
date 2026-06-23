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


# --- Task 7: graph + HITL router ------------------------------------------------

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
    # Narration calls return "ok"; the router classifies "done".
    engine = _engine(tmp_path, monkeypatch,
                     StubLLM(lambda s, p: "done" if "allowed labels" in p.lower() else "ok"))
    graph, cfg, _ = _run_first_pass(engine)
    final = graph.invoke(Command(resume="thanks, that's all"), cfg)
    assert final.get("last_intent") == "done"


# --- Task 8: compare + interview nodes ------------------------------------------


def test_compare_intent_produces_comparison_report(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch,
                     StubLLM(lambda s, p: "compare" if "allowed labels" in p.lower() else "ok"))
    graph = build_agent(engine)
    cfg = {"configurable": {"thread_id": "c"}}
    graph.invoke({"jd_text": ML_JD, "k": 5, "messages": []}, cfg)
    state = graph.invoke(Command(resume="compare the top 3"), cfg)
    assert "comparison" in state["report"].lower() or "ranking" in state["report"].lower()
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


# --- Task 9: multi-round screening (Send fan-out) -------------------------------


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
