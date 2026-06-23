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
