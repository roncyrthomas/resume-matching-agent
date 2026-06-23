from __future__ import annotations

from agent_tools import DEFAULT_WEIGHTS, compare_candidates, extract_requirements, rag_search
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
