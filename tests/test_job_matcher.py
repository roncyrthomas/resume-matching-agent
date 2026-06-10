"""Tests for job_matcher.py: JD parsing, must-haves, scoring, end-to-end match."""

from __future__ import annotations

import pytest

from job_matcher import (
    JobMatcher,
    check_must_haves,
    parse_job_description,
)
from tests.rag_test_utils import make_corpus

ML_JD = """Job Title: Machine Learning Engineer

About the role:
We build fraud models.

Requirements:
- 5+ years of professional Python experience.
- Experience with PyTorch and pandas.
- Bachelor's degree in Computer Science or a related field.

Nice to have:
- AWS experience.
"""


# --- JD parsing -----------------------------------------------------------------

def test_parse_jd_title_and_skills():
    jd = parse_job_description(ML_JD)
    assert jd.title == "Machine Learning Engineer"
    assert "Python" in jd.required_skills and "PyTorch" in jd.required_skills
    assert jd.nice_to_have == ("AWS",)


def test_parse_jd_must_have_kinds():
    jd = parse_job_description(ML_JD)
    kinds = {mh.kind for mh in jd.must_haves}
    assert kinds == {"skill_years", "skills", "education"}
    sy = next(mh for mh in jd.must_haves if mh.kind == "skill_years")
    assert sy.years == 5 and "Python" in sy.skills
    edu = next(mh for mh in jd.must_haves if mh.kind == "education")
    assert edu.level == "bachelor"
    sk = next(mh for mh in jd.must_haves if mh.kind == "skills")
    assert set(sk.skills) == {"PyTorch", "pandas"} and sk.any_of is False


def test_parse_jd_any_of_and_total_years():
    text = """Job Title: Backend

Requirements:
- 4+ years of professional experience.
- Production experience with FastAPI or Django.
"""
    jd = parse_job_description(text)
    total = next(mh for mh in jd.must_haves if mh.kind == "total_years")
    assert total.years == 4
    sk = next(mh for mh in jd.must_haves if mh.kind == "skills")
    assert sk.any_of is True and set(sk.skills) == {"FastAPI", "Django"}


def test_parse_jd_without_requirements_section_uses_all_skills():
    jd = parse_job_description("Looking for a React and TypeScript engineer.")
    assert set(jd.required_skills) == {"React", "TypeScript"}
    assert jd.must_haves == ()


def test_parse_jd_inline_years_fallback():
    jd = parse_job_description("Senior Python engineer, 4+ years of experience required.")
    assert jd.must_haves
    assert jd.must_haves[0].years == 4


def test_parse_jd_marketing_years_not_a_must_have():
    jd = parse_job_description("We are a 10+ years old company hiring a Python engineer.")
    assert jd.must_haves == ()


def test_parse_jd_empty_raises():
    with pytest.raises(ValueError):
        parse_job_description("   ")


# --- must-have checks --------------------------------------------------------------

def _profile(**over):
    base = {
        "candidate": "X", "skills": ["Python", "PyTorch", "pandas"],
        "exp_years": 8, "education_level": "master",
    }
    base.update(over)
    return base


def test_check_must_haves_pass():
    jd = parse_job_description(ML_JD)
    assert check_must_haves(_profile(), jd.must_haves) == []


def test_check_must_haves_failures():
    jd = parse_job_description(ML_JD)
    too_junior = check_must_haves(_profile(exp_years=3), jd.must_haves)
    assert any("needs 5+" in r for r in too_junior)
    missing_skill = check_must_haves(_profile(skills=["Python"]), jd.must_haves)
    assert any("PyTorch" in r for r in missing_skill)
    low_edu = check_must_haves(_profile(education_level="unknown"), jd.must_haves)
    assert any("education" in r for r in low_edu)


def test_check_must_haves_doctorate_filters():
    jd = parse_job_description(
        "Job Title: Scientist\n\nRequirements:\n- Doctorate in a quantitative field.\n"
    )
    edu = next(mh for mh in jd.must_haves if mh.kind == "education")
    assert edu.level == "phd"
    assert check_must_haves(_profile(education_level="bachelor"), jd.must_haves)
    assert check_must_haves(_profile(education_level="phd"), jd.must_haves) == []


# --- end-to-end matching (fake embedder) ----------------------------------------------

@pytest.fixture()
def matcher(tmp_path, monkeypatch):
    rag = make_corpus(tmp_path, monkeypatch)
    rag.build_index(rebuild=True)
    return JobMatcher(rag=rag)


def test_match_output_schema_and_ranking(matcher):
    result = matcher.match(ML_JD, k=10)
    assert set(["job_description", "top_matches"]) <= set(result)
    top = result["top_matches"][0]
    assert set(["candidate_name", "resume_path", "match_score", "matched_skills",
                "relevant_excerpts", "reasoning"]) <= set(top)
    assert top["candidate_name"] == "Riley Carter"
    assert isinstance(top["match_score"], int) and 0 <= top["match_score"] <= 100
    assert {"Python", "PyTorch", "pandas"} <= set(top["matched_skills"])
    assert top["relevant_excerpts"] and all(isinstance(e, str) for e in top["relevant_excerpts"])
    assert "Riley" not in [f["candidate_name"] for f in result["filtered_out"]]


def test_match_filters_must_have_failures(matcher):
    result = matcher.match(ML_JD, k=10)
    filtered_names = {f["candidate_name"] for f in result["filtered_out"]}
    assert "Jordan Blake" in filtered_names  # 2 yrs < 5+, missing PyTorch/pandas
    jordan = next(f for f in result["filtered_out"] if f["candidate_name"] == "Jordan Blake")
    assert jordan["failed_requirements"]
    top_names = {m["candidate_name"] for m in result["top_matches"]}
    assert "Jordan Blake" not in top_names


def test_match_no_filters_keeps_everyone(matcher):
    result = matcher.match(ML_JD, k=10, apply_filters=False)
    assert result["filtered_out"] == []
    names = {m["candidate_name"] for m in result["top_matches"]}
    assert {"Riley Carter", "Jordan Blake", "Casey Morgan"} <= names


def test_match_semantic_only_mode(matcher):
    result = matcher.match(ML_JD, k=5, semantic_only=True)
    assert result["query"]["mode"] == "semantic_only"
    assert result["top_matches"][0]["candidate_name"] == "Riley Carter"
    assert result["top_matches"][0]["score_breakdown"]["keyword_bm25"] == 0.0


def test_match_scores_discriminate(matcher):
    result = matcher.match(ML_JD, k=10, apply_filters=False)
    scores = {m["candidate_name"]: m["match_score"] for m in result["top_matches"]}
    assert scores["Riley Carter"] > scores["Casey Morgan"]  # frontend ranks below ML


def test_match_reasoning_names_sections(matcher):
    result = matcher.match(ML_JD, k=3, apply_filters=False)
    reasoning = result["top_matches"][0]["reasoning"]
    assert "section" in reasoning.lower()
    assert any(s in reasoning for s in ("SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "HEADER", "PROJECTS"))


def test_match_latency_reported(matcher):
    first = matcher.match(ML_JD, k=5)
    second = matcher.match(ML_JD, k=5)
    assert first["latency_ms"]["keyword_index_build"] > 0
    assert second["latency_ms"]["keyword_index_build"] == 0.0
    assert second["latency_ms"]["total"] >= 0


def test_match_empty_index_raises(tmp_path, monkeypatch):
    import chromadb
    from resume_rag import ResumeRAG
    from tests.rag_test_utils import FakeEmbedder

    empty_dir = tmp_path / "resumes"
    empty_dir.mkdir()
    monkeypatch.setenv("FS_TOOLS_BASE_DIR", str(tmp_path))
    rag = ResumeRAG(resumes_dir="resumes", embedder=FakeEmbedder(),
                    client=chromadb.EphemeralClient())
    with pytest.raises(RuntimeError):
        JobMatcher(rag=rag).match(ML_JD)


def test_match_invalid_k_raises(matcher):
    with pytest.raises(ValueError):
        matcher.match(ML_JD, k=0)
