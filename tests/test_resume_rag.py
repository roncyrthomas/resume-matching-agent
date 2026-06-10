"""Tests for resume_rag.py: sections, chunking, skills, metadata, indexing."""

from __future__ import annotations

import pytest

from resume_rag import (
    MAX_CHUNK_CHARS,
    ResumeSection,
    chunk_section,
    extract_metadata,
    extract_skills,
    split_into_sections,
)
from tests.rag_test_utils import ML_RESUME, make_corpus


# --- section splitting -------------------------------------------------------

def test_split_into_sections_caps_headers():
    sections = split_into_sections(ML_RESUME)
    kinds = [s.kind for s in sections]
    assert kinds[0] == "header"  # name/title/contact preamble
    for expected in ("summary", "skills", "experience", "education"):
        assert expected in kinds
    exp = next(s for s in sections if s.kind == "experience")
    assert "Vision Corp" in exp.text and "EXPERIENCE" not in exp.text


def test_split_into_sections_title_case_and_alt_headers():
    text = "Pat Doe\n\nProfessional Summary\nEngineer.\n\nEMPLOYMENT HISTORY\nDev, X (2020 - 2024)\n\nCore Competencies\nPython"
    kinds = {s.kind for s in split_into_sections(text)}
    assert {"summary", "experience", "skills"} <= kinds


def test_split_handles_unknown_allcaps_header():
    text = "A B\n\nSUMMARY\nx\n\nAWARDS\nWon a thing."
    sections = split_into_sections(text)
    assert any(s.kind == "other" and s.header == "AWARDS" for s in sections)


def test_split_ignores_allcaps_bullets_and_acronyms():
    text = "A B\n\nSKILLS\n- HTML\n- CI/CD\nAWS\nCISSP\n\nEDUCATION\nB.S. in CS"
    sections = split_into_sections(text)
    skills = next(s for s in sections if s.kind == "skills")
    for token in ("HTML", "CI/CD", "AWS", "CISSP"):
        assert token in skills.text
    assert not any(s.kind == "other" for s in sections)


# --- chunking ----------------------------------------------------------------

def test_chunk_section_keeps_small_sections_whole():
    sec = ResumeSection(kind="education", header="EDUCATION", text="B.S. in CS, State University (2019)")
    chunks = chunk_section(sec)
    assert chunks == ["B.S. in CS, State University (2019)"]


def test_chunk_section_splits_long_sections_within_limit():
    body = "\n\n".join(
        f"Engineer, Company {i} (20{10+i} - 20{11+i})\n- Did impactful thing number {i} with Python."
        for i in range(30)
    )
    sec = ResumeSection(kind="experience", header="EXPERIENCE", text=body)
    chunks = chunk_section(sec)
    assert len(chunks) >= 2
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert "Company 0" in chunks[0] and "Company 29" in chunks[-1]


# --- skill extraction ----------------------------------------------------------

def test_extract_skills_word_boundaries():
    skills = extract_skills("JavaScript and TypeScript, PostgreSQL tuning, HTML pages")
    assert "JavaScript" in skills and "TypeScript" in skills
    assert "Java" not in skills          # not inside JavaScript
    assert "PostgreSQL" in skills and "SQL" not in skills  # not inside PostgreSQL
    assert "Machine Learning" not in skills  # 'ml' must not fire inside HTML


def test_extract_skills_aliases():
    skills = extract_skills("Built ML pipelines on k8s with sklearn and Postgres")
    assert {"Machine Learning", "Kubernetes", "scikit-learn", "PostgreSQL"} <= set(skills)


# --- metadata extraction ---------------------------------------------------------

def test_extract_metadata_full_resume():
    meta = extract_metadata(ML_RESUME, fallback_name="riley_carter", today_year=2026)
    assert meta.name == "Riley Carter"
    assert meta.experience_years == 8          # 2018 -> Present(2026)
    assert meta.education_level == "master"
    assert "PyTorch" in meta.skills and "Python" in meta.skills
    assert meta.education.startswith("M.S. in Computer Science")


def test_extract_metadata_years_from_multiple_ranges():
    text = "A B\n\nEXPERIENCE\nDev, X (2016 - 2019)\nDev, Y (2019 - Present)\n"
    meta = extract_metadata(text, today_year=2026)
    assert meta.experience_years == 10          # 2016 -> 2026


def test_extract_metadata_stated_years_fallback():
    text = "A B\n\nSUMMARY\nEngineer with 6+ years of experience.\n"
    meta = extract_metadata(text, today_year=2026)
    assert meta.experience_years == 6


def test_extract_metadata_name_fallback_and_unknown_education():
    meta = extract_metadata("just some text\nno proper header", fallback_name="jane_roe")
    assert meta.name == "Jane Roe"
    assert meta.education_level == "unknown"


# --- indexing + retrieval (fake embedder, ephemeral chroma) ------------------------

def test_build_index_and_query(tmp_path, monkeypatch):
    rag = make_corpus(tmp_path, monkeypatch)
    stats = rag.build_index(rebuild=True)
    assert stats.files_indexed == 3
    assert stats.chunks_indexed >= 12  # >= 4 sections + preamble per resume
    assert not stats.failures
    assert rag.count() == stats.chunks_indexed

    hits = rag.query("machine learning pytorch python experience", k=5)
    assert hits and hits[0].candidate == "Riley Carter"
    assert all(0.0 <= h.similarity <= 1.001 for h in hits)


def test_query_metadata_filter(tmp_path, monkeypatch):
    rag = make_corpus(tmp_path, monkeypatch)
    rag.build_index(rebuild=True)
    hits = rag.query("python developer", k=6, where={"exp_years": {"$gte": 5}})
    assert hits
    assert all(int(h.metadata["exp_years"]) >= 5 for h in hits)
    assert all(h.candidate != "Jordan Blake" for h in hits)  # 2 years


def test_candidate_profiles_aggregation(tmp_path, monkeypatch):
    rag = make_corpus(tmp_path, monkeypatch)
    rag.build_index(rebuild=True)
    profiles = rag.candidate_profiles()
    assert len(profiles) == 3
    riley = profiles["resumes/riley_carter.txt"]
    assert riley["candidate"] == "Riley Carter"
    assert riley["exp_years"] == 8
    assert riley["education_level"] == "master"
    assert "PyTorch" in riley["skills"]


def test_query_empty_text_raises(tmp_path, monkeypatch):
    rag = make_corpus(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        rag.query("   ")
