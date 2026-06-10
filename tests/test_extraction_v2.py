"""Tests for extraction v2: month dates, per-skill tenure, LLM fallback."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import llm_extractor
from resume_rag import extract_date_ranges, extract_metadata, extract_skill_years, split_into_sections

MESSY_RESUME = """Dana Fox
dana.fox@example.com | Austin, TX

What I Bring
Hands-on engineer comfortable across Python services and React frontends.

Career History
Senior Engineer, Acme (Jan 2022 - present)
- Led Python microservices and Kubernetes deployments.
Engineer, Beta Corp (03/2019 - 11/2021)
- Built React dashboards with TypeScript.
"""


def test_month_name_and_slash_ranges():
    ranges = extract_date_ranges("Jan 2020 - Mar 2023 and 03/2019 - present", today=(2026, 6))
    assert ((2020, 1), (2023, 3)) in ranges
    assert ((2019, 3), (2026, 6)) in ranges


def test_year_only_ranges_keep_v1_semantics():
    meta = extract_metadata("A B\n\nEXPERIENCE\nDev, X (2016 - 2019)\nDev, Y (2019 - Present)\n",
                            today_year=2026)
    assert meta.experience_years == 10


def test_company_names_with_years_are_not_dates():
    ranges = extract_date_ranges("Worked at Studio 2000 - a creative agency", today=(2026, 6))
    assert ranges == []


def test_skill_years_attribution_and_merging():
    sections = split_into_sections(MESSY_RESUME)
    sy = dict(extract_skill_years(sections, today=(2026, 6)))
    assert sy["Python"] == pytest.approx(4.4, abs=0.15)        # Jan22->Jun26
    assert sy["React"] == pytest.approx(2.7, abs=0.15)         # Mar19->Nov21
    assert sy["Kubernetes"] == pytest.approx(sy["Python"], abs=0.01)


def test_extract_metadata_messy_resume_low_confidence_fields():
    meta = extract_metadata(MESSY_RESUME, fallback_name="dana_fox", today_year=2026)
    assert meta.name == "Dana Fox"
    assert meta.experience_years == 7          # Mar 2019 -> Jun 2026 floor
    assert dict(meta.skill_years)["Python"] > 4
    assert llm_extractor.should_use_llm(meta, split_into_sections(MESSY_RESUME)) is False


def test_should_use_llm_on_garbage():
    meta = extract_metadata("totally unstructured blob of words", fallback_name="x_y")
    assert llm_extractor.should_use_llm(meta, split_into_sections("blob")) is True


def _mock_client(payload):
    block = MagicMock(); block.type = "tool_use"; block.input = payload
    resp = MagicMock(); resp.content = [block]
    client = MagicMock(); client.messages.create.return_value = resp
    return client


def test_extract_with_llm_uses_cache(tmp_path):
    payload = {"name": "Pat Lee", "skills": ["Python"], "experience_years": 5,
               "education_level": "master"}
    client = _mock_client(payload)
    out1 = llm_extractor.extract_with_llm("resume text", client=client, cache_dir=str(tmp_path))
    out2 = llm_extractor.extract_with_llm("resume text", client=client, cache_dir=str(tmp_path))
    assert out1 == out2 == payload
    assert client.messages.create.call_count == 1   # second hit served from cache


def test_extract_with_llm_offline_safe():
    client = MagicMock(); client.messages.create.side_effect = RuntimeError("no network")
    assert llm_extractor.extract_with_llm("text", client=client, cache_dir=".cache/_t") is None


def test_merge_metadata_fills_gaps_only():
    base = extract_metadata("garbage words", fallback_name="a_b")   # name fallback, 0 years
    merged = llm_extractor.merge_metadata(base, {
        "name": "Real Name", "skills": ["python", "k8s"], "experience_years": 6,
        "education_level": "master", "skill_years": {"Python": 6}})
    assert merged.name == "Real Name"
    assert merged.experience_years == 6
    assert merged.education_level == "master"
    assert "Python" in merged.skills and "Kubernetes" in merged.skills
    good = extract_metadata(MESSY_RESUME, fallback_name="dana_fox", today_year=2026)
    kept = llm_extractor.merge_metadata(good, {"name": "Wrong", "skills": [],
                                               "experience_years": 99, "education_level": "phd"})
    assert kept.name == "Dana Fox" and kept.experience_years == 7   # regex wins when confident
