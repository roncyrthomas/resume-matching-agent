"""llm_extractor.py — Claude-assisted resume metadata extraction.

Assists extraction when regex confidence is low; cached; offline-safe.

Public API:
    LLM_TOOL               — Anthropic tool schema dict
    DEFAULT_MODEL          — default model id (override via RESUME_RAG_LLM_MODEL)
    should_use_llm(meta, sections) -> bool
    extract_with_llm(text, *, client, model, cache_dir) -> Optional[dict]
    merge_metadata(regex_meta, llm) -> ResumeMetadata
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from resume_rag import ResumeMetadata, ResumeSection

DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # override via RESUME_RAG_LLM_MODEL

LLM_TOOL = {
    "name": "record_resume_metadata",
    "description": "Record structured metadata extracted from one resume.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "title": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "experience_years": {"type": "number"},
            "education_level": {
                "type": "string",
                "enum": ["phd", "master", "bachelor", "unknown"],
            },
            "education": {"type": "string"},
            "skill_years": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
        },
        "required": ["name", "skills", "experience_years", "education_level"],
    },
}


def should_use_llm(
    meta: "ResumeMetadata",
    sections: "Sequence[ResumeSection]",
) -> bool:
    """Return True when regex extraction likely failed and LLM would help.

    Signals (any one triggers):
    - No experience section found in sections (can't parse dates)
    - experience_years == 0  (no parseable date ranges)
    - No skills detected (vocabulary miss)
    """
    has_experience_section = any(s.kind == "experience" for s in sections)
    if not has_experience_section:
        return True
    if meta.experience_years == 0:
        return True
    if not meta.skills:
        return True
    return False


def extract_with_llm(
    text: str,
    *,
    client: Optional[object] = None,
    model: Optional[str] = None,
    cache_dir: str = ".cache/llm_extract",
) -> Optional[dict]:
    """Call the Anthropic API to extract resume metadata, with disk caching.

    Returns a dict matching the LLM_TOOL input schema, or None on any failure.
    Never raises — all exceptions are caught and None is returned.
    """
    try:
        effective_model = model or os.environ.get("RESUME_RAG_LLM_MODEL", DEFAULT_MODEL)

        # Build cache key from model + text content
        cache_key = hashlib.sha1(f"{effective_model}::{text}".encode("utf-8")).hexdigest()
        cache_path = Path(cache_dir) / f"{cache_key}.json"

        # Cache hit
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass  # corrupt cache — fall through to API call

        # Build client if not provided
        if client is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            from anthropic import Anthropic
            client = Anthropic()

        # Call the API with forced tool use
        resp = client.messages.create(  # type: ignore[union-attr]
            model=effective_model,
            max_tokens=1024,
            tools=[LLM_TOOL],
            tool_choice={"type": "tool", "name": "record_resume_metadata"},
            messages=[{
                "role": "user",
                "content": "Extract metadata from this resume:\n\n" + text[:8000],
            }],
        )

        # Find the tool_use block
        result_dict: Optional[dict] = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                result_dict = dict(block.input)
                break

        if result_dict is None:
            return None

        # Write to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result_dict), encoding="utf-8")
        except OSError:
            pass  # cache write failure is non-fatal

        return result_dict

    except Exception:  # noqa: BLE001
        # ANY failure (import error, auth, network, schema) → return None
        return None


def merge_metadata(
    regex_meta: "ResumeMetadata",
    llm: dict,
) -> "ResumeMetadata":
    """Merge LLM extraction results into regex metadata, filling gaps only.

    Rules (deterministic and testable):
    - name: LLM wins iff regex failed (experience_years == 0 AND no skills)
    - experience_years: LLM wins iff regex == 0
    - education_level / education: LLM wins iff regex level == "unknown"
    - skills: union — canonicalize LLM skills via extract_skills()
    - skill_years: LLM entries (canonicalized) added only for skills
      that regex skill_years lacks
    """
    from resume_rag import ResumeMetadata, extract_skills

    regex_failed = (regex_meta.experience_years == 0 and not regex_meta.skills)

    # Name
    llm_name = str(llm.get("name", "")).strip()
    name = llm_name if (regex_failed and llm_name) else regex_meta.name

    # Experience years
    llm_years_raw = llm.get("experience_years", 0)
    try:
        llm_years = round(float(llm_years_raw))  # schema allows floats; don't truncate 6.9 -> 6
    except (TypeError, ValueError):
        llm_years = 0
    experience_years = llm_years if regex_meta.experience_years == 0 else regex_meta.experience_years

    # Education
    llm_edu_level = str(llm.get("education_level", "unknown")).strip()
    llm_edu_detail = str(llm.get("education", "")).strip()
    if regex_meta.education_level == "unknown" and llm_edu_level != "unknown":
        education_level = llm_edu_level
        education = llm_edu_detail or regex_meta.education
    else:
        education_level = regex_meta.education_level
        education = regex_meta.education

    # Skills: union of regex skills + canonicalized LLM skills
    llm_raw_skills: list = llm.get("skills") or []
    llm_canonical = extract_skills(", ".join(str(s) for s in llm_raw_skills))
    merged_skills = tuple(sorted(set(regex_meta.skills) | set(llm_canonical)))

    # Skill years: add LLM entries only for skills missing from regex skill_years
    regex_sy_dict = dict(regex_meta.skill_years)
    llm_sy_raw: dict = llm.get("skill_years") or {}
    # Canonicalize LLM skill_years keys
    llm_sy_canonical: dict = {}
    for raw_skill, yrs in llm_sy_raw.items():
        canonical_list = extract_skills(str(raw_skill))
        for cs in canonical_list:
            if cs not in regex_sy_dict and cs not in llm_sy_canonical:
                try:
                    llm_sy_canonical[cs] = float(yrs)
                except (TypeError, ValueError):
                    pass

    merged_sy = {**regex_sy_dict, **llm_sy_canonical}

    return ResumeMetadata(
        name=name,
        title=regex_meta.title,
        skills=merged_skills,
        experience_years=experience_years,
        education_level=education_level,
        education=education,
        skill_years=tuple(sorted(merged_sy.items())),
    )
