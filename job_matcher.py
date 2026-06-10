"""job_matcher.py — Milestone 2 Part B: semantic + keyword job matching.

Given a job description (text or a .txt/.pdf/.docx file read through the
Milestone 1 ``fs_tools``), the matcher:

1. **Parses the JD**: title, skills mentioned anywhere, and *must-have*
   requirements taken from the Requirements section — "5+ years Python",
   "FastAPI or Django", "Bachelor's degree", and plain required skills.
2. **Hybrid retrieval**: the JD is embedded and run against the ChromaDB index
   built by ``resume_rag.py`` (semantic), while a BM25 index over the same
   chunks provides exact-keyword evidence for critical skills. The two signals
   are blended per chunk and aggregated per candidate.
3. **Scores 0-100**: weighted blend of retrieval strength, required-skill
   coverage, experience fit and nice-to-have coverage, with a transparent
   per-component breakdown.
4. **Filters must-haves**: candidates failing a hard requirement are excluded
   from ``top_matches`` and reported in ``filtered_out`` with the exact reason.
5. **Explains every match**: matched skills, the resume sections that drove
   retrieval, and 2-3 relevant excerpts.

Note: per-skill tenure is rarely stated on resumes, so "5+ years Python" is
checked as *has the skill* AND *total experience >= 5 years* — documented
behaviour, same approximation most ATS systems make.

CLI (run from the project root, after ``python resume_rag.py --rebuild``):

    python job_matcher.py job_descriptions/senior_ml_engineer.txt
    python job_matcher.py job_descriptions/backend_python_engineer.txt -k 10 \
        --output results/backend_matches.json
    python job_matcher.py --text "Looking for a React engineer..." --semantic-only
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import fs_tools
from resume_rag import ChunkHit, ResumeRAG, extract_skills

# --- Scoring weights ---------------------------------------------------------------

TOP_K = 10                 # assignment default
CHUNK_POOL = 60            # semantic pool; candidates with no chunk in this top-N are not ranked
SEMANTIC_WEIGHT = 0.65     # semantic share inside the retrieval signal (vs BM25)
W_RETRIEVAL = 0.45
W_SKILLS = 0.35
W_EXPERIENCE = 0.15
W_NICE = 0.05
EXCERPT_CHARS = 260

_EDU_RANK = {"unknown": 0, "bachelor": 1, "master": 2, "phd": 3}
_YEARS_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*years?\b", re.IGNORECASE)
_EXPERIENCE_CUE = re.compile(
    r"\b(experience|experienced|professional|industry|required|needed|minimum|at least)\b",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(r"\b(ph\.?d|doctorate|master|bachelor)('?s)?\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9+#./-]+")


# --- Job-description parsing ---------------------------------------------------------


@dataclass(frozen=True)
class MustHave:
    """One hard requirement parsed from the JD's Requirements section."""

    kind: str                  # "skill_years" | "total_years" | "education" | "skills"
    raw: str                   # the original requirement line
    skills: Tuple[str, ...] = ()
    years: int = 0
    level: str = ""            # for kind == "education"
    any_of: bool = False       # True: any listed skill satisfies; False: all required


@dataclass(frozen=True)
class JobDescription:
    title: str
    text: str
    skills: Tuple[str, ...]            # every vocabulary skill mentioned in the JD
    required_skills: Tuple[str, ...]   # skills named in the Requirements section
    nice_to_have: Tuple[str, ...]      # skills named in the Nice-to-have section
    must_haves: Tuple[MustHave, ...]


_SECTION_HEADER = re.compile(r"^([A-Za-z][A-Za-z /&-]{0,40}):\s*$")
_REQUIREMENT_HEADERS = ("requirements", "must have", "must-haves", "must haves",
                        "qualifications", "what you'll need", "what you will need")
_NICE_HEADERS = ("nice to have", "nice-to-haves", "preferred", "preferred qualifications",
                 "bonus", "bonus points")


def _split_jd_sections(text: str) -> Dict[str, List[str]]:
    """Group JD lines under their lowercase '<Header>:' names ('' = preamble)."""
    sections: Dict[str, List[str]] = {"": []}
    current = ""
    for line in text.splitlines():
        match = _SECTION_HEADER.match(line.strip())
        if match:
            current = match.group(1).strip().lower()
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return sections


def _parse_requirement_line(line: str) -> Optional[MustHave]:
    """Turn one requirement bullet into a MustHave (or None if not parseable)."""
    clean = line.strip().lstrip("-*• ").strip()
    if not clean:
        return None
    years_match = _YEARS_RE.search(clean)
    line_skills = tuple(extract_skills(clean))
    degree_match = _DEGREE_RE.search(clean)

    if years_match:
        years = int(years_match.group(1))
        if line_skills:
            return MustHave(kind="skill_years", raw=clean, skills=line_skills,
                            years=years, any_of=True)
        return MustHave(kind="total_years", raw=clean, years=years)
    if degree_match:
        raw_level = degree_match.group(1).lower().replace(".", "")
        level = "phd" if raw_level in ("phd", "doctorate") else raw_level
        return MustHave(kind="education", raw=clean, level=level)
    if line_skills:
        return MustHave(kind="skills", raw=clean, skills=line_skills,
                        any_of=" or " in clean.lower())
    return None


def parse_job_description(text: str) -> JobDescription:
    """Parse a free-text JD into title, skills and structured must-haves."""
    if not text or not text.strip():
        raise ValueError("job description text must not be empty")

    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        title = re.sub(r"^job title\s*:\s*", "", stripped, flags=re.IGNORECASE)
        break

    sections = _split_jd_sections(text)
    req_lines: List[str] = []
    nice_lines: List[str] = []
    for header, lines in sections.items():
        if any(header.startswith(h) for h in _REQUIREMENT_HEADERS):
            req_lines.extend(lines)
        elif any(header.startswith(h) for h in _NICE_HEADERS):
            nice_lines.extend(lines)

    must_haves = tuple(
        mh for mh in (_parse_requirement_line(ln) for ln in req_lines) if mh
    )
    if not must_haves:
        # No Requirements section (e.g. an inline one-liner JD). Fall back to
        # high-precision patterns anywhere in the text: degree demands, and
        # years that read as experience demands ("4+ years required") — but
        # not incidental years ("a 10+ years old company") or bare skills.
        parsed = (_parse_requirement_line(ln) for ln in text.splitlines())
        must_haves = tuple(
            mh for mh in parsed
            if mh and (
                mh.kind == "education"
                or (mh.kind in ("skill_years", "total_years")
                    and _EXPERIENCE_CUE.search(mh.raw))
            )
        )
    required_skills = tuple(sorted({s for mh in must_haves for s in mh.skills}))
    nice_to_have = tuple(extract_skills("\n".join(nice_lines)))
    all_skills = tuple(extract_skills(text))

    if not required_skills:  # JD without a Requirements section: every skill counts
        required_skills = all_skills

    return JobDescription(
        title=title or "Untitled role",
        text=text,
        skills=all_skills,
        required_skills=required_skills,
        nice_to_have=nice_to_have,
        must_haves=must_haves,
    )


# --- Must-have evaluation --------------------------------------------------------------


def check_must_haves(profile: Dict[str, object],
                     must_haves: Sequence[MustHave]) -> List[str]:
    """Return the list of failure reasons (empty list = candidate passes)."""
    skills = set(profile.get("skills") or [])
    years = int(profile.get("exp_years", 0) or 0)
    edu_rank = _EDU_RANK.get(str(profile.get("education_level", "unknown")), 0)

    failures: List[str] = []
    for mh in must_haves:
        if mh.kind == "skill_years":
            sy = profile.get("skill_years") or {}
            has_skill = any(s in skills for s in mh.skills)
            if not has_skill:
                failures.append(f"missing required skill {' / '.join(mh.skills)} ({mh.raw})")
            elif isinstance(sy, dict) and sy:
                # Use per-skill tenure when available
                best = max((float(sy.get(s, 0.0)) for s in mh.skills), default=0.0)
                if best < mh.years:
                    failures.append(
                        f"has {best:.1f} yrs of {'/'.join(mh.skills)}, needs {mh.years}+ ({mh.raw})"
                    )
            elif years < mh.years:
                failures.append(f"has {years} years, needs {mh.years}+ ({mh.raw})")
        elif mh.kind == "total_years":
            if years < mh.years:
                failures.append(f"has {years} years total, needs {mh.years}+ ({mh.raw})")
        elif mh.kind == "education":
            if edu_rank < _EDU_RANK.get(mh.level, 0):
                failures.append(
                    f"education level '{profile.get('education_level', 'unknown')}' "
                    f"below required '{mh.level}' ({mh.raw})"
                )
        elif mh.kind == "skills":
            matched = [s for s in mh.skills if s in skills]
            ok = bool(matched) if mh.any_of else len(matched) == len(mh.skills)
            if not ok:
                missing = [s for s in mh.skills if s not in skills]
                failures.append(f"missing {' / '.join(missing)} ({mh.raw})")
    return failures


# --- Matcher ----------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class _CandidateSignals:
    file: str
    best_cosine: float
    top_cosines: Tuple[float, ...]
    best_bm25: float
    top_sections: Tuple[str, ...]
    excerpts: Tuple[str, ...]


class JobMatcher:
    """Hybrid (semantic + BM25 keyword) matcher over the resume RAG index.

    The BM25 index and candidate profiles are cached on first use; create a
    fresh JobMatcher if the underlying ChromaDB collection is rebuilt.
    """

    def __init__(self, rag: Optional[ResumeRAG] = None,
                 semantic_weight: float = SEMANTIC_WEIGHT) -> None:
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be within [0, 1]")
        self.rag = rag or ResumeRAG()
        self.semantic_weight = semantic_weight
        self._chunks: List[ChunkHit] = []
        self._bm25 = None
        self._profiles: Dict[str, Dict[str, object]] = {}
        self.keyword_index_ms: float = 0.0

    # -- keyword side ---------------------------------------------------------

    def _ensure_keyword_index(self) -> bool:
        """Build the BM25 index once; return True only when built on this call."""
        if self._bm25 is not None:
            return False
        from rank_bm25 import BM25Okapi

        start = time.perf_counter()
        self._chunks = self.rag.all_chunks()
        if not self._chunks:
            raise RuntimeError(
                "resume index is empty — run `python resume_rag.py --rebuild` first"
            )
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks])
        self._profiles = self.rag.candidate_profiles()
        self.keyword_index_ms = round((time.perf_counter() - start) * 1000, 1)
        return True

    def _bm25_by_candidate(self, jd_text: str) -> Dict[str, float]:
        """Best normalised BM25 chunk score per resume file."""
        assert self._bm25 is not None
        scores = self._bm25.get_scores(_tokenize(jd_text))
        best: Dict[str, float] = {}
        for chunk, score in zip(self._chunks, scores):
            best[chunk.file] = max(best.get(chunk.file, 0.0), float(score))
        top = max(best.values()) if best else 0.0
        if top <= 0:
            return {file: 0.0 for file in best}
        return {file: value / top for file, value in best.items()}

    # -- semantic side ----------------------------------------------------------

    def _semantic_by_candidate(self, jd_text: str, pool: int) -> Dict[str, _CandidateSignals]:
        hits = self.rag.query(jd_text, k=pool)
        grouped: Dict[str, List[ChunkHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.file, []).append(hit)

        signals: Dict[str, _CandidateSignals] = {}
        for file, file_hits in grouped.items():
            file_hits.sort(key=lambda h: h.similarity, reverse=True)
            top = file_hits[:3]
            excerpts = tuple(
                f"[{h.section.upper()}] " + re.sub(r"\s+", " ", h.text).strip()[:EXCERPT_CHARS]
                for h in top[:3]
            )
            signals[file] = _CandidateSignals(
                file=file,
                best_cosine=top[0].similarity,
                top_cosines=tuple(h.similarity for h in top),
                best_bm25=0.0,
                top_sections=tuple(dict.fromkeys(h.section for h in top)),
                excerpts=excerpts,
            )
        return signals

    # -- scoring -----------------------------------------------------------------

    @staticmethod
    def _semantic_strength(signals: _CandidateSignals) -> float:
        """Map raw cosine similarity (MiniLM, ~0.15-0.75) onto [0, 1]."""
        cosines = signals.top_cosines or (signals.best_cosine,)
        blended = 0.75 * signals.best_cosine + 0.25 * (sum(cosines) / len(cosines))
        return _clamp((blended - 0.15) / 0.60)

    def _score_candidate(
        self, jd: JobDescription, signals: _CandidateSignals,
        bm25_norm: float, profile: Dict[str, object], semantic_only: bool,
    ) -> Tuple[int, Dict[str, float], List[str]]:
        semantic = self._semantic_strength(signals)
        retrieval = semantic if semantic_only else (
            self.semantic_weight * semantic + (1 - self.semantic_weight) * bm25_norm
        )

        cand_skills = set(profile.get("skills") or [])
        required = list(jd.required_skills)
        matched_required = [s for s in required if s in cand_skills]
        skill_cov = len(matched_required) / len(required) if required else 1.0

        nice = list(jd.nice_to_have)
        matched_nice = [s for s in nice if s in cand_skills]
        nice_cov = len(matched_nice) / len(nice) if nice else 0.0

        years = int(profile.get("exp_years", 0) or 0)
        req_years = max((mh.years for mh in jd.must_haves if mh.years), default=0)
        exp_fit = _clamp(years / req_years) if req_years else 1.0

        blend = (W_RETRIEVAL * retrieval + W_SKILLS * skill_cov
                 + W_EXPERIENCE * exp_fit + W_NICE * nice_cov)
        score = int(round(100 * _clamp(blend)))
        breakdown = {
            "semantic": round(semantic, 3),
            "keyword_bm25": round(bm25_norm, 3),
            "skill_coverage": round(skill_cov, 3),
            "experience_fit": round(exp_fit, 3),
            "nice_to_have": round(nice_cov, 3),
        }
        return score, breakdown, matched_required + [s for s in matched_nice
                                                     if s not in matched_required]

    @staticmethod
    def _reasoning(jd: JobDescription, profile: Dict[str, object], score: int,
                   matched: Sequence[str], signals: _CandidateSignals) -> str:
        band = ("Strong" if score >= 75 else "Good" if score >= 60
                else "Partial" if score >= 45 else "Weak")
        n_req = len(jd.required_skills)
        n_hit = len([s for s in matched if s in jd.required_skills])
        years = int(profile.get("exp_years", 0) or 0)
        req_years = max((mh.years for mh in jd.must_haves if mh.years), default=0)

        parts = [f"{band} match for '{jd.title}'"]
        if n_req:
            shown = ", ".join([s for s in matched if s in jd.required_skills][:5]) or "none"
            parts.append(f"covers {n_hit}/{n_req} required skills ({shown})")
        if years:
            exp = f"{years} years' experience"
            if req_years:
                exp += f" (requirement: {req_years}+)" if years >= req_years else \
                       f" (below the {req_years}+ requirement)"
            parts.append(exp)
        level = str(profile.get("education_level", "unknown"))
        if level != "unknown":
            parts.append(f"{level}-level education")
        sections = ", ".join(s.upper() for s in signals.top_sections[:3])
        parts.append(f"strongest sections: {sections}")
        return "; ".join(parts) + "."

    # -- public API -----------------------------------------------------------------

    def match(self, jd_text: str, k: int = TOP_K, *, apply_filters: bool = True,
              semantic_only: bool = False) -> Dict[str, object]:
        """Match resumes to *jd_text* and return the assignment's JSON shape."""
        if k <= 0:
            raise ValueError("k must be positive")
        total_start = time.perf_counter()
        jd = parse_job_description(jd_text)
        built_keyword_index = self._ensure_keyword_index()

        sem_start = time.perf_counter()
        signals = self._semantic_by_candidate(jd.text, pool=CHUNK_POOL)
        semantic_ms = round((time.perf_counter() - sem_start) * 1000, 1)

        kw_start = time.perf_counter()
        bm25 = {} if semantic_only else self._bm25_by_candidate(jd.text)
        keyword_ms = round((time.perf_counter() - kw_start) * 1000, 1)

        matches: List[Dict[str, object]] = []
        filtered_out: List[Dict[str, object]] = []
        for file, sig in signals.items():
            profile = self._profiles.get(file, {})
            score, breakdown, matched = self._score_candidate(
                jd, sig, bm25.get(file, 0.0), profile, semantic_only
            )
            entry: Dict[str, object] = {
                "candidate_name": profile.get("candidate", file),
                "resume_path": file,
                "match_score": score,
                "matched_skills": matched,
                "relevant_excerpts": list(sig.excerpts[:3]),
                "reasoning": self._reasoning(jd, profile, score, matched, sig),
                "score_breakdown": breakdown,
            }
            failures = check_must_haves(profile, jd.must_haves) if apply_filters else []
            if failures:
                filtered_out.append({
                    "candidate_name": entry["candidate_name"],
                    "resume_path": file,
                    "match_score": score,
                    "failed_requirements": failures,
                })
            else:
                matches.append(entry)

        matches.sort(key=lambda m: (-int(m["match_score"]), str(m["candidate_name"])))
        filtered_out.sort(key=lambda m: -int(m["match_score"]))
        total_ms = round((time.perf_counter() - total_start) * 1000, 1)

        return {
            "job_description": jd.text.strip(),
            "top_matches": matches[:k],
            "filtered_out": filtered_out,
            "query": {
                "title": jd.title,
                "required_skills": list(jd.required_skills),
                "nice_to_have": list(jd.nice_to_have),
                "must_haves": [mh.raw for mh in jd.must_haves],
                "mode": "semantic_only" if semantic_only else "hybrid",
                "k": k,
            },
            "latency_ms": {
                "semantic_search": semantic_ms,
                "keyword_search": keyword_ms,
                "keyword_index_build": self.keyword_index_ms if built_keyword_index else 0.0,
                "total": total_ms,
            },
        }

    def match_file(self, jd_path: str, k: int = TOP_K, **kwargs: object) -> Dict[str, object]:
        """Read a JD with the Milestone 1 tools and match it."""
        result = fs_tools.read_file(jd_path)
        if not result.get("success"):
            raise FileNotFoundError(f"could not read job description: {result.get('error')}")
        return self.match(str(result["content"]), k=k, **kwargs)  # type: ignore[arg-type]


# --- CLI -------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Match resumes to a job description.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("jd_path", nargs="?", help="path to a .txt/.pdf/.docx JD")
    source.add_argument("--text", help="job description text passed inline")
    parser.add_argument("-k", type=int, default=TOP_K, help="number of matches (default 10)")
    parser.add_argument("--semantic-only", action="store_true",
                        help="disable the BM25 keyword signal (ablation)")
    parser.add_argument("--no-filters", action="store_true",
                        help="do not exclude candidates failing must-haves")
    parser.add_argument("--output", help="also write the JSON result to this path")
    args = parser.parse_args(argv)

    matcher = JobMatcher()
    try:
        if args.jd_path:
            result = matcher.match_file(args.jd_path, k=args.k,
                                        apply_filters=not args.no_filters,
                                        semantic_only=args.semantic_only)
        else:
            result = matcher.match(args.text, k=args.k,
                                   apply_filters=not args.no_filters,
                                   semantic_only=args.semantic_only)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        write = fs_tools.write_file(args.output, payload)
        status = "saved to " + str(write.get("filepath")) if write.get("success") \
            else f"failed to save: {write.get('error')}"
        print(f"\n[{status}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
