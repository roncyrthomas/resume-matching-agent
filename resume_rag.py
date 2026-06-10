"""resume_rag.py — Milestone 2 Part A: RAG pipeline over the resume folder.

Document processing pipeline:

1. **Load** resumes (.txt/.pdf/.docx) with the Milestone 1 file-system tools
   (``fs_tools.list_files`` / ``fs_tools.read_file``).
2. **Chunk** section-aware: resumes are split on section headers (SUMMARY,
   SKILLS, EXPERIENCE, EDUCATION, ...) so a chunk never mixes, say, Education
   with Experience. Long sections are further split on paragraph boundaries
   with overlap.
3. **Extract metadata** per resume: candidate name, skills (matched against a
   curated vocabulary with aliases), years of experience (from dated job
   ranges, falling back to "N+ years" statements) and education level.
4. **Embed** chunks with a HuggingFace model (``all-MiniLM-L6-v2``), via
   sentence-transformers when available or ChromaDB's ONNX build of the same
   model otherwise.
5. **Store** embeddings + metadata in a persistent ChromaDB collection
   (cosine space) so the job matcher can filter on metadata at query time.

CLI smoke tests (run from the project root):

    python resume_rag.py --rebuild              # index ./resumes into .chroma/
    python resume_rag.py --query "ML engineer with PyTorch" -k 5
    python resume_rag.py --stats
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import fs_tools

# --- Constants -----------------------------------------------------------------

DEFAULT_RESUMES_DIR = "resumes"
DEFAULT_PERSIST_DIR = ".chroma"
DEFAULT_COLLECTION = "resumes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHUNK_CHARS = 1100
CHUNK_OVERLAP_CHARS = 150

# --- Skill vocabulary ------------------------------------------------------------
# Canonical skill -> lowercase aliases. Used for metadata extraction here and for
# keyword/hybrid matching in job_matcher.py. Word-boundary matched, so "Java"
# never fires inside "JavaScript" and "SQL" never fires inside "PostgreSQL".

SKILL_VOCAB: Dict[str, Tuple[str, ...]] = {
    "Python": ("python",),
    "Java": ("java",),
    "JavaScript": ("javascript",),
    "TypeScript": ("typescript",),
    "SQL": ("sql",),
    "HTML": ("html",),
    "CSS": ("css",),
    "Bash": ("bash", "shell scripting"),
    "Kotlin": ("kotlin",),
    "Swift": ("swift",),
    "React": ("react",),
    "React Native": ("react native",),
    "Redux": ("redux",),
    "Vue.js": ("vue.js", "vuejs", "vue"),
    "Next.js": ("next.js", "nextjs"),
    "Node.js": ("node.js", "nodejs", "node js"),
    "Tailwind CSS": ("tailwind css", "tailwind"),
    "GraphQL": ("graphql",),
    "REST APIs": ("rest apis", "rest api", "restful"),
    "Microservices": ("microservices", "microservice"),
    "Django": ("django",),
    "FastAPI": ("fastapi", "fast api"),
    "Flask": ("flask",),
    "Spring Boot": ("spring boot", "springboot"),
    "Maven": ("maven",),
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql",),
    "MongoDB": ("mongodb", "mongo"),
    "Redis": ("redis",),
    "Kafka": ("kafka",),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Helm": ("helm",),
    "Terraform": ("terraform",),
    "Ansible": ("ansible",),
    "Prometheus": ("prometheus",),
    "Grafana": ("grafana",),
    "CI/CD": ("ci/cd", "cicd", "ci cd"),
    "Linux": ("linux",),
    "Git": ("git",),
    "AWS": ("aws", "amazon web services"),
    "GCP": ("gcp", "google cloud"),
    "Azure": ("azure",),
    "Machine Learning": ("machine learning", "ml"),
    "Deep Learning": ("deep learning",),
    "NLP": ("nlp", "natural language processing"),
    "Computer Vision": ("computer vision",),
    "MLOps": ("mlops",),
    "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "pandas": ("pandas",),
    "Spark": ("spark", "pyspark"),
    "Airflow": ("airflow",),
    "ETL": ("etl",),
    "dbt": ("dbt",),
    "Snowflake": ("snowflake",),
    "Data Warehousing": ("data warehousing", "data warehouse"),
    "Android": ("android",),
    "iOS": ("ios",),
    "Flutter": ("flutter",),
    "Firebase": ("firebase",),
    "Selenium": ("selenium",),
    "Cypress": ("cypress",),
    "Playwright": ("playwright",),
    "pytest": ("pytest",),
    "JUnit": ("junit",),
    "Penetration Testing": ("penetration testing", "pen testing", "pentest"),
    "OWASP": ("owasp",),
    "Burp Suite": ("burp suite", "burp"),
    "SIEM": ("siem",),
    "Threat Modeling": ("threat modeling", "threat modelling"),
    "Figma": ("figma",),
    "User Research": ("user research",),
    "Prototyping": ("prototyping",),
    "Design Systems": ("design systems", "design system"),
    "Accessibility": ("accessibility", "wcag"),
}

_SKILL_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        canonical,
        re.compile(
            "|".join(
                rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])"
                for alias in aliases
            ),
            re.IGNORECASE,
        ),
    )
    for canonical, aliases in SKILL_VOCAB.items()
]


def extract_skills(text: str) -> List[str]:
    """Return canonical skills mentioned in *text* (word-boundary matched)."""
    if not text:
        return []
    return sorted(canonical for canonical, pattern in _SKILL_PATTERNS if pattern.search(text))


# --- Section splitting ------------------------------------------------------------

SECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "about", "about me", "objective",
                "what i bring", "my background"),
    "skills": ("skills", "technical skills", "core competencies", "key skills", "technologies", "tech stack",
               "toolbox"),
    "experience": ("experience", "work experience", "professional experience",
                   "employment history", "work history", "employment",
                   "career history", "where i've worked"),
    "education": ("education", "education & training", "academic background", "qualifications",
                  "studies"),
    "projects": ("projects", "key projects", "personal projects", "selected projects"),
    "certifications": ("certifications", "licenses & certifications", "licenses and certifications", "certificates"),
    "contact": ("contact", "contact information"),
}

_HEADER_LOOKUP: Dict[str, str] = {
    alias: canonical for canonical, aliases in SECTION_ALIASES.items() for alias in aliases
}


@dataclass(frozen=True)
class ResumeSection:
    """One logical resume section: canonical kind, original header, body text."""

    kind: str          # canonical name ("experience", ...) or "header"/"other"
    header: str        # header line as written in the document ("" for preamble)
    text: str


def _header_kind(line: str) -> Optional[str]:
    """Return the canonical section name if *line* looks like a section header."""
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped) > 40:
        return None
    if stripped.startswith(("-", "*", "•")):
        return None  # bullet list item — never a header, even if ALL CAPS
    known = _HEADER_LOOKUP.get(stripped.lower())
    if known:
        return known
    # Unknown but header-shaped: short ALL-CAPS line without list/date traits.
    # len >= 6 keeps headers like "AWARDS" while rejecting acronyms such as
    # "AWS" or "CISSP" that docx bullet extraction leaves as bare lines.
    if (
        stripped.isupper()
        and len(stripped) >= 6
        and len(stripped.split()) <= 5
        and "," not in stripped
        and not any(ch.isdigit() for ch in stripped)
        and not stripped.endswith(".")
    ):
        return "other"
    return None


def split_into_sections(text: str) -> List[ResumeSection]:
    """Split resume text into sections on recognised header lines.

    Anything before the first header (name / title / contact block) is returned
    as a section with ``kind="header"``.
    """
    sections: List[ResumeSection] = []
    kind, header, buf = "header", "", []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(ResumeSection(kind=kind, header=header, text=body))

    for line in (text or "").splitlines():
        found = _header_kind(line)
        if found:
            flush()
            kind, header, buf = found, line.strip().rstrip(":").strip(), []
        else:
            buf.append(line)
    flush()
    return sections


# --- Chunking ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeChunk:
    """A chunk ready for embedding: section-scoped text plus flat metadata."""

    chunk_id: str
    text: str
    section: str
    metadata: Dict[str, object]


def _split_long(body: str, max_chars: int, overlap: int) -> List[str]:
    """Split one oversized block on line boundaries, with character overlap."""
    lines = body.splitlines()
    parts: List[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars and current:
            parts.append(current)
            tail = current[-overlap:]
            cut = tail.find("\n")
            current = (tail[cut + 1:] + "\n" + line) if cut != -1 else line
        else:
            current = candidate
    if current.strip():
        parts.append(current)
    return parts


def chunk_section(section: ResumeSection, max_chars: int = MAX_CHUNK_CHARS,
                  overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Chunk a section body, preferring paragraph (blank-line) boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
        if len(current) > max_chars:
            parts = _split_long(current, max_chars, overlap)
            chunks.extend(parts[:-1])
            current = parts[-1] if parts else ""
    if current.strip():
        chunks.append(current)
    return chunks


# --- Metadata extraction --------------------------------------------------------------

# Keep _YEAR_RANGE for any external code that may reference it.
_YEAR_RANGE = re.compile(
    r"\b(19|20)(\d{2})\s*(?:-|–|to)\s*(?:((?:19|20)\d{2})|present|current|now)\b",
    re.IGNORECASE,
)
_STATED_YEARS = re.compile(r"\b(\d{1,2})\s*\+?\s*years?\b", re.IGNORECASE)

# --- Month-aware date engine ---------------------------------------------------

_MONTHS: Dict[str, int] = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

_DATE_TOKEN = (
    r"(?:[A-Za-z]{3,9}\.?\s+(?:19|20)\d{2}"   # month-name YYYY
    r"|\d{1,2}/(?:19|20)\d{2}"                  # MM/YYYY
    r"|(?:19|20)\d{2}"                           # bare YYYY
    r"|present|current|now)"                     # open-ended
)
_DATE_RANGE_V2 = re.compile(
    rf"({_DATE_TOKEN})\s*(?:-|–|—|to)\s*({_DATE_TOKEN})",
    re.IGNORECASE,
)


def _parse_date_token(token: str, *, is_end: bool,
                      today: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Parse a single date token into (year, month).

    Rules:
    - present/current/now → today
    - MM/YYYY → that month (reject if month > 12)
    - Xxx YYYY → month from _MONTHS (only if first-3-letters are a valid month name)
    - bare YYYY → month 1 (start) or 12 (end)
    Returns None if the token cannot be parsed or is invalid.
    """
    t = token.strip()
    tl = t.lower().rstrip(".")

    # present / current / now
    if tl in ("present", "current", "now"):
        return today

    # MM/YYYY
    slash_m = re.match(r"^(\d{1,2})/((?:19|20)\d{2})$", t)
    if slash_m:
        month = int(slash_m.group(1))
        year = int(slash_m.group(2))
        if month < 1 or month > 12:
            return None
        return (year, month)

    # month-name YYYY  e.g. "Jan 2020" or "January 2020"
    month_name_m = re.match(r"^([A-Za-z]{3,9})\.?\s+((?:19|20)\d{2})$", t)
    if month_name_m:
        prefix = month_name_m.group(1)[:3].lower()
        year = int(month_name_m.group(2))
        if prefix not in _MONTHS:
            return None  # "Acme 2020" and similar company-name tokens
        return (year, _MONTHS[prefix])

    # bare YYYY
    year_m = re.match(r"^((?:19|20)\d{2})$", t)
    if year_m:
        year = int(year_m.group(1))
        return (year, 12 if is_end else 1)

    return None


def extract_date_ranges(
    text: str,
    today: Optional[Tuple[int, int]] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Extract all (start, end) month-precise date ranges from *text*.

    Each element is ((start_year, start_month), (end_year, end_month)).
    Only ranges where start <= end and span <= 50 years are returned.
    """
    if today is None:
        lt = time.localtime()
        today = (lt.tm_year, lt.tm_mon)

    ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for m in _DATE_RANGE_V2.finditer(text):
        start = _parse_date_token(m.group(1), is_end=False, today=today)
        end = _parse_date_token(m.group(2), is_end=True, today=today)
        if start is None or end is None:
            continue
        # Validate: start <= end and span <= 50 years
        start_months = start[0] * 12 + start[1]
        end_months = end[0] * 12 + end[1]
        if end_months < start_months:
            continue
        if (end_months - start_months) > 50 * 12:
            continue
        ranges.append((start, end))
    return ranges

_EDUCATION_LEVELS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("phd", ("ph.d", "phd", "doctorate", "doctor of")),
    ("master", ("m.s.", "m.s ", "msc", "m.sc", "m.tech", "mtech", "master", "mba")),
    ("bachelor", ("b.s.", "b.s ", "bsc", "b.sc", "b.tech", "btech", "b.e.", "bachelor")),
)

_NAME_PATTERN = re.compile(r"^[A-Z][A-Za-z.'-]*(?: [A-Z][A-Za-z.'-]*){1,3}$")


@dataclass(frozen=True)
class ResumeMetadata:
    """Key fields extracted from one resume (stored alongside embeddings)."""

    name: str
    title: str
    skills: Tuple[str, ...]
    experience_years: int
    education_level: str   # "phd" | "master" | "bachelor" | "unknown"
    education: str
    skill_years: Tuple[Tuple[str, float], ...] = ()  # per-skill tenure (canonical, years)


def _extract_name(sections: Sequence[ResumeSection], fallback: str) -> Tuple[str, str]:
    """Return (name, title) from the preamble block, with a filename fallback."""
    preamble = next((s for s in sections if s.kind == "header"), None)
    name, title = "", ""
    if preamble:
        lines = [ln.strip() for ln in preamble.text.splitlines() if ln.strip()]
        if lines and _NAME_PATTERN.match(lines[0]) and "@" not in lines[0]:
            name = lines[0]
            if len(lines) > 1 and "@" not in lines[1] and "|" not in lines[1]:
                title = lines[1]
    if not name:
        name = fallback.replace("_", " ").replace("-", " ").title()
    return name, title


def _extract_experience_years(sections: Sequence[ResumeSection], full_text: str, *,
                              today_year: Optional[int] = None) -> int:
    """Years of experience: span of dated job ranges (month-aware), else stated 'N+ years'.

    today_year maps to (today_year, 6) — a mid-year pin that preserves exact
    expectations for year-only ranges (e.g. 2018→Present with today_year=2026
    gives 101 months → 8 years).
    """
    today_yr = today_year or time.localtime().tm_year
    today: Tuple[int, int] = (today_yr, 6)

    exp_text = "\n".join(s.text for s in sections if s.kind == "experience") or full_text
    ranges = extract_date_ranges(exp_text, today=today)

    if ranges:
        min_start = min(r[0][0] * 12 + r[0][1] for r in ranges)
        max_end = max(r[1][0] * 12 + r[1][1] for r in ranges)
        months = max_end - min_start
        return max(0, months // 12)

    stated = [int(m.group(1)) for m in _STATED_YEARS.finditer(full_text)]
    plausible = [y for y in stated if y <= 50]
    return max(plausible) if plausible else 0


def _merge_intervals(
    intervals: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Sort and merge overlapping/adjacent month-precise intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda iv: (iv[0][0] * 12 + iv[0][1]))
    merged: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        prev_start, prev_end = merged[-1]
        pe = prev_end[0] * 12 + prev_end[1]
        s = start[0] * 12 + start[1]
        e = end[0] * 12 + end[1]
        if s <= pe + 1:  # overlapping or adjacent
            new_end = end if e > pe else prev_end
            merged[-1] = (prev_start, new_end)
        else:
            merged.append((start, end))
    return merged


def extract_skill_years(
    sections: Sequence[ResumeSection],
    today: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """Compute per-skill tenure (years) from experience sections.

    Algorithm:
    1. Split experience section text into job blocks: a new block starts at any
       line matching _DATE_RANGE_V2; bullets below belong to that block.
    2. Per block: extract skills + intervals; accumulate per-skill intervals.
    3. Per skill: merge overlapping/adjacent intervals → sum months → round to 1dp.
    4. Skills found in the whole resume but not in any job block get the full
       career span (matches dataset labelling: skills not tied to a job block
       are treated as spanning the whole career).

    Note: per-skill years round to 1 decimal, while total experience in
    ``_extract_experience_years`` floors whole years — intentional, mirrors
    how the dataset labels both.
    """
    if today is None:
        lt = time.localtime()
        today = (lt.tm_year, lt.tm_mon)

    exp_text = "\n".join(s.text for s in sections if s.kind == "experience")
    full_text = "\n".join(s.text for s in sections)

    # Split exp_text into job blocks keyed on their header line
    blocks: List[Tuple[str, List[str]]] = []  # (header_line, body_lines)
    current_header = ""
    current_body: List[str] = []

    for line in exp_text.splitlines():
        if _DATE_RANGE_V2.search(line):
            # flush previous block
            if current_header or current_body:
                blocks.append((current_header, current_body))
            current_header = line
            current_body = []
        else:
            current_body.append(line)
    if current_header or current_body:
        blocks.append((current_header, current_body))

    # Accumulate per-skill intervals from blocks
    skill_intervals: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}
    skills_in_blocks: set = set()

    for header_line, body_lines in blocks:
        block_text = header_line + "\n" + "\n".join(body_lines)
        block_skills = extract_skills(block_text)
        block_ranges = extract_date_ranges(header_line, today=today)
        if not block_ranges:
            # no date range in header — check full block text
            block_ranges = extract_date_ranges(block_text, today=today)
        if not block_ranges:
            # Date-less prose block (e.g. tier-2 leading paragraph): leave its
            # skills to the career-span fallback below rather than pin them 0.0.
            continue
        for skill in block_skills:
            skills_in_blocks.add(skill)
            skill_intervals.setdefault(skill, []).extend(block_ranges)

    # Compute career span for skills present globally but not in any block
    all_resume_skills = set(extract_skills(full_text))
    career_ranges = extract_date_ranges(exp_text, today=today)
    if career_ranges:
        career_min = min(r[0][0] * 12 + r[0][1] for r in career_ranges)
        career_max = max(r[1][0] * 12 + r[1][1] for r in career_ranges)
        start_y, start_m = career_min // 12, career_min % 12
        if start_m == 0:
            start_y -= 1
            start_m = 12
        end_y, end_m = career_max // 12, career_max % 12
        if end_m == 0:
            end_y -= 1
            end_m = 12
        career_span = [((start_y, start_m), (end_y, end_m))]
    else:
        career_span = []

    # Skills in resume but not in any block → assign full career span
    for skill in all_resume_skills:
        if skill not in skills_in_blocks:
            skill_intervals.setdefault(skill, []).extend(career_span)

    # Merge intervals and compute years
    result: Dict[str, float] = {}
    for skill, intervals in skill_intervals.items():
        merged = _merge_intervals(intervals)
        total_months = sum(
            (e[0] * 12 + e[1]) - (s[0] * 12 + s[1])
            for s, e in merged
        )
        result[skill] = round(total_months / 12, 1)

    return result


def _extract_education(sections: Sequence[ResumeSection], full_text: str) -> Tuple[str, str]:
    """Return (level, detail line) for the highest degree found."""
    edu_text = "\n".join(s.text for s in sections if s.kind == "education")
    haystack = (edu_text or full_text).lower()
    level = "unknown"
    for name, needles in _EDUCATION_LEVELS:
        if any(needle in haystack for needle in needles):
            level = name
            break
    detail = ""
    if edu_text:
        detail = next((ln.strip() for ln in edu_text.splitlines() if ln.strip()), "")
    return level, detail


def extract_metadata(text: str, *, fallback_name: str = "unknown",
                     today_year: Optional[int] = None) -> ResumeMetadata:
    """Extract candidate metadata from raw resume text."""
    sections = split_into_sections(text)
    name, title = _extract_name(sections, fallback_name)
    level, detail = _extract_education(sections, text)
    today_yr = today_year or time.localtime().tm_year
    today: Tuple[int, int] = (today_yr, 6)
    sy = extract_skill_years(sections, today=today)
    return ResumeMetadata(
        name=name,
        title=title,
        skills=tuple(extract_skills(text)),
        experience_years=_extract_experience_years(sections, text, today_year=today_year),
        education_level=level,
        education=detail,
        skill_years=tuple(sorted(sy.items())),
    )


# --- Embedding backends -----------------------------------------------------------------


class EmbeddingBackend(Protocol):
    """Anything that can turn a batch of texts into unit-length vectors."""

    name: str

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


class SentenceTransformerBackend:
    """HuggingFace all-MiniLM-L6-v2 via sentence-transformers (preferred)."""

    name = f"sentence-transformers:{EMBEDDING_MODEL}"

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]


class ChromaOnnxBackend:
    """The same MiniLM model in ONNX form, shipped with ChromaDB (no torch)."""

    name = "chromadb-onnx:all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._ef(list(texts))]


def default_backend() -> EmbeddingBackend:
    """Pick the best available embedder (override with RESUME_RAG_EMBEDDER)."""
    choice = os.environ.get("RESUME_RAG_EMBEDDER", "").lower()
    if choice == "onnx":
        return ChromaOnnxBackend()
    if choice in ("", "st", "sentence-transformers"):
        try:
            return SentenceTransformerBackend()
        except ImportError:
            if choice:
                raise
    return ChromaOnnxBackend()


# --- The RAG store ------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkHit:
    """One retrieved chunk with its similarity and stored metadata."""

    chunk_id: str
    text: str
    similarity: float  # cosine similarity in [0, 1]-ish
    candidate: str
    file: str
    section: str
    metadata: Dict[str, object]


@dataclass(frozen=True)
class IndexStats:
    files_indexed: int
    chunks_indexed: int
    embed_seconds: float
    total_seconds: float
    embedder: str
    failures: Tuple[str, ...] = field(default_factory=tuple)
    llm_assisted: int = 0


class ResumeRAG:
    """Persistent ChromaDB index over section-aware resume chunks."""

    def __init__(
        self,
        resumes_dir: str = DEFAULT_RESUMES_DIR,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedder: Optional[EmbeddingBackend] = None,
        client: Optional[object] = None,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.resumes_dir = resumes_dir
        self._embedder = embedder or default_backend()
        self._client = client or chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # -- indexing ------------------------------------------------------------

    @property
    def embedder(self) -> EmbeddingBackend:
        return self._embedder

    def _resume_files(self) -> List[Dict[str, object]]:
        files = fs_tools.list_files(self.resumes_dir)
        return [f for f in files if str(f["extension"]) in fs_tools.SUPPORTED_EXTENSIONS]

    def _chunks_for_file(self, rel_path: str, content: str,
                         meta: ResumeMetadata) -> List[ResumeChunk]:
        sections = split_into_sections(content)
        stem = Path(rel_path).stem
        chunks: List[ResumeChunk] = []
        for sec_idx, section in enumerate(sections):
            label = section.header or section.kind.upper()
            for i, body in enumerate(chunk_section(section)):
                text = f"{label}\n{body}" if section.kind != "header" else body
                chunks.append(
                    ResumeChunk(
                        chunk_id=f"{stem}::{section.kind}::{sec_idx}::{i}",
                        text=text,
                        section=section.kind,
                        metadata={
                            "candidate": meta.name,
                            "file": rel_path,
                            "section": section.kind,
                            "chunk_index": i,
                            "title": meta.title,
                            "skills": ", ".join(meta.skills),
                            "exp_years": int(meta.experience_years),
                            "education_level": meta.education_level,
                            "education": meta.education,
                            "skill_years": json.dumps(dict(meta.skill_years)),
                        },
                    )
                )
        return chunks

    def build_index(self, rebuild: bool = False) -> IndexStats:
        """Load, chunk, embed and store every resume in ``resumes_dir``."""
        start = time.perf_counter()
        if rebuild and self._collection.count():
            name = self._collection.name
            self._client.delete_collection(name)
            self._collection = self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )

        base = Path(os.environ.get("FS_TOOLS_BASE_DIR", os.getcwd())).resolve()
        failures: List[str] = []
        all_chunks: List[ResumeChunk] = []
        files_ok = 0
        llm_assisted = 0
        llm_mode = os.environ.get("RESUME_RAG_LLM", "off").lower()

        for info in self._resume_files():
            result = fs_tools.read_file(str(info["path"]))
            if not result.get("success") or not str(result.get("content", "")).strip():
                failures.append(f"{info['name']}: {result.get('error', 'empty file')}")
                continue
            try:
                rel_path = Path(str(info["path"])).resolve().relative_to(base).as_posix()
            except ValueError:
                rel_path = str(info["path"])
            content = str(result["content"])
            stem = Path(rel_path).stem
            meta = extract_metadata(content, fallback_name=stem)

            # Lazy LLM fallback
            if llm_mode != "off":
                try:
                    import llm_extractor  # noqa: PLC0415
                    sections = split_into_sections(content)
                    use_llm = (llm_mode == "always") or llm_extractor.should_use_llm(meta, sections)
                    if use_llm:
                        llm_result = llm_extractor.extract_with_llm(content)
                        if llm_result is not None:
                            meta = llm_extractor.merge_metadata(meta, llm_result)
                            llm_assisted += 1
                except Exception:  # noqa: BLE001
                    pass  # silently keep regex metadata on any failure

            all_chunks.extend(self._chunks_for_file(rel_path, content, meta))
            files_ok += 1

        embed_seconds = 0.0
        if all_chunks:
            embed_start = time.perf_counter()
            vectors = self._embedder.embed([c.text for c in all_chunks])
            embed_seconds = time.perf_counter() - embed_start
            for lo in range(0, len(all_chunks), 256):
                batch = all_chunks[lo:lo + 256]
                self._collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=vectors[lo:lo + 256],
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                )

        return IndexStats(
            files_indexed=files_ok,
            chunks_indexed=len(all_chunks),
            embed_seconds=round(embed_seconds, 3),
            total_seconds=round(time.perf_counter() - start, 3),
            embedder=self._embedder.name,
            failures=tuple(failures),
            llm_assisted=llm_assisted,
        )

    # -- querying ------------------------------------------------------------

    def count(self) -> int:
        return int(self._collection.count())

    def query(
        self,
        text: str,
        k: int = 10,
        where: Optional[Dict[str, object]] = None,
        where_document: Optional[Dict[str, object]] = None,
    ) -> List[ChunkHit]:
        """Semantic search: return the top-*k* chunks for *text*."""
        if not text or not text.strip():
            raise ValueError("query text must not be empty")
        total = self.count()
        if total == 0:
            return []
        vector = self._embedder.embed([text])[0]
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(k, total),
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[ChunkHit] = []
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        ):
            hits.append(
                ChunkHit(
                    chunk_id=chunk_id,
                    text=doc,
                    similarity=round(1.0 - float(dist), 4),
                    candidate=str(meta.get("candidate", "")),
                    file=str(meta.get("file", "")),
                    section=str(meta.get("section", "")),
                    metadata=dict(meta),
                )
            )
        return hits

    def all_chunks(self) -> List[ChunkHit]:
        """Return every stored chunk (used to build the BM25 keyword index)."""
        if self.count() == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            ChunkHit(
                chunk_id=chunk_id,
                text=doc,
                similarity=0.0,
                candidate=str(meta.get("candidate", "")),
                file=str(meta.get("file", "")),
                section=str(meta.get("section", "")),
                metadata=dict(meta),
            )
            for chunk_id, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def candidate_profiles(self) -> Dict[str, Dict[str, object]]:
        """Aggregate per-candidate metadata from the stored chunks."""
        profiles: Dict[str, Dict[str, object]] = {}
        for chunk in self.all_chunks():
            meta = chunk.metadata
            file_key = str(meta.get("file", ""))
            if file_key in profiles:
                continue
            # Parse skill_years from JSON stored in chunk metadata
            raw_sy = meta.get("skill_years", "{}")
            try:
                sy_dict: Dict[str, float] = json.loads(str(raw_sy)) if raw_sy else {}
            except (json.JSONDecodeError, TypeError):
                sy_dict = {}
            profiles[file_key] = {
                "candidate": meta.get("candidate", ""),
                "title": meta.get("title", ""),
                "skills": [s for s in str(meta.get("skills", "")).split(", ") if s],
                "exp_years": int(meta.get("exp_years", 0) or 0),
                "education_level": meta.get("education_level", "unknown"),
                "education": meta.get("education", ""),
                "skill_years": sy_dict,
            }
        return profiles


# --- CLI -------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build/query the resume RAG index.")
    parser.add_argument("--resumes", default=DEFAULT_RESUMES_DIR, help="resume folder")
    parser.add_argument("--persist", default=DEFAULT_PERSIST_DIR, help="ChromaDB dir")
    parser.add_argument("--rebuild", action="store_true", help="re-index from scratch")
    parser.add_argument("--query", help="run a semantic query against the index")
    parser.add_argument("-k", type=int, default=5, help="top-k chunks for --query")
    parser.add_argument("--stats", action="store_true", help="print index stats")
    parser.add_argument("--llm", choices=["off", "auto", "always"], default=None,
                        help="LLM extraction mode (sets RESUME_RAG_LLM env var)")
    args = parser.parse_args(argv)

    if args.llm is not None:
        os.environ["RESUME_RAG_LLM"] = args.llm

    rag = ResumeRAG(resumes_dir=args.resumes, persist_dir=args.persist)

    if args.rebuild or rag.count() == 0:
        stats = rag.build_index(rebuild=args.rebuild)
        print(f"Indexed {stats.files_indexed} files -> {stats.chunks_indexed} chunks "
              f"in {stats.total_seconds}s (embedding {stats.embed_seconds}s, "
              f"backend {stats.embedder})")
        for failure in stats.failures:
            print(f"  [skip] {failure}")

    if args.stats:
        profiles = rag.candidate_profiles()
        print(f"Collection holds {rag.count()} chunks across {len(profiles)} resumes")

    if args.query:
        for hit in rag.query(args.query, k=args.k):
            snippet = hit.text.replace("\n", " ")[:110]
            print(f"{hit.similarity:.3f}  {hit.candidate:<18} {hit.section:<12} {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
