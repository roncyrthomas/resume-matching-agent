# Next Level: Hard Dataset + LLM Extraction + Reranker + UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the shipped Milestone 2 RAG system to the next level with four upgrades: (1) a hard-mode labelled resume corpus that de-saturates the metrics, (2) month-aware date parsing + per-skill tenure + Claude-assisted metadata extraction for low-confidence resumes, (3) a cross-encoder reranking stage, (4) a Streamlit UI — plus updated notebook, tests and docs telling the clean-vs-hard before/after story.

**Architecture:** The hard corpus lives beside the clean one (`resumes_hard/` + `dataset/labels_hard.json`, separate Chroma collection) so the M2 demo stays intact. Extraction gains a months-precision date engine and job-block skill attribution; when regex confidence is low, an Anthropic tool-use call (cached on disk, offline-safe fallback) fills the gaps. The matcher gains an optional cross-encoder rerank stage (lazy-loaded, off by default in the library so tests stay network-free; on by default in the UI). The UI imports `JobMatcher` directly — no API layer.

**Tech Stack:** existing stack (Python 3.14, chromadb 1.5.9, sentence-transformers 5.5.1, rank-bm25, fpdf2/python-docx, pytest, nbclient) + `streamlit` (fallback: `gradio` if cp314 wheels fail) + `cross-encoder/ms-marco-MiniLM-L-6-v2` (via sentence-transformers `CrossEncoder`) + `anthropic` SDK (already installed, key in `.env`).

---

## CURRENT REPO STATE (read first)

Milestone 2 is SHIPPED and reviewed (49 tests green, commits through `1d12850`). Key facts:

| Artifact | State |
|---|---|
| `resume_rag.py` | Section-aware RAG: `split_into_sections`, `chunk_section`, `extract_skills`, `extract_metadata(text, *, fallback_name, today_year)`, `ResumeRAG(resumes_dir, persist_dir, collection_name, embedder, client)` with `build_index/query/all_chunks/candidate_profiles/count`, `SKILL_VOCAB` (75 skills). Index at `.chroma/` collection `resumes` (231 chunks, 36 resumes) |
| `job_matcher.py` | `parse_job_description`, `MustHave(kind, raw, skills, years, level, any_of)`, `check_must_haves(profile, must_haves)`, `JobMatcher(rag, semantic_weight)` → `.match(jd_text, k=10, *, apply_filters=True, semantic_only=False)`, `.match_file`. Weights: W_RETRIEVAL .45 / W_SKILLS .35 / W_EXPERIENCE .15 / W_NICE .05; retrieval = .65 semantic + .35 BM25 |
| `scripts/make_dataset.py` | Deterministic generator: 36 resumes (`CANDIDATES` list), `ROLES` catalogue, `SECTION_STYLES` (caps/title/alt), `_make_jobs` (contiguous spans, most-recent "Present"), writers for txt/pdf/docx, 6 JDs, `dataset/labels.json` |
| `tests/` | 49 tests: `test_fs_tools` (14), `test_resume_rag` (16), `test_job_matcher` (19); `tests/rag_test_utils.py` has `FakeEmbedder` (crc32 bag-of-words, 64-dim) and `make_corpus(tmp_path, monkeypatch)` (unique collection per tmp_path — EphemeralClient is a process singleton) |
| `scripts/build_notebook.py` → `rag_analysis.ipynb` | Executed: metadata 100%, hybrid soft P@5 .967 / MRR 1.0 / R@10 .921 vs semantic .886, latency p50 30.2ms |
| `.env` | `ANTHROPIC_API_KEY` present (Milestone 1). `llm_file_assistant.py` shows the tool-use pattern + mocked-client test pattern |

**Environment:** Windows 11 PowerShell (NO `&&` — use `;`), cwd `D:\Airtribe\fs-tools`, `python` = 3.14.3. Conventional commits, NO attribution footers. HF model downloads print harmless stderr warnings.

**Hard rules for every task:**
- The existing 49 tests must pass UNCHANGED at the end of every task (`pytest -q`). New behaviour must be additive/opt-in.
- Library defaults stay offline-safe: no network in tests (LLM extraction mocked; reranker injected fake; real models only in CLI/UI/notebook paths).
- If a test you add fails, fix the implementation (tests below encode the spec) unless it's a mere numeric threshold.

---

### Task 1: Install + pin UI dependency, decide framework

**Files:** Modify `requirements.txt`

- [ ] **Step 1:** `pip install -q --no-input streamlit; python -c "import streamlit; print('streamlit', streamlit.__version__)"`
  - If install fails on missing cp314 wheels (likely culprit: pyarrow): `pip install -q --no-input gradio; python -c "import gradio; print('gradio', gradio.__version__)"` and use gradio everywhere Task 5 says streamlit.
- [ ] **Step 2:** Append to `requirements.txt` under a new section `# Next-level — UI`: `streamlit>=1.40` (or `gradio>=5` if that's the outcome). Run `pip install -q -r requirements.txt` to confirm resolution.
- [ ] **Step 3:** `pytest -q` → 49 passed (nothing else changed).
- [ ] **Step 4:** Commit: `chore: add UI framework dependency`
- [ ] **Report** which framework won — Task 5 depends on it.

---

### Task 2: Hard-mode labelled corpus (`--hard`)

**Files:** Modify `scripts/make_dataset.py`; generated: `resumes_hard/` (40 files), `dataset/labels_hard.json`; modify `.gitignore` only if needed (hard corpus IS committed, like the clean one).

Extend the generator with a `--hard` CLI flag (argparse; default behaviour without the flag must stay byte-identical — guard every new branch behind the flag/profile).

**Hard-corpus spec (40 resumes, seeded `random.Random(4242)`, TODAY_YEAR 2026):**
- Reuse `ROLES`/`COMPANIES`/`SCHOOLS`/`CITIES` and the role mix (≥3 per major role: ml, backend_python, backend_java, frontend, fullstack, devops, data_engineer, mobile, qa, security, design). New candidate names (40, ASCII) — do NOT reuse clean-corpus names (keeps eval corpora independent).
- Three difficulty tiers recorded per resume in labels (`"tier": 0|1|2`):
  - **tier 0 — clean** (~14): exactly the existing renderer.
  - **tier 1 — messy** (~14): nonstandard headers sampled from `{"Career History", "Where I've Worked", "What I Bring", "Toolbox", "Studies", "My Background"}` mapped in labels to their canonical kinds (experience/skills/skills/skills/education/summary); month-level date formats (see below); summary as a 2-3 sentence prose paragraph.
  - **tier 2 — hard** (~12): NO skills section (skills appear only inside experience bullets and the prose summary); education line present for only half (others: `education_level: "unknown"` in labels); no header at all before experience for some (prose flows straight into job blocks); contact info mashed into one line.
- **Date formats** (per-resume style, mixed across the corpus): `"Jan 2020 - Mar 2023"` / `"January 2020 to March 2023"` (month names), `"03/2019 - 11/2022"` and `"03/2019 - present"` (MM/YYYY), plus the existing `"2019 - 2021"`. Jobs remain contiguous; most recent ends `Present`/`present`/`Current` (vary the casing/word).
- **Ground truth per resume** in `labels_hard.json` (same top-level schema as labels.json, plus): `tier`, `total_years` (int, floor of months/12 across the career span), and `skill_years`: a dict mapping each skill to years (1 decimal), computed by the generator as the merged duration of jobs whose bullets/title mention that skill. Core skills not mentioned in any bullet get attributed to ALL jobs (they're in the skills line/summary) — document this rule in a comment; the extractor task mirrors it only for section-listed skills.
- JDs are NOT regenerated — the same 6 JDs serve both corpora. `labels_hard.json` copies the `job_descriptions` block from labels.json.

- [ ] **Step 1:** Implement; `python scripts\make_dataset.py --hard` → `Generated 40 hard resumes ... labels_hard.json`.
- [ ] **Step 2:** Verify: `python scripts\make_dataset.py` (no flag) leaves `git status` clean for `resumes/` and `dataset/labels.json` (byte-identical regeneration — determinism guard).
- [ ] **Step 3:** Integrity probe: every labels_hard file exists; 40 resumes; tiers distributed ~14/14/12; at least 10 resumes contain a month-name or MM/YYYY date; at least 8 have no skills section (grep for absence of all skills-header aliases); parse all 40 through `fs_tools.read_file` successfully.
- [ ] **Step 4:** `pytest -q` → 49 passed. Commit: `feat: hard-mode labelled corpus (40 resumes, 3 difficulty tiers)`

---

### Task 3: Extraction v2 — month dates, per-skill tenure, Claude fallback

**Files:** Modify `resume_rag.py`; Create `llm_extractor.py`; Create `tests/test_extraction_v2.py`; Modify `tests/rag_test_utils.py` only if a new helper is needed.

**3a. Month-aware date engine (resume_rag.py).** Replace the year-only logic inside `_extract_experience_years` with a shared parser (new module-level functions):

```python
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
# token forms: "Jan 2020" / "January 2020" / "03/2019" / "2019" / "present|current|now"
_DATE_TOKEN = r"(?:[A-Za-z]{3,9}\.?\s+(?:19|20)\d{2}|\d{1,2}/(?:19|20)\d{2}|(?:19|20)\d{2}|present|current|now)"
_DATE_RANGE_V2 = re.compile(rf"({_DATE_TOKEN})\s*(?:-|–|—|to)\s*({_DATE_TOKEN})", re.IGNORECASE)

def _parse_date_token(token: str, *, is_end: bool, today: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """-> (year, month). Year-only: month=1 for starts, 12 for ends. None if not a date."""
```
Rules: month-name → `_MONTHS[token[:3].lower()]` (guard: only if the 3-letter prefix is in `_MONTHS`, else None — rejects "Acme 2020"); `MM/YYYY` → that month (reject MM>12); bare year → 1 or 12 by `is_end`; present/current/now → `today`. `extract_date_ranges(text, today) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]` drops ranges where end < start or span > 50y. Experience years = `max(0, (max_end_ym - min_start_ym) months // 12)` (floor). `today` defaults to `(time.localtime().tm_year, time.localtime().tm_mon)`; `extract_metadata` keeps its `today_year` kwarg (maps to `(today_year, 6)` — mid-year pin keeps the existing tests' exact integer expectations: 2018→Present(2026,6) = 101 months → 8 ✓; 2016→Present = 125 → 10 ✓).

**3b. Per-skill tenure (resume_rag.py).** New function + metadata field:

```python
def extract_skill_years(sections, today) -> Dict[str, float]:
    """Split the experience section into job blocks (a block starts at a line
    containing a date range; bullets that follow belong to it). For each block:
    skills = extract_skills(block_text). Per skill, merge the blocks' (start, end)
    intervals (classic sorted-merge) and sum to years, rounded to 1 decimal."""
```
`ResumeMetadata` gains `skill_years: Tuple[Tuple[str, float], ...]` (frozen-safe). Chroma chunk metadata gains `"skill_years": json.dumps(dict(...))`; `candidate_profiles()` parses it back to a dict. Skills found in the resume but in NO dated block fall back to total experience years (matches the generator's labelling rule for section-listed skills).

**3c. Claude fallback (`llm_extractor.py`, new ~180-line module).**

```python
LLM_TOOL = {
    "name": "record_resume_metadata",
    "description": "Record structured metadata extracted from one resume.",
    "input_schema": {"type": "object", "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "experience_years": {"type": "number"},
        "education_level": {"type": "string", "enum": ["phd", "master", "bachelor", "unknown"]},
        "education": {"type": "string"},
        "skill_years": {"type": "object", "additionalProperties": {"type": "number"}},
    }, "required": ["name", "skills", "experience_years", "education_level"]},
}

def should_use_llm(meta: "ResumeMetadata", sections) -> bool:
    # low confidence: name fell back to the filename, OR no experience section
    # was detected, OR zero experience years, OR no skills found
def extract_with_llm(text, *, client=None, model=None, cache_dir=".cache/llm_extract") -> Optional[dict]:
    # sha1(model + text) disk cache (JSON) -> skip API on hit
    # client = client or anthropic.Anthropic() (key from env/.env via dotenv)
    # messages.create(model=os.environ.get("RESUME_RAG_LLM_MODEL", "claude-haiku-4-5-20251001"),
    #   max_tokens=1024, tools=[LLM_TOOL], tool_choice={"type": "tool", "name": "record_resume_metadata"},
    #   messages=[{"role": "user", "content": f"Extract metadata from this resume:\n\n{text[:8000]}"}])
    # return the tool_use block's input dict; on ANY exception return None (offline-safe)
def merge_metadata(regex_meta: "ResumeMetadata", llm: dict) -> "ResumeMetadata":
    # LLM fills gaps only: name if regex fell back, education if unknown, years if 0,
    # skills = union restricted to SKILL_VOCAB canonicals (case-insensitive match via
    # resume_rag.extract_skills(", ".join(llm_skills))), skill_years merged the same way.
```
Integration in `ResumeRAG.build_index`: env `RESUME_RAG_LLM` = `off` (default for safety) / `auto` (call only when `should_use_llm`) / `always`. `IndexStats` gains `llm_assisted: int = 0`. CLI `resume_rag.py` gains `--llm {off,auto,always}` which sets the env var before indexing.

**3d. Matcher uses per-skill tenure (`job_matcher.py`).** In `check_must_haves`, for `kind == "skill_years"`: if the profile has a non-empty `skill_years` dict, the check becomes `max(skill_years.get(s, 0.0) for s in mh.skills) >= mh.years`, failure message `f"has {best:.1f} yrs of {skill} (needs {mh.years}+) ({mh.raw})"`; otherwise the existing total-years approximation (message format UNCHANGED — existing tests assert `"needs 5+" in r`... keep the substring `needs {mh.years}+` in the new message too, which the format above does).

**3e. Tests — create `tests/test_extraction_v2.py` with exactly:**

```python
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
```

Note `test_skill_years_attribution_and_merging` expected values: Jan 2022→Jun 2026 = 53 months ≈ 4.4y; Mar 2019→Nov 2021 = 32 months ≈ 2.7y. If your interval arithmetic lands within the stated tolerance, pass; outside it, fix the arithmetic, not the tolerance.

- [ ] **Step 1:** Implement 3a-3d. **Step 2:** Create the test file; `pytest tests\test_extraction_v2.py -q` → 9 passed.
- [ ] **Step 3:** Full suite `pytest -q` → 58 passed (49 + 9) — the old `_YEAR_RANGE`-based tests must still pass via the new engine.
- [ ] **Step 4:** Live check on the hard corpus (no LLM): `python -c "from resume_rag import ResumeRAG; import json,pathlib; rag=ResumeRAG(resumes_dir='resumes_hard', collection_name='resumes_hard'); s=rag.build_index(rebuild=True); print(s.files_indexed, s.chunks_indexed); L=json.loads(pathlib.Path('dataset/labels_hard.json').read_text())['resumes']; p=rag.candidate_profiles(); years_ok=sum(1 for f,v in L.items() if p.get('resumes_hard/'+f, {}).get('exp_years')==v['total_years']); print('years exact:', years_ok, '/', len(L))"` → expect 40 files and years exact ≥ 30 (tier-2 prose resumes may miss; that's the LLM's job). Report the number.
- [ ] **Step 5:** Optional live LLM smoke (key exists): `$env:RESUME_RAG_LLM='auto'; python resume_rag.py --resumes resumes_hard --rebuild` then check `llm_assisted` in the printed stats; afterwards `Remove-Item env:RESUME_RAG_LLM`. If the API errors, report it but do not block (offline path is the contract).
- [ ] **Step 6:** Commit: `feat: month-aware dates, per-skill tenure, Claude-assisted extraction`

---

### Task 4: Cross-encoder reranking stage

**Files:** Create `reranker.py` (~80 lines); Modify `job_matcher.py`; Create `tests/test_reranker.py`.

**`reranker.py`:**

```python
"""Cross-encoder reranking: scores (JD, chunk) PAIRS jointly — slower but far
more precise than bi-encoder cosine. Lazy-loaded; never imported by default."""
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CrossEncoderReranker:
    name = f"cross-encoder:{RERANK_MODEL}"
    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)
    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Sigmoid-squashed logits in [0, 1]; [] for empty input."""
```

**`job_matcher.py` integration:**
- Ctor: `JobMatcher(rag=None, semantic_weight=SEMANTIC_WEIGHT, reranker=None, rerank=False)` — `rerank=False` default keeps the library offline (existing tests construct `JobMatcher(rag=rag)`); when `rerank=True` and `reranker is None`, lazily build `CrossEncoderReranker` on first match.
- `match(..., *, apply_filters=True, semantic_only=False, rerank=None)` — `None` → ctor default; `semantic_only=True` forces rerank off (it's a pure-embedding ablation).
- Mechanics: after `_semantic_by_candidate`, take the pooled chunks' top `RERANK_PAIRS = 40` by cosine, score pairs, sigmoid → per-candidate best rerank score. Retrieval blend with rerank ON: `0.40*semantic + 0.20*bm25 + 0.40*rerank`; OFF: existing `0.65/0.35`. Add `"rerank": <ms>` to `latency_ms` (0.0 when off) and `"rerank"` to `score_breakdown` (0.0 when off). `query.mode` becomes `"hybrid+rerank"` when on.
- CLI: `--rerank` flag.

**`tests/test_reranker.py` with exactly:**

```python
"""Tests for the rerank stage using an injected fake cross-encoder."""

from __future__ import annotations

from job_matcher import JobMatcher
from tests.rag_test_utils import make_corpus

ML_JD = """Job Title: Machine Learning Engineer

Requirements:
- Experience with PyTorch and pandas.
"""


class FakeReranker:
    name = "fake-reranker"
    def __init__(self):
        self.calls = 0
    def score(self, query, texts):
        self.calls += 1
        # deterministic: reward exact 'pytorch' mentions, hard zero otherwise
        return [1.0 if "pytorch" in t.lower() else 0.0 for t in texts]


def _matcher(tmp_path, monkeypatch, **kw):
    rag = make_corpus(tmp_path, monkeypatch)
    rag.build_index(rebuild=True)
    return JobMatcher(rag=rag, **kw)


def test_rerank_off_by_default(tmp_path, monkeypatch):
    fake = FakeReranker()
    m = _matcher(tmp_path, monkeypatch, reranker=fake)
    res = m.match(ML_JD, k=5)
    assert fake.calls == 0
    assert res["latency_ms"]["rerank"] == 0.0
    assert res["query"]["mode"] == "hybrid"


def test_rerank_changes_breakdown_and_mode(tmp_path, monkeypatch):
    fake = FakeReranker()
    m = _matcher(tmp_path, monkeypatch, reranker=fake, rerank=True)
    res = m.match(ML_JD, k=5, apply_filters=False)
    assert fake.calls == 1
    assert res["query"]["mode"] == "hybrid+rerank"
    top = res["top_matches"][0]
    assert top["candidate_name"] == "Riley Carter"          # only resume mentioning PyTorch
    assert top["score_breakdown"]["rerank"] == 1.0
    others = [m_ for m_ in res["top_matches"][1:]]
    assert all(m_["score_breakdown"]["rerank"] == 0.0 for m_ in others)


def test_semantic_only_forces_rerank_off(tmp_path, monkeypatch):
    fake = FakeReranker()
    m = _matcher(tmp_path, monkeypatch, reranker=fake, rerank=True)
    res = m.match(ML_JD, k=5, semantic_only=True)
    assert fake.calls == 0
    assert res["query"]["mode"] == "semantic_only"


def test_rerank_per_call_override(tmp_path, monkeypatch):
    fake = FakeReranker()
    m = _matcher(tmp_path, monkeypatch, reranker=fake, rerank=False)
    res = m.match(ML_JD, k=5, rerank=True)
    assert fake.calls == 1 and res["query"]["mode"] == "hybrid+rerank"
```

- [ ] **Step 1:** Implement. **Step 2:** `pytest tests\test_reranker.py -q` → 4 passed. **Step 3:** `pytest -q` → 62 passed.
- [ ] **Step 4:** Live smoke (downloads ~80MB once): `python job_matcher.py job_descriptions\senior_ml_engineer.txt -k 5 --rerank` → JSON, mode `hybrid+rerank`, rerank latency reported, ML candidates on top.
- [ ] **Step 5:** Commit: `feat: cross-encoder reranking stage (opt-in)`

---

### Task 5: UI (`app.py`)

**Files:** Create `app.py` (~300-380 lines, single file; extract `_ui_helpers.py` ONLY if you exceed ~450 lines). Use the framework Task 1 reported (instructions below assume Streamlit; mirror features 1:1 in gradio if that's the outcome).

**Feature checklist (all required):**
1. `st.set_page_config(page_title="Resume Matcher", layout="wide")`; title + one-line pipeline description.
2. Sidebar: corpus selector (radio: `resumes` "Clean (36)" / `resumes_hard` "Hard (40)") → maps to `collection_name` = dir name; K slider (1-20, default 10); mode radio (Hybrid + rerank / Hybrid / Semantic only); semantic-weight slider (0.0-1.0, default 0.65, disabled unless Hybrid); "Apply must-have filters" checkbox (default on); "Rebuild index" button (spinner + `build_index(rebuild=True)` + report files/chunks/llm_assisted); "Use Claude for tricky resumes" checkbox → sets `RESUME_RAG_LLM=auto` env for the rebuild.
3. Cached resources: `@st.cache_resource def get_rag(corpus): ...` and `def get_matcher(corpus, weight): ...` (reranker instance cached separately so toggling mode doesn't reload the model). If the selected collection is empty, build it on first use with a spinner.
4. Main pane — JD input: `st.text_area` (pre-filled with `job_descriptions/senior_ml_engineer.txt` content) + `st.selectbox` to load any file from `job_descriptions/` + `st.file_uploader` (txt/pdf/docx; persist via `fs_tools.write_file` to `job_descriptions/uploads/<name>` for txt, raw bytes via plain write for pdf/docx then read back through `fs_tools.read_file`). "Match" button.
5. Results: for each match a bordered container: rank + name + `st.metric("Score", ...)`; matched skills as pills (`st.markdown` with backticks is fine); `score_breakdown` as `st.progress` bars labelled semantic / keyword / rerank / skill coverage / experience fit; excerpts in an expander with the `[SECTION]` prefix bolded; reasoning as caption. Below: "Filtered out (N)" expander listing names + `failed_requirements`.
6. Latency footer: semantic / keyword / rerank / total ms from `latency_ms`.
7. Resume upload (sidebar or tab): file_uploader (txt/pdf/docx) → write into the SELECTED corpus dir (sandbox-safe: reject names with path separators; use `Path(name).name`) → `build_index(rebuild=False)` upserts → toast "indexed N chunks for <name>"; the new candidate is immediately matchable.
8. NO direct `open()` for resume/JD content — route reads through `fs_tools.read_file` (binary uploads may use `Path.write_bytes` to land the file, then fs_tools for reading).

- [ ] **Step 1:** Implement. **Step 2:** Headless smoke: `streamlit run app.py --server.headless true --server.port 8599` in background, wait ~10s, `Invoke-WebRequest http://localhost:8599 -UseBasicParsing` → 200, then kill the process. Also `python -c "import ast; ast.parse(open('app.py',encoding='utf-8').read()); print('parse OK')"`.
- [ ] **Step 3:** `pytest -q` → still 62 passed. **Step 4:** Commit: `feat: Streamlit UI (match, explain, filter, upload, rerank toggle)`

---

### Task 6: Notebook v2 — the before/after story

**Files:** Modify `scripts/build_notebook.py`; re-execute → `rag_analysis.ipynb`.

Add AFTER the existing §7 (keep §1-7 and the conclusions cell LAST):
- **§8 Hard corpus, harder numbers:** build `resumes_hard` index (collection `resumes_hard`, `RESUME_RAG_LLM` off) timed; run the §4 `evaluate()` for THREE modes on BOTH corpora: `semantic-only`, `hybrid`, `hybrid+rerank` (`rerank=True`) → one summary DataFrame indexed (corpus, mode) for soft P@5 / R@10 / MRR / hit@1 + a grouped bar chart (soft P@5 by corpus × mode). Note: pass a fresh `JobMatcher(rag=hard_rag, rerank=...)` per mode; reuse one CrossEncoderReranker instance across modes to download once.
- **§9 Extraction accuracy on the hard corpus:** regex-only profiles vs `labels_hard.json` → name/years/education accuracy + per-skill-years MAE (compare `skill_years` dicts on intersecting keys). THEN, if `ANTHROPIC_API_KEY` is present, rebuild with `RESUME_RAG_LLM=auto` and show the same table side by side + `llm_assisted` count; if the key is missing or the call fails, print "LLM pass skipped (no key)" and continue (cells must not error).
- **§10 Rerank latency:** hard corpus, 6 JDs × 3 repeats with rerank on → extend the latency table with the `rerank` column; one sentence comparing p50 with/without.
- Update the conclusions cell with the measured hard-corpus numbers: state plainly whether rerank beats hybrid on the hard corpus and by how much, and what the LLM pass recovered (use REAL numbers from this run — overclaiming is a review-blocker; the M2 review specifically fixed that once).

- [ ] **Step 1:** Edit builder. **Step 2:** `python scripts\build_notebook.py` (allow ~10 min: two index builds + cross-encoder download + ~40 matcher runs). **Step 3:** integrity probe (≥14 code cells with outputs, 0 errors). **Step 4:** Report the §8 table + §9 accuracies verbatim — Task 7 needs them. **Step 5:** Commit: `docs: notebook v2 — clean-vs-hard, rerank and LLM-extraction results`

---

### Task 7: Docs + final verification

**Files:** Modify `README.md`; final checks.

- [ ] **Step 1:** README: add `app.py`, `resumes_hard/`, `llm_extractor.py`, `reranker.py` to the layout tree; add a "Next level" section after the M2 results: how to run the UI (`streamlit run app.py`), the hard-corpus results table (REAL numbers from Task 6's report: clean vs hard × semantic/hybrid/hybrid+rerank soft P@5, extraction accuracy regex vs +LLM, rerank latency cost), `RESUME_RAG_LLM` env documentation, and a 4-line addendum to the demo-video script (show the UI matching + uploading a resume live, show the hard-corpus chart).
- [ ] **Step 2:** `pytest -q` → 62 passed. Fresh grader path: `python scripts\make_dataset.py --hard; python resume_rag.py --resumes resumes_hard --rebuild --stats; python job_matcher.py job_descriptions\data_platform_engineer.txt -k 10 --rerank` → sane data-engineer ranking.
- [ ] **Step 3:** `git status --short` clean; `git log --oneline` shows the ~7 new conventional commits, no attribution footers.
- [ ] **Step 4:** Commit: `docs: next-level README (UI, hard corpus, rerank, LLM extraction)`
- [ ] **Step 5:** FINAL REVIEW (controller dispatches): whole-delta review vs this plan + the deliverables checklist; verdict SHIP/FIX-FIRST.

---

## Self-review notes

- **Coverage vs "do all 4":** UI → Task 5; real-data hardening → Task 2 (labelled hard corpus — chosen over unlabelled external resumes so retrieval metrics stay computable; UI upload covers true real files); LLM extraction → Task 3 (+ per-skill tenure, which also fixes the documented M2 limitation); reranker → Task 4. Notebook/docs carry the evidence.
- **Backward compatibility is a hard rule:** 49 existing tests untouched; rerank + LLM are opt-in; clean dataset regeneration stays byte-identical (Task 2 Step 2 guards it).
- **Offline-safety:** all new tests use mocks/fakes; live model/API calls happen only in CLI/UI/notebook steps.
- **Type consistency:** new APIs named exactly once and reused: `extract_date_ranges`, `extract_skill_years`, `ResumeMetadata.skill_years`, `llm_extractor.{should_use_llm, extract_with_llm, merge_metadata, LLM_TOOL}`, `CrossEncoderReranker.score`, `JobMatcher(reranker=, rerank=)`, `match(rerank=)`, `latency_ms["rerank"]`, `score_breakdown["rerank"]`, modes `hybrid+rerank`.
