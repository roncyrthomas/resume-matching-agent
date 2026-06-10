# Resume File-System Tools + LLM Assistant

A small Python project that exposes four file-system tools and lets an LLM
(Anthropic Claude) call them to answer natural-language questions about a folder
of resume files.

- **Part A — `fs_tools.py`**: `read_file`, `list_files`, `write_file`,
  `search_in_file`. Each returns a structured response and handles errors
  gracefully.
- **Part B — `llm_file_assistant.py`**: registers those tools with Claude's
  tool-use API and runs a query loop so the model can read, list, search and
  write files on your behalf.

## Project layout

```
fs-tools/
├── fs_tools.py             # M1 Part A: the four file-system tools
├── llm_file_assistant.py   # M1 Part B: Anthropic tool-use loop + CLI
├── resume_rag.py           # M2 Part A: chunking + embeddings + ChromaDB
├── job_matcher.py          # M2 Part B: hybrid search, scoring, filtering
├── llm_extractor.py        # NL: Claude-assisted extraction fallback (cached)
├── reranker.py             # NL: cross-encoder rerank stage (opt-in)
├── app.py                  # NL: Streamlit UI (python -m streamlit run app.py)
├── rag_analysis.ipynb      # executed experiments (accuracy, latency, ablations)
├── scripts/make_samples.py # generates the original 8 dummy resumes
├── scripts/make_dataset.py # 36 labelled resumes + 6 JDs (+ --hard corpus)
├── scripts/build_notebook.py # builds + executes rag_analysis.ipynb
├── resumes/                # 36 clean resumes (.txt/.pdf/.docx) [generated]
├── resumes_hard/           # 40 hard-mode resumes, 3 difficulty tiers [generated]
├── job_descriptions/       # 6 job descriptions [generated]
├── dataset/                # labels.json + labels_hard.json ground truth [generated]
├── tests/                  # pytest suites (fs tools, RAG, matcher)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

Requires Python 3.9+.

```powershell
# 1. (recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell on Windows
# source .venv/bin/activate          # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. configure your API key
copy .env.example .env               # cp .env.example .env on macOS/Linux
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 4. generate the sample resumes
python scripts/make_samples.py
```

## Part A — the tools

```python
import fs_tools

fs_tools.read_file("resumes/john_doe.pdf")
# {'success': True, 'filepath': '...john_doe.pdf', 'content': '...',
#  'metadata': {'size': 1234, 'modified': '2026-05-30T12:00:00',
#               'extension': '.pdf', 'unit': 'pages', 'count': 1}}

fs_tools.list_files("resumes", extension=".pdf")
# [{'name': 'john_doe.pdf', 'path': '...', 'size': 1234,
#   'modified': '...', 'extension': '.pdf'}, ...]

fs_tools.write_file("summaries/john.txt", "Strong Python backend candidate.")
# {'success': True, 'filepath': '...john.txt', 'bytes_written': 32}

fs_tools.search_in_file("resumes/alex_lee.txt", "python")
# {'success': True, 'keyword': 'python', 'match_count': 2,
#  'matches': [{'char_start': 41, 'char_end': 47, 'snippet': '...',
#               'line_number': 6}, ...]}
```

**Behaviour & safety**

- Supported formats: `.txt`, `.pdf` (pdfplumber, with a pypdf fallback) and
  `.docx` (paragraphs **plus** tables and headers/footers).
- All paths are confined to a base directory (the project root by default, or
  `$FS_TOOLS_BASE_DIR`). Attempts to escape it are rejected — the LLM cannot read
  arbitrary system files.
- Text is decoded as UTF-8 with `errors="replace"`.
- Tools never raise to the caller: failures come back as
  `{"success": False, "error": "..."}` (and `list_files` returns `[]`).
- `search_in_file` reports **character offsets** (reliable for extracted PDF
  text) plus a `line_number` for `.txt`/`.docx`.

## Part B — the LLM assistant

```powershell
python llm_file_assistant.py "Read all resumes in the resumes folder"
python llm_file_assistant.py "Find resumes mentioning Python experience"
python llm_file_assistant.py "Create a summary file for resumes/john_doe.pdf"
```

Each tool call is printed as it happens, so you can watch Claude orchestrate the
tools. The loop:

1. Sends the query plus the four tool definitions to Claude.
2. While Claude responds with `stop_reason == "tool_use"`, executes each
   requested tool and returns the result (failures are flagged with
   `is_error: true` so Claude can recover).
3. Stops when Claude returns a final text answer, or after `MAX_ITERATIONS`
   (10) rounds as a safety cap.

Configuration (via `.env` or environment):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | yes | — | Anthropic API key |
| `ANTHROPIC_MODEL` | no | `claude-sonnet-4-6` | Model id |
| `FS_TOOLS_BASE_DIR` | no | current directory | Sandbox root for file access |

## Tests

```powershell
pytest -q
```

The suite covers each tool's success and error paths (missing file, unsupported
extension, path traversal, empty keyword) and the assistant loop with a mocked
Anthropic client (tool execution, `is_error` propagation, and the
max-iterations guard) — no API key or network required.

## Demo video

A 2–3 minute walkthrough should show: generating the samples, running each of
the three example queries, and the printed tool calls Claude makes for each.

---

# Milestone 2 — Resume RAG + Job Matching Engine

Builds on the Milestone 1 tools: resumes are **loaded via `fs_tools`**, chunked
section-aware, embedded with HuggingFace **all-MiniLM-L6-v2** and stored in a
persistent **ChromaDB** collection with extracted metadata. A hybrid matcher
ranks resumes against a job description with 0-100 scores, reasoning and
must-have filtering.

## Quickstart

```powershell
pip install -r requirements.txt
python scripts\make_dataset.py            # 36 resumes + 6 JDs + ground truth
python resume_rag.py --rebuild --stats    # build the vector index (.chroma/)
python job_matcher.py job_descriptions\senior_ml_engineer.txt -k 10
python scripts\build_notebook.py          # rebuild + execute rag_analysis.ipynb
pytest -q                                 # full test suite (49 tests)
```

## Part A — `resume_rag.py`

1. **Load** — `fs_tools.list_files` / `fs_tools.read_file` (txt/pdf/docx, sandboxed).
2. **Chunk** — split on section headers (3 header styles supported:
   `SUMMARY` / `Professional Summary` / `PROFILE`, ...), then paragraph-packed
   to ~1100 chars with 150 overlap. A chunk never mixes two sections; bullet
   items and bare acronyms are never mistaken for headers.
3. **Metadata** — per resume: name (header block), skills (75-entry canonical
   vocabulary with aliases, word-boundary matched so `Java` ≠ `JavaScript`),
   experience years (dated ranges like `2018 - Present`, fallback "N+ years"),
   education level (PhD/Master/Bachelor).
4. **Embed + store** — `all-MiniLM-L6-v2` via sentence-transformers (or
   ChromaDB's ONNX build of the same model: set `RESUME_RAG_EMBEDDER=onnx`),
   persisted in ChromaDB (cosine space) with metadata on every chunk, so
   queries can filter, e.g. `where={"exp_years": {"$gte": 5}}`.

## Part B — `job_matcher.py`

- **JD parsing** — title, skills, and structured must-haves from the
  Requirements section: `5+ years Python` → skill+years, `FastAPI or Django`
  → any-of group, `Bachelor's degree` → education floor. JDs without a
  Requirements section fall back to experience-cued patterns only, so
  "a 10+ years old company" never becomes a fake requirement.
- **Hybrid retrieval** — semantic (ChromaDB) + BM25 keyword scores blended
  `0.65 / 0.35` per candidate; keyword evidence anchors critical skills.
- **Scoring (0-100)** — `45%` retrieval strength + `35%` required-skill
  coverage + `15%` experience fit + `5%` nice-to-haves, with a per-component
  `score_breakdown` on every match.
- **Filtering** — candidates failing any must-have are excluded from
  `top_matches` and listed in `filtered_out` with exact reasons. Per-skill
  tenure is approximated as *has skill + total years ≥ N* (documented).
- **Output** — the assignment's JSON shape (`job_description`, `top_matches`
  with `candidate_name`, `resume_path`, `match_score`, `matched_skills`,
  `relevant_excerpts`, `reasoning`) plus extras (`filtered_out`, `latency_ms`).

## Results (from the executed `rag_analysis.ipynb`)

| Metric | Value |
|---|---|
| Metadata extraction (name / years / education / skills recall) | 100% on all four (36 resumes) |
| Retrieval P@5, soft relevance (hybrid) | 0.967 |
| Retrieval MRR / hit@1 (hybrid, soft) | 1.000 / 1.000 |
| Recall@10, soft (hybrid vs semantic-only) | 0.921 vs 0.886 |
| Query latency p50 / p95 (end-to-end match) | 30.2 ms / 35.6 ms |
| Index build (36 resumes → 231 chunks) | ~2 s one-off |

This clean synthetic corpus saturates ranking quality for both modes
(identical P@5/MRR); hybrid's measured gain is soft Recall@10, where BM25's
exact-term anchoring pulls extra adjacent-role candidates into the top-10.
The weight-sweep ablation is flat, so the default `w=0.65` is safe. Full
methodology, charts and the must-have-filtering demo are in the notebook.

## Demo video script (3-4 min)

1. **0:00-0:30** — repo tour: M1 tools, M2 files, dataset folders.
2. **0:30-1:00** — `python scripts\make_dataset.py`: show a generated PDF
   resume + `dataset/labels.json` ground truth.
3. **1:00-1:40** — `python resume_rag.py --rebuild --stats`, then a
   `--query` call: point out section-aware hits and similarities.
4. **1:40-2:40** — `python job_matcher.py job_descriptions\senior_ml_engineer.txt`:
   walk the JSON (scores, matched_skills, excerpts, reasoning), highlight
   `filtered_out` (Daniel Okafor fails "5+ years Python").
5. **2:40-3:30** — open `rag_analysis.ipynb`: metadata accuracy table,
   hybrid-vs-semantic bar chart, latency table, weight-sweep plot.
6. **3:30-4:00** — wrap: architecture recap + limitations.

---

# Next level — Hard corpus, smarter extraction, reranking, UI

Four upgrades on top of Milestone 2, with every claim below measured in the
re-executed `rag_analysis.ipynb` (§8-§10).

## Streamlit UI

```powershell
python -m streamlit run app.py
```

Pick a corpus (clean 36 / hard 40), load or paste a JD, and match: ranked
candidate cards with the 0-100 score, per-component breakdown bars
(semantic / BM25 / rerank / skill coverage / experience fit), section-labelled
evidence excerpts, reasoning, and a "Filtered out" panel showing exactly which
must-have each excluded candidate failed. Sidebar extras: retrieval-mode
toggle, semantic-weight slider, index rebuild (optionally Claude-assisted),
and drag-and-drop upload of a new resume or JD — uploads are parsed, indexed
and matchable immediately.

## Hard-mode corpus (`scripts/make_dataset.py --hard`)

40 additional labelled resumes in `resumes_hard/` across 3 difficulty tiers:
tier 0 clean; tier 1 nonstandard headers ("Career History", "Toolbox") and
month-level dates (`Jan 2020 - Mar 2023`, `03/2019 - present`); tier 2 no
skills section at all (skills live only in prose), missing education lines,
headerless lead-ins. `dataset/labels_hard.json` carries per-resume ground
truth incl. per-skill tenure, derived strictly from what the rendered text
shows — so accuracy numbers are fair.

## Extraction v2 + Claude fallback

Month-aware date parsing (month names, `MM/YYYY`, bare years, en-dashes,
"to"), per-skill tenure via job-block attribution with interval merging, and
an optional Anthropic pass (`RESUME_RAG_LLM=off|auto|always`, model override
via `RESUME_RAG_LLM_MODEL`) that fills only low-confidence gaps — results are
disk-cached in `.cache/llm_extract/` and every failure path falls back to the
deterministic extractor (offline-safe; mocked in tests).

## Cross-encoder reranking (opt-in)

`ms-marco-MiniLM-L-6-v2` rescores the top retrieved chunks as (JD, chunk)
pairs: `python job_matcher.py <jd> --rerank`, the UI's "Hybrid + rerank"
mode, or `JobMatcher(rerank=True)`. Off by default — see the measurements.

## Measured results (notebook §8-§10, this repo, CPU)

| Metric | Clean (36) | Hard (40) |
|---|---|---|
| P@5 soft — semantic / hybrid / hybrid+rerank | 0.967 / 0.967 / 0.967 | 0.933 / 0.933 / 0.933 |
| R@10 soft — semantic / hybrid / hybrid+rerank | 0.886 / 0.921 / 0.900 | 0.889 / 0.889 / 0.889 |
| MRR / hit@1 (all modes) | 1.000 / 1.000 | 1.000 / 1.000 |
| Extraction: name / years / education / skills recall (regex only) | 1.00 each | 1.00 each |
| Per-skill tenure MAE | — | 0.13 yrs |
| Rerank latency (median, on top of ~25 ms semantic) | — | ~1.1 s |

**Honest findings:** role-separated corpora nearly saturate MiniLM ranking —
the hard corpus only moves P@5 from 0.967 to 0.933, all three retrieval modes
tie there, and the cross-encoder's ~1.1 s/query buys no measurable ranking
gain on this data (its value is expected on genuinely noisy, overlapping
real-world resumes; it stays opt-in for exactly that reason). The month-aware
extractor recovers 100% of the hard-corpus metadata, so the Claude fallback
correctly never fires (`llm_assisted: 0` in auto mode) — its trigger logic
and gap-filling merge are unit-tested with mocked clients, and it exists for
truly unstructured resumes the deterministic path can't parse, e.g. arbitrary
uploads through the UI.

## Demo video addendum (+45 s)

7. **4:00-4:30** — `python -m streamlit run app.py`: match the ML JD on the hard corpus,
   point at breakdown bars + filtered-out reasons; drag in a new resume and
   re-match to show it ranked.
8. **4:30-4:45** — notebook §8 bar chart (clean vs hard) + the generated
   takeaways cell: metrics computed live, no hand-written claims.
