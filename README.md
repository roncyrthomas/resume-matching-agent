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
├── fs_tools.py             # Part A: the four file-system tools
├── llm_file_assistant.py   # Part B: Anthropic tool-use loop + CLI
├── scripts/make_samples.py # generates the dummy resumes
├── resumes/                # 8 sample resumes (.txt/.pdf/.docx) [generated]
├── tests/test_fs_tools.py  # pytest suite
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
