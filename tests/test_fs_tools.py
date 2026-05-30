"""Tests for fs_tools and the llm_file_assistant tool-use loop.

Run from the project root:

    pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fs_tools  # noqa: E402
import llm_file_assistant as assistant  # noqa: E402


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Confine fs_tools to a temp directory for every test."""
    monkeypatch.setenv("FS_TOOLS_BASE_DIR", str(tmp_path))
    return tmp_path


# --- write_file --------------------------------------------------------------


def test_write_file_creates_dirs_and_reports_bytes(sandbox):
    result = fs_tools.write_file("out/note.txt", "hello")
    assert result["success"] is True
    assert result["bytes_written"] == 5
    assert (sandbox / "out" / "note.txt").read_text(encoding="utf-8") == "hello"


def test_write_file_rejects_path_traversal(sandbox):
    result = fs_tools.write_file("../escape.txt", "nope")
    assert result["success"] is False
    assert "escapes" in result["error"]


# --- read_file ---------------------------------------------------------------


def test_read_file_txt_roundtrip(sandbox):
    fs_tools.write_file("resume.txt", "Python developer\nLoves testing")
    result = fs_tools.read_file("resume.txt")
    assert result["success"] is True
    assert "Python developer" in result["content"]
    assert result["metadata"]["extension"] == ".txt"
    assert result["metadata"]["unit"] == "lines"


def test_read_file_missing(sandbox):
    result = fs_tools.read_file("nope.txt")
    assert result["success"] is False
    assert "does not exist" in result["error"]


def test_read_file_unsupported_extension(sandbox):
    fs_tools.write_file("data.csv", "a,b,c")
    result = fs_tools.read_file("data.csv")
    assert result["success"] is False
    assert "unsupported file type" in result["error"]


# --- list_files --------------------------------------------------------------


def test_list_files_filters_by_extension(sandbox):
    fs_tools.write_file("docs/a.txt", "x")
    fs_tools.write_file("docs/b.txt", "y")
    fs_tools.write_file("docs/c.md", "z")

    all_files = fs_tools.list_files("docs")
    assert {f["name"] for f in all_files} == {"a.txt", "b.txt", "c.md"}

    txt_only = fs_tools.list_files("docs", extension="txt")  # no leading dot
    assert {f["name"] for f in txt_only} == {"a.txt", "b.txt"}
    assert all(f["extension"] == ".txt" for f in txt_only)


def test_list_files_missing_directory_returns_empty(sandbox):
    assert fs_tools.list_files("ghost") == []


# --- search_in_file ----------------------------------------------------------


def test_search_is_case_insensitive_with_context_and_lines(sandbox):
    fs_tools.write_file(
        "r.txt", "Skilled in PYTHON and python.\nAlso enjoys Rust."
    )
    result = fs_tools.search_in_file("r.txt", "python")
    assert result["success"] is True
    assert result["match_count"] == 2
    first = result["matches"][0]
    assert "char_start" in first and "char_end" in first
    assert "python" in first["snippet"].lower()
    assert first["line_number"] == 1


def test_search_empty_keyword_errors(sandbox):
    fs_tools.write_file("r.txt", "content")
    result = fs_tools.search_in_file("r.txt", "")
    assert result["success"] is False
    assert "keyword" in result["error"]


def test_search_no_matches(sandbox):
    fs_tools.write_file("r.txt", "nothing here")
    result = fs_tools.search_in_file("r.txt", "zzz")
    assert result["success"] is True
    assert result["match_count"] == 0
    assert result["matches"] == []


# --- llm_file_assistant loop -------------------------------------------------


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id: str, name: str, tool_input: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


class FakeMessages:
    def __init__(self, responses: List[SimpleNamespace]):
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: List[SimpleNamespace]):
        self.messages = FakeMessages(responses)


def test_run_agent_executes_tool_then_returns_text(sandbox):
    fs_tools.write_file("resumes/a.txt", "Python engineer")

    responses = [
        SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _tool_use_block("t1", "list_files", {"directory": "resumes"})
            ],
        ),
        SimpleNamespace(
            stop_reason="end_turn",
            content=[_text_block("There is 1 resume: a.txt.")],
        ),
    ]
    client = FakeClient(responses)

    answer = assistant.run_agent("List resumes", client=client, verbose=False)

    assert answer == "There is 1 resume: a.txt."
    # The tool_result that we sent back must carry the executed result.
    second_call_messages = client.messages.calls[1]["messages"]
    tool_result = second_call_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "t1"
    assert "a.txt" in tool_result["content"]


def test_run_agent_flags_tool_errors_with_is_error(sandbox):
    responses = [
        SimpleNamespace(
            stop_reason="tool_use",
            content=[_tool_use_block("t1", "read_file", {"filepath": "missing.txt"})],
        ),
        SimpleNamespace(
            stop_reason="end_turn",
            content=[_text_block("That file does not exist.")],
        ),
    ]
    client = FakeClient(responses)

    assistant.run_agent("Read missing.txt", client=client, verbose=False)

    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    payload = json.loads(tool_result["content"])
    assert payload["success"] is False


def test_run_agent_respects_max_iterations(sandbox):
    # A client that always asks for another tool call.
    looping = [
        SimpleNamespace(
            stop_reason="tool_use",
            content=[_tool_use_block(f"t{i}", "list_files", {"directory": "."})],
        )
        for i in range(10)
    ]
    client = FakeClient(looping)

    answer = assistant.run_agent(
        "loop forever", client=client, max_iterations=3, verbose=False
    )
    assert "Stopped after 3" in answer
    assert len(client.messages.calls) == 3


def test_run_agent_handles_unknown_tool(sandbox):
    result = assistant._execute_tool("does_not_exist", {})
    assert result["success"] is False
    assert "unknown tool" in result["error"]
