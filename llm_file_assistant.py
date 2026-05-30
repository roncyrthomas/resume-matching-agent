"""llm_file_assistant.py — Anthropic Claude tool-use over the fs_tools module.

The assistant lets Claude answer natural-language questions about resume files
by calling the four tools in :mod:`fs_tools`. Example:

    python llm_file_assistant.py "Find resumes mentioning Python experience"
    python llm_file_assistant.py "Read all resumes in the resumes folder"
    python llm_file_assistant.py "Create a summary file for resumes/john_doe.pdf"

Set ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_MODEL`` /
``FS_TOOLS_BASE_DIR``) in a ``.env`` file or the environment.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import fs_tools

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 10
MAX_TOKENS = 4096

SYSTEM_PROMPT = (
    "You are a file assistant for a folder of resume files. Use the provided "
    "tools to read, list, search and write files on the user's behalf. Resume "
    "files normally live in the 'resumes' directory. When asked to summarise a "
    "resume, read it first, then write the summary with write_file. Always base "
    "answers on actual tool results, and report any errors the tools return."
)

# Map tool names to their implementations.
DISPATCH: Dict[str, Callable[..., Any]] = {
    "read_file": fs_tools.read_file,
    "list_files": fs_tools.list_files,
    "write_file": fs_tools.write_file,
    "search_in_file": fs_tools.search_in_file,
}

# JSON-schema tool definitions sent to the Anthropic API.
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a resume file (.txt, .pdf or .docx) and return its extracted "
            "text content plus metadata (size, modified date, page/paragraph "
            "count)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file, e.g. 'resumes/john_doe.pdf'.",
                }
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files in a directory with their name, size and modified date. "
            "Optionally filter by extension."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list, e.g. 'resumes'.",
                },
                "extension": {
                    "type": "string",
                    "description": "Optional extension filter, e.g. '.pdf' or 'txt'.",
                },
            },
            "required": ["directory"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write text content to a file, creating parent directories if "
            "needed. Use this to save summaries or reports."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Destination path, e.g. 'summaries/john_doe.txt'.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write.",
                },
            },
            "required": ["filepath", "content"],
        },
    },
    {
        "name": "search_in_file",
        "description": (
            "Case-insensitively search a file for a keyword and return each "
            "match with surrounding text context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "File to search.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Keyword or phrase to search for.",
                },
            },
            "required": ["filepath", "keyword"],
        },
    },
]


def _execute_tool(name: str, tool_input: Dict[str, Any]) -> Any:
    """Run a single tool by name, returning its structured result."""
    fn = DISPATCH.get(name)
    if fn is None:
        return {"success": False, "error": f"unknown tool: {name}"}
    try:
        return fn(**tool_input)
    except TypeError as exc:
        # Bad / missing arguments from the model.
        return {"success": False, "error": f"invalid arguments for {name}: {exc}"}


def _is_error_result(result: Any) -> bool:
    """A dict result with ``success is False`` signals a tool failure."""
    return isinstance(result, dict) and result.get("success") is False


def _tool_result_block(tool_use_id: str, result: Any) -> Dict[str, Any]:
    """Build a tool_result content block, flagging failures with is_error."""
    block: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(result, default=str),
    }
    if _is_error_result(result):
        block["is_error"] = True
    return block


def run_agent(
    query: str,
    *,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = True,
) -> str:
    """Drive a tool-use conversation with Claude until it produces text.

    Args:
        query: The user's natural-language request.
        client: An Anthropic client (injected in tests). Created if omitted.
        model: Model id; defaults to ``$ANTHROPIC_MODEL`` or ``DEFAULT_MODEL``.
        max_iterations: Safety cap on tool-call rounds.
        verbose: Print each tool call/result for the demo.

    Returns:
        Claude's final text answer, or a notice if the iteration cap was hit.
    """
    if client is None:
        client = _build_client()
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

    messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return _collect_text(response) or "(no textual response)"

        # Echo the assistant's full turn (text + tool_use blocks) back verbatim.
        messages.append({"role": "assistant", "content": response.content})

        tool_results: List[Dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_input = dict(block.input or {})
            if verbose:
                print(f"  -> {block.name}({json.dumps(tool_input)})")
            result = _execute_tool(block.name, tool_input)
            if verbose:
                status = "ERROR" if _is_error_result(result) else "ok"
                print(f"     [{status}]")
            tool_results.append(_tool_result_block(block.id, result))

        messages.append({"role": "user", "content": tool_results})

    return f"Stopped after {max_iterations} tool-call rounds without a final answer."


def _collect_text(response: Any) -> str:
    """Concatenate the text blocks of a response."""
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()


def _build_client() -> Any:
    """Create an Anthropic client, validating configuration up front."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is not set. Add it to a .env file or your "
            "environment before running the assistant."
        )

    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("The 'anthropic' package is not installed. Run: pip install -r requirements.txt")

    return Anthropic()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print('Usage: python llm_file_assistant.py "your request"')
        return 2

    query = " ".join(argv)
    print(f"User: {query}\n")
    answer = run_agent(query)
    print(f"\nAssistant: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
