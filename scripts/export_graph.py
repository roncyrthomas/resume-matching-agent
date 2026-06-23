"""Render the matching agent's state machine to Mermaid (+ PNG if possible).

    python scripts/export_graph.py

Writes docs/diagrams/matching_agent_state_machine.md (always) and .png (when a
renderer is available). Uses a StubLLM and an unconfigured matcher — no graph
node runs at build time, so no index or API key is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_llm import StubLLM
from job_matcher import JobMatcher
from matching_agent import Engine, build_agent


def main() -> int:
    # No nodes execute during graph construction, so a bare matcher is fine.
    engine = Engine(matcher=JobMatcher.__new__(JobMatcher), llm=StubLLM([]))
    graph = build_agent(engine)

    out_dir = Path("docs/diagrams")
    out_dir.mkdir(parents=True, exist_ok=True)
    mermaid = graph.get_graph().draw_mermaid()
    (out_dir / "matching_agent_state_machine.md").write_text(
        f"# Matching Agent — State Machine\n\n```mermaid\n{mermaid}\n```\n",
        encoding="utf-8")
    print("wrote docs/diagrams/matching_agent_state_machine.md")

    try:
        png = graph.get_graph().draw_mermaid_png()
        (out_dir / "matching_agent_state_machine.png").write_bytes(png)
        print("wrote docs/diagrams/matching_agent_state_machine.png")
    except Exception as exc:  # noqa: BLE001 — PNG needs network/graphviz; mermaid suffices
        print(f"(PNG export skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
