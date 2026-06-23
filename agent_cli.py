"""agent_cli.py — streaming conversational CLI for the matching agent.

Usage (after `python resume_rag.py --rebuild`):

    python agent_cli.py job_descriptions/senior_ml_engineer.txt
    python agent_cli.py "Senior React engineer, 5+ years" -k 5 --anonymize

Then chat: refine ("weight experience higher"), compare ("compare the top 3"),
interview ("interview questions for <name>"), screen ("deep-screen the top 10"),
or done.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import fs_tools
from agent_llm import AnthropicLLM
from job_matcher import JobMatcher
from matching_agent import MatchingAgent, anonymize_jd_or_resume


def render_state(state: dict) -> str:
    """Format the latest report plus a shortlist line and a 'what next?' prompt."""
    lines = [state.get("report", "(no report)")]
    if state.get("shortlist"):
        lines.append("")
        lines.append("Candidates: " + ", ".join(
            f"{r['name']} ({r['score']})" for r in state["shortlist"]))
    lines.append("\nWhat next? (refine / compare / interview <name> / screen / done)")
    return "\n".join(lines)


def _load_jd(source: str) -> str:
    """Read *source* as a file if it exists, else treat it as inline JD text."""
    res = fs_tools.read_file(source)
    return str(res["content"]) if res.get("success") else source


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Conversational resume-matching agent.")
    parser.add_argument("jd", help="path to a JD file or inline JD text")
    parser.add_argument("-k", type=int, default=10, help="number of matches (default 10)")
    parser.add_argument("--anonymize", action="store_true",
                        help="redact the name/contact preamble before matching")
    args = parser.parse_args(argv)

    jd_text = _load_jd(args.jd)
    if args.anonymize:
        jd_text = anonymize_jd_or_resume(jd_text)

    agent = MatchingAgent(JobMatcher(), AnthropicLLM())
    state = agent.start(jd_text, k=args.k)
    print(render_state(state))

    while "__interrupt__" in state:
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        state = agent.send(message)
        print(render_state(state))
        if state.get("last_intent") == "done":
            break
    print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
