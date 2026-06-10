"""Shared helpers for RAG tests: a deterministic, network-free embedder."""

from __future__ import annotations

import re
import zlib
from typing import List, Sequence

DIM = 64


class FakeEmbedder:
    """Bag-of-words hash embedding. Deterministic across processes (crc32)."""

    name = "fake-embedder"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * DIM
            for token in re.findall(r"[a-z0-9+#./-]+", text.lower()):
                vec[zlib.crc32(token.encode("utf-8")) % DIM] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


ML_RESUME = """Riley Carter
Senior Machine Learning Engineer
Remote (US) | riley.carter@example.com | +1-555-0001

SUMMARY
Senior Machine Learning Engineer with 8+ years of experience in Python and PyTorch.

SKILLS
Python, PyTorch, Machine Learning, pandas, SQL, Docker

EXPERIENCE
Senior ML Engineer, Vision Corp (2021 - Present)
- Deployed PyTorch models serving 3M inferences per day.
Data Scientist, Insight AI (2018 - 2021)
- Built pandas and SQL feature pipelines for forecasting.

EDUCATION
M.S. in Computer Science, State University (2017)
"""

JUNIOR_RESUME = """Jordan Blake
Software Developer
Austin, TX | jordan.blake@example.com | +1-555-0002

SUMMARY
Software developer with 2+ years of experience building web apps in Python.

SKILLS
Python, Django, PostgreSQL

EXPERIENCE
Developer, WebStart (2024 - Present)
- Built Django REST endpoints for the customer portal.

EDUCATION
B.S. in Computer Science, Northfield College (2023)
"""

FRONTEND_RESUME = """Casey Morgan
Frontend Engineer
Berlin, Germany | casey.morgan@example.com | +1-555-0003

SUMMARY
Frontend engineer with 7+ years of experience in React and TypeScript.

SKILLS
React, TypeScript, JavaScript, CSS, HTML

EXPERIENCE
Frontend Engineer, Pixel Labs (2019 - Present)
- Built a React design system used by 4 product teams.

EDUCATION
B.Sc. in Software Engineering, Westlake University (2018)
"""


def make_corpus(tmp_path, monkeypatch):
    """Write 3 small resumes, sandbox fs_tools there, return a ResumeRAG.

    Uses a per-call unique collection name so multiple make_corpus() calls in
    the same pytest process (which shares a single EphemeralClient singleton)
    do not write into the default 'resumes' collection and pollute tests that
    assert on an empty index.
    """
    import chromadb

    from resume_rag import ResumeRAG

    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "riley_carter.txt").write_text(ML_RESUME, encoding="utf-8")
    (resumes / "jordan_blake.txt").write_text(JUNIOR_RESUME, encoding="utf-8")
    (resumes / "casey_morgan.txt").write_text(FRONTEND_RESUME, encoding="utf-8")
    monkeypatch.setenv("FS_TOOLS_BASE_DIR", str(tmp_path))
    # Unique collection per corpus so the shared EphemeralClient singleton does
    # not contaminate the default "resumes" collection used by empty-index tests.
    collection_name = f"resumes_{abs(hash(str(tmp_path)))}"
    return ResumeRAG(
        resumes_dir="resumes",
        embedder=FakeEmbedder(),
        client=chromadb.EphemeralClient(),
        collection_name=collection_name,
    )
