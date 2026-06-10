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
