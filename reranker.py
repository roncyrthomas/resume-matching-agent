"""Cross-encoder reranking: scores (JD, chunk) PAIRS jointly — slower but far
more precise than bi-encoder cosine. Lazy-loaded; never imported at module
import time by job_matcher, so the default pipeline stays offline."""

from __future__ import annotations

import math
from typing import List, Sequence

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    name = f"cross-encoder:{RERANK_MODEL}"

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Relevance of each text to the query, sigmoid-squashed to [0, 1]."""
        if not texts:
            return []
        logits = self._model.predict([(query, t) for t in texts])
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]
