"""Hybrid retrieval: BM25 + dense, merged with reciprocal rank fusion.

RRF over raw score mixing because BM25 and cosine scores live on different
scales; rank-based fusion sidesteps the calibration problem entirely. If no
dense index is present (e.g. in CI, where we don't call embedding APIs),
this degrades cleanly to BM25-only.
"""

from dataclasses import dataclass

from .chunking import Chunk
from .index import ChunkStore, Embedder


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


def rrf_merge(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Standard RRF: score(d) = sum over rankings of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    def __init__(self, store: ChunkStore, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        candidate_k = max(top_k * 4, 20)
        rankings = [self.store.search_bm25(query, k=candidate_k)]

        if self.embedder is not None and self.store.has_dense:
            qvec = self.embedder.embed([query])[0]
            rankings.append(self.store.search_dense(qvec, k=candidate_k))

        merged = rrf_merge(rankings)[:top_k]
        return [
            RetrievedChunk(chunk=self.store.chunks[idx], score=score, rank=i)
            for i, (idx, score) in enumerate(merged)
        ]
