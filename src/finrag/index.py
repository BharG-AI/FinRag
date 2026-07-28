"""Lexical (BM25) and dense (FAISS) indexes over a shared chunk list.

The store persists chunks as JSON and the dense vectors as a FAISS flat
index. BM25 is cheap to rebuild, so it's reconstructed on load instead of
being serialized.
"""

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        # The embeddings endpoint caps batch size; 256 stays well under it.
        for i in range(0, len(texts), 256):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + 256])
            out.extend(d.embedding for d in resp.data)
        vecs = np.asarray(out, dtype=np.float32)
        return _normalize(vecs)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class ChunkStore:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: list[Chunk] = chunks or []
        self._bm25: BM25Okapi | None = None
        self._faiss = None

    def build_bm25(self) -> None:
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def build_dense(self, embedder: Embedder) -> None:
        import faiss

        vecs = embedder.embed([c.text for c in self.chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self._faiss = index

    @property
    def has_dense(self) -> bool:
        return self._faiss is not None

    def search_bm25(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        if self._bm25 is None:
            self.build_bm25()
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search_dense(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        if self._faiss is None:
            return []
        q = _normalize(query_vec.reshape(1, -1).astype(np.float32))
        scores, ids = self._faiss.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, path: Path) -> None:
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        rows = [
            {"text": c.text, "source": c.source, "section": c.section, "chunk_id": c.chunk_id}
            for c in self.chunks
        ]
        (path / "chunks.json").write_text(json.dumps(rows))
        if self._faiss is not None:
            faiss.write_index(self._faiss, str(path / "dense.faiss"))

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        rows = json.loads((path / "chunks.json").read_text())
        store = cls([Chunk(**r) for r in rows])
        store.build_bm25()
        dense_path = path / "dense.faiss"
        if dense_path.exists():
            import faiss

            store._faiss = faiss.read_index(str(dense_path))
        return store
