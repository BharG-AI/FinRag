import json
from pathlib import Path

import numpy as np
import pytest

from finrag.chunking import Chunk
from finrag.index import ChunkStore

# Small hand-written corpus imitating 10-K language, shared with the CI eval.
# Enough signal for deterministic retrieval tests without shipping real
# filings in the repo.
CORPUS_PATH = Path(__file__).parent.parent / "eval" / "fixture_corpus.jsonl"


def load_fixture_chunks() -> list[Chunk]:
    rows = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    return [Chunk(**r) for r in rows]


@pytest.fixture
def store() -> ChunkStore:
    s = ChunkStore(load_fixture_chunks())
    s.build_bm25()
    return s


class FakeEmbedder:
    """Deterministic embeddings from token hashes — no network, no model.

    Not semantically meaningful, but overlapping tokens produce similar
    vectors, which is all the dense-path tests need.
    """

    dim = 64

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                vecs[i, hash(tok) % self.dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
