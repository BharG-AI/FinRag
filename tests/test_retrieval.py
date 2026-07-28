from conftest import FakeEmbedder
from finrag.index import ChunkStore, tokenize
from finrag.retrieval import Retriever, rrf_merge


def test_tokenize_lowercases_and_strips():
    assert tokenize("Net Revenue was $391.0 billion!") == [
        "net", "revenue", "was", "391", "0", "billion",
    ]


def test_bm25_finds_obvious_match(store: ChunkStore):
    results = store.search_bm25("foreign exchange hedging derivative", k=3)
    top_chunk = store.chunks[results[0][0]]
    assert top_chunk.section == "Item 7A"


def test_bm25_no_match_returns_empty(store: ChunkStore):
    assert store.search_bm25("zzzz qqqq xxxx", k=5) == []


def test_rrf_prefers_agreement():
    # Doc 3 is mid-ranked in both lists; doc 1 and doc 5 each appear once at top.
    a = [(1, 10.0), (3, 8.0), (2, 5.0)]
    b = [(5, 0.9), (3, 0.8), (4, 0.5)]
    merged = rrf_merge([a, b])
    assert merged[0][0] == 3


def test_hybrid_retrieval_returns_ranked_chunks(store: ChunkStore):
    embedder = FakeEmbedder()
    store.build_dense(embedder)
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("research and development expense", top_k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [0, 1, 2]
    assert results[0].chunk.chunk_id == "ACME 10-K 2025::Item 7::1"


def test_retrieval_degrades_to_bm25_without_embedder(store: ChunkStore):
    retriever = Retriever(store, embedder=None)
    results = retriever.retrieve("legal proceedings antitrust", top_k=2)
    assert results[0].chunk.section == "Item 3"


def test_store_save_load_roundtrip(store: ChunkStore, tmp_path):
    store.build_dense(FakeEmbedder())
    store.save(tmp_path / "idx")
    loaded = ChunkStore.load(tmp_path / "idx")
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.has_dense
    results = loaded.search_bm25("foreign exchange", k=1)
    assert loaded.chunks[results[0][0]].section == "Item 7A"
