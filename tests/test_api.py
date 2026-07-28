from fastapi.testclient import TestClient

from finrag.api import create_app
from finrag.generation import Answer
from finrag.retrieval import Retriever


class FakeGenerator:
    def answer(self, question, chunks):
        return Answer(
            text="Revenue was $391.0 billion [1].",
            sources=[{"ref": 1, "source": "ACME 10-K 2025", "section": "Item 7",
                      "chunk_id": chunks[0].chunk.chunk_id}],
        )


def make_client(store):
    app = create_app(retriever=Retriever(store), generator=FakeGenerator())
    return TestClient(app)


def test_health(store):
    client = make_client(store)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "index_loaded": True}


def test_query_returns_answer_and_sources(store):
    client = make_client(store)
    resp = client.post("/query", json={"question": "What was total revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "391.0 billion" in body["answer"]
    assert body["sources"][0]["section"] == "Item 7"
    assert body["retrieval_ms"] >= 0


def test_query_validates_input(store):
    client = make_client(store)
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert client.post("/query", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_query_503_when_index_missing():
    app = create_app()  # no retriever injected, no index on disk
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "What was revenue?"})
        assert resp.status_code == 503
