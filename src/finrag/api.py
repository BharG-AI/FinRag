import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .generation import Generator
from .index import ChunkStore, OpenAIEmbedder
from .retrieval import Retriever

logger = logging.getLogger("finrag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_ms: int
    generation_ms: int


def create_app(retriever: Retriever | None = None, generator: Generator | None = None) -> FastAPI:
    """App factory. Tests inject a retriever/generator; production wiring
    happens in the lifespan handler from settings."""
    state = {"retriever": retriever, "generator": generator}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["retriever"] is None:
            settings = get_settings()
            if (settings.index_dir / "chunks.json").exists():
                store = ChunkStore.load(settings.index_dir)
                embedder = None
                if settings.llm_api_key and store.has_dense:
                    embedder = OpenAIEmbedder(
                        settings.embed_model, settings.llm_api_key, settings.llm_base_url
                    )
                state["retriever"] = Retriever(store, embedder)
                state["generator"] = Generator(
                    settings.llm_model, settings.llm_api_key, settings.llm_base_url
                )
                logger.info("Loaded index: %d chunks, dense=%s", len(store.chunks), store.has_dense)
            else:
                logger.warning(
                    "No index at %s — run finrag.ingest and finrag.build_index first",
                    settings.index_dir,
                )
        yield

    app = FastAPI(title="finrag", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "index_loaded": state["retriever"] is not None}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        if state["retriever"] is None or state["generator"] is None:
            raise HTTPException(status_code=503, detail="Index not loaded")

        t0 = time.monotonic()
        chunks = state["retriever"].retrieve(req.question, top_k=req.top_k)
        t1 = time.monotonic()
        answer = state["generator"].answer(req.question, chunks)
        t2 = time.monotonic()

        logger.info(
            "query top_k=%d retrieved=%d retrieval_ms=%d generation_ms=%d",
            req.top_k, len(chunks), int((t1 - t0) * 1000), int((t2 - t1) * 1000),
        )
        return QueryResponse(
            answer=answer.text,
            sources=answer.sources,
            retrieval_ms=int((t1 - t0) * 1000),
            generation_ms=int((t2 - t1) * 1000),
        )

    return app


app = create_app()
