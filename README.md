# finrag

Retrieval-augmented QA over SEC filings, built like a service rather than a
notebook: FastAPI API, hybrid retrieval (BM25 + dense embeddings fused with
RRF), section-aware chunking of 10-Ks, and a retrieval evaluation harness
that runs as a regression gate in CI.

```
EDGAR ──> ingest ──> section-aware chunking ──> BM25 index ──┐
                                          └──> FAISS index ──┼──> RRF ──> top-k ──> LLM ──> answer + citations
                                                    query ───┘
```

## Why these choices

**Section-aware chunking.** Naive fixed-size chunking throws away the one
piece of structure that matters most in a 10-K: whether a passage is from
Risk Factors (Item 1A) or MD&A (Item 7). The chunker detects item headings
first and carries the section label as metadata on every chunk, so answers
can cite "AAPL 10-K 2025 — Item 7" instead of an opaque offset.

**Hybrid retrieval with RRF.** Filings questions split into two kinds:
exact-term lookups ("effective tax rate", a specific product name) where
BM25 wins, and paraphrased questions ("how do they hedge currency risk?")
where embeddings win. BM25 and cosine scores aren't on comparable scales,
so results are merged by reciprocal rank fusion instead of score mixing —
no calibration to maintain. If no dense index exists, retrieval degrades
cleanly to BM25-only; that's also what CI uses, so the test suite never
makes network calls.

**Direct LLM client, no chain framework.** The generation step is one
prompt template and one request. A framework here would mostly hide the two
things I want visible in logs: the exact prompt and the exact context. (The
agentic follow-up project is where LangGraph earns its place.)

**Evals as a CI gate, not a screenshot.** `eval/run_retrieval_eval.py`
computes hit@k and MRR against labeled questions and exits non-zero below a
threshold, so a chunking or retrieval change that regresses quality fails
the build. CI runs it against a small synthetic fixture corpus
(deterministic, free); the same runner works against a real index with
`--index-dir` and your own labeled questions.

Current fixture-eval numbers (BM25-only path, the one CI exercises):

| metric | value |
|--------|-------|
| hit@5  | 1.00  |
| MRR    | 0.92  |

The fixture set is deliberately small and easy — it's a regression tripwire,
not a benchmark. Corpus-level numbers on real filings depend on the filings
you index and the questions you label.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in OPENAI_API_KEY and FINRAG_SEC_USER_AGENT

# 1. download filings (SEC requires a User-Agent identifying you)
python -m finrag.ingest AAPL MSFT

# 2. build the index (--dense adds embeddings; omit for BM25-only)
python -m finrag.build_index --dense

# 3. serve
uvicorn finrag.api:app --port 8000
```

Query it:

```bash
curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "What supply chain risks does Apple disclose?"}'
```

Response includes the answer with inline citations, the source
chunks (filing + item section), and retrieval/generation latency in ms.

Works fully local too: point `FINRAG_LLM_BASE_URL` at Ollama and use a
local embedding-capable endpoint, or skip `--dense` and run BM25-only with
a local generation model.

## Development

```bash
pytest -q                                              # 17 tests, no network
ruff check .
python eval/run_retrieval_eval.py --fixtures --min-hit-rate 0.8
```

Docker:

```bash
docker build -t finrag .
docker run -v $(pwd)/data:/app/data -p 8000:8000 --env-file .env finrag
```

## Roadmap

- Cross-encoder reranking stage after RRF, with before/after eval numbers
- Answer-quality evals (faithfulness, relevance) via RAGAS on a labeled set
- Streaming responses
