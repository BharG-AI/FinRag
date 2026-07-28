"""Retrieval evaluation: hit rate and MRR against labeled questions.

Two modes:
  --fixtures         run against the small synthetic corpus (deterministic,
                     no API calls — this is what CI runs)
  --index-dir PATH   run against a real built index, using questions you've
                     labeled for your own corpus

Exit code is non-zero if hit@k falls below --min-hit-rate, so CI fails when
a retrieval change regresses quality.
"""

import argparse
import json
import sys
from pathlib import Path

from finrag.chunking import Chunk
from finrag.index import ChunkStore
from finrag.retrieval import Retriever

ROOT = Path(__file__).parent


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_fixture_store() -> ChunkStore:
    rows = load_jsonl(ROOT / "fixture_corpus.jsonl")
    store = ChunkStore([Chunk(**r) for r in rows])
    store.build_bm25()
    return store


def evaluate(retriever: Retriever, questions: list[dict], k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    misses = []
    for q in questions:
        results = retriever.retrieve(q["question"], top_k=k)
        got_ids = [r.chunk.chunk_id for r in results]
        expected = set(q["expected_chunk_ids"])
        rank = next((i for i, cid in enumerate(got_ids) if cid in expected), None)
        if rank is None:
            misses.append(q["question"])
        else:
            hits += 1
            rr_sum += 1.0 / (rank + 1)
    n = len(questions)
    return {
        "n": n,
        "hit_rate": hits / n,
        "mrr": rr_sum / n,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    args = parser.parse_args()

    if args.fixtures:
        store = load_fixture_store()
        questions = load_jsonl(args.questions or ROOT / "questions_fixture.jsonl")
    elif args.index_dir:
        store = ChunkStore.load(args.index_dir)
        if not args.questions:
            raise SystemExit("--index-dir requires --questions with labels for your corpus")
        questions = load_jsonl(args.questions)
    else:
        raise SystemExit("Pass --fixtures or --index-dir")

    report = evaluate(Retriever(store), questions, k=args.k)

    print(f"questions:  {report['n']}")
    print(f"hit@{args.k}:     {report['hit_rate']:.2f}")
    print(f"MRR:        {report['mrr']:.2f}")
    for m in report["misses"]:
        print(f"  miss: {m}")

    if report["hit_rate"] < args.min_hit_rate:
        print(f"FAIL: hit rate {report['hit_rate']:.2f} < required {args.min_hit_rate:.2f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
