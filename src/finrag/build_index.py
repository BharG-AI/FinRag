"""Build the search index from downloaded filings.

Usage:
    python -m finrag.build_index            # BM25 only, no API calls
    python -m finrag.build_index --dense    # also embed chunks (needs OPENAI_API_KEY)
"""

import argparse
import re

from .chunking import chunk_filing
from .config import get_settings
from .index import ChunkStore, OpenAIEmbedder


def source_label(filename: str) -> str:
    # AAPL_10-K_2025.txt -> "AAPL 10-K 2025"
    m = re.match(r"([A-Z.]+)_([\w-]+)_(\d{4})", filename)
    return " ".join(m.groups()) if m else filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="store_true", help="build the FAISS index too")
    args = parser.parse_args()

    settings = get_settings()
    files = sorted(settings.raw_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No filings in {settings.raw_dir} — run finrag.ingest first")

    chunks = []
    for f in files:
        filing_chunks = chunk_filing(
            f.read_text(),
            source=source_label(f.name),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        print(f"{f.name}: {len(filing_chunks)} chunks")
        chunks.extend(filing_chunks)

    store = ChunkStore(chunks)
    store.build_bm25()
    if args.dense:
        if not settings.llm_api_key:
            raise SystemExit("--dense requires OPENAI_API_KEY")
        print(f"Embedding {len(chunks)} chunks with {settings.embed_model}...")
        store.build_dense(
            OpenAIEmbedder(settings.embed_model, settings.llm_api_key, settings.llm_base_url)
        )

    store.save(settings.index_dir)
    print(f"Saved {len(chunks)} chunks to {settings.index_dir} (dense={store.has_dense})")


if __name__ == "__main__":
    main()
