"""Answer generation over retrieved chunks.

Direct OpenAI-compatible client rather than a chain abstraction: the prompt
is one template and the call is one request, so a framework here would only
hide the two things I most want visible in logs (the exact prompt and the
exact context).
"""

from dataclasses import dataclass

from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about SEC filings using only the numbered context \
passages provided. Cite passages inline like [1] or [2][3]. If the context does not \
contain the answer, say so plainly instead of guessing. Financial figures must come \
verbatim from the context."""


@dataclass
class Answer:
    text: str
    sources: list[dict]


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(chunks, start=1):
        header = f"[{i}] {rc.chunk.source} — {rc.chunk.section}"
        parts.append(f"{header}\n{rc.chunk.text}")
    return "\n\n".join(parts)


class Generator:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        context = build_context(chunks)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
            ],
        )
        text = resp.choices[0].message.content or ""
        sources = [
            {
                "ref": i + 1,
                "source": rc.chunk.source,
                "section": rc.chunk.section,
                "chunk_id": rc.chunk.chunk_id,
            }
            for i, rc in enumerate(chunks)
        ]
        return Answer(text=text, sources=sources)
