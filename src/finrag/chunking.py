"""Split a filing into item-level sections, then into overlapping chunks.

Chunking on raw text loses the document structure that matters most in a
10-K: whether a passage came from Risk Factors (1A) or MD&A (7) changes how
an answer should be read. So we detect item headings first and carry the
section label through as metadata on every chunk.
"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Matches headings like "Item 1A. Risk Factors" or "ITEM 7 — MD&A" at the
# start of a line. Filings are inconsistent about punctuation and case.
_ITEM_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\u2014\u2013-]?\s*(.{0,80})$",
    re.IGNORECASE | re.MULTILINE,
)

SECTION_NAMES = {
    "1": "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Common Equity",
    "7": "Management's Discussion and Analysis",
    "7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9a": "Controls and Procedures",
}


@dataclass
class Chunk:
    text: str
    source: str          # e.g. "AAPL 10-K 2025"
    section: str         # e.g. "Item 1A"
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_label, section_text) pairs.

    Text before the first item heading is kept under "Front Matter" so
    nothing silently disappears.
    """
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [("Full Document", text)]

    sections: list[tuple[str, str]] = []
    front = text[: matches[0].start()].strip()
    if front:
        sections.append(("Front Matter", front))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        label = f"Item {m.group(1).upper()}"
        # A table of contents produces item headings with almost no body.
        # Keep only sections with enough text to be the real thing.
        if len(body) > 200:
            sections.append((label, body))
    return sections


def chunk_filing(
    text: str,
    source: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for i, piece in enumerate(splitter.split_text(body)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    section=section,
                    chunk_id=f"{source}::{section}::{i}",
                )
            )
    return chunks
