from finrag.chunking import chunk_filing, split_sections

FILING = (
    "ACME CORP ANNUAL REPORT\n\n"
    "Item 1. Business\n" + ("ACME designs and sells widgets worldwide. " * 20) + "\n\n"
    "Item 1A. Risk Factors\n" + ("Demand for widgets may decline for many reasons. " * 20) + "\n\n"
    "Item 7. Management's Discussion and Analysis\n"
    + ("Revenue grew due to higher widget volume. " * 20)
)


def test_sections_detected():
    labels = [label for label, _ in split_sections(FILING)]
    assert "Item 1" in labels
    assert "Item 1A" in labels
    assert "Item 7" in labels


def test_front_matter_kept():
    labels = [label for label, _ in split_sections(FILING)]
    assert labels[0] == "Front Matter"


def test_toc_stub_sections_dropped():
    text = (
        "Item 1. Business ... 3\nItem 1A. Risk Factors ... 12\n\n"
        "Item 1. Business\n" + ("Real section body with substance. " * 20)
    )
    labels = [label for label, _ in split_sections(text)]
    # The two ToC lines are tiny and get dropped; the real Item 1 survives.
    assert labels.count("Item 1") == 1


def test_chunks_carry_metadata():
    chunks = chunk_filing(FILING, source="ACME 10-K 2025", chunk_size=300, chunk_overlap=50)
    assert all(c.source == "ACME 10-K 2025" for c in chunks)
    sections = {c.section for c in chunks}
    assert "Item 1A" in sections


def test_chunk_size_respected():
    chunks = chunk_filing(FILING, source="ACME 10-K 2025", chunk_size=300, chunk_overlap=50)
    assert all(len(c.text) <= 350 for c in chunks)
    assert len(chunks) > 3


def test_no_item_headings_falls_back_to_full_document():
    sections = split_sections("Just some text with no headings at all.")
    assert sections == [("Full Document", "Just some text with no headings at all.")]
