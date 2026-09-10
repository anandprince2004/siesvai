import re
from typing import List, Dict

def is_headed_document(text: str, min_headings: int = 2) -> bool:
    """
    Heuristic check: does this document have enough heading-like lines
    to be worth splitting on structure, rather than blind char-chunking?

    Deliberately biased toward over-detecting — a false-positive heading
    just creates one extra (harmless) split point. A false negative means
    falling back to plain chunking, which is the exact problem we're
    trying to reduce, so the bar to clear is kept low on purpose.
    """
    lines = text.split("\n")
    heading_count = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and _looks_like_heading(stripped, lines, i):
            heading_count += 1
            if heading_count >= min_headings:
                return True
    return False

def _looks_like_heading(stripped: str, lines: List[str], i: int) -> bool:
    """
    A line counts as heading-like if it's short, doesn't end like a
    sentence, isn't a bare page marker/number, is preceded by a blank
    line (or is the very first line), AND is immediately followed by
    actual content (not another blank line).

    The "preceded by blank, followed by content" shape is what actually
    distinguishes a heading from an ordinary short line inside a
    paragraph — checking only "followed by blank" (an earlier version
    of this function) wrongly matched the LAST line of a paragraph
    instead, since paragraphs also end right before a blank line.
    """
    if len(stripped) > 80:
        return False
    if re.match(r'^(-{2,}\s*Page\s*\d+\s*-{2,}|\d+)$', stripped, re.IGNORECASE):
        return False
    if stripped[-1] in ".,;:?!":
        return False

    prev_line = lines[i - 1].strip() if i > 0 else ""
    if prev_line != "":
        return False

    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
    if next_line == "":
        return False

    return True

def chunk_by_headings(
    text: str,
    source_name: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Dict[str, str]]:
    """
    Split text into sections at detected heading lines. Within each
    section, apply the existing char-count chunking only if the section
    itself is still too long.

    The key fix: every chunk is PREFIXED with its nearest heading, e.g.

        "Section: Visiting Hours on Working Days\n\nPrincipal's Office: ..."

    That heading text becomes part of what gets embedded, pulling the
    chunk's embedding toward the actual topic — this is what makes a
    query like "office timing" retrieve it, instead of the chunk reading
    as an undifferentiated blob of prospectus text.
    """
    lines = text.split("\n")
    sections: List[tuple] = []
    current_heading = "General"
    current_body: List[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and _looks_like_heading(stripped, lines, i):
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = stripped
            current_body = []
        elif stripped:
            current_body.append(stripped)

    if current_body:
        sections.append((current_heading, current_body))

    chunks: List[Dict[str, str]] = []
    for heading, body_lines in sections:
        section_text = " ".join(body_lines).strip()
        if not section_text:
            continue

        sub_chunks = (
            [section_text]
            if len(section_text) <= max_chars
            else _char_chunk(section_text, max_chars, overlap)
        )

        for sub in sub_chunks:
            chunks.append({
                "text": f"Section: {heading}\n\n{sub}",
                "heading": heading,
                "source": source_name,
            })

    return chunks

def _char_chunk(text: str, max_chars: int, overlap: int) -> List[str]:
    """Same sliding-window logic as the existing plain-chunking fallback."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

if __name__ == "__main__":
    sample = """Visiting Hours on Working Days
    Principal's Office: 10.30 am - 1.00 pm
    College Office: 10.00 am - 1.00 p.m

    Fee Structure
    BMS: Rs. 45,000 per year
    BSc IT: Rs. 40,000 per year

    Library Timings
    Monday to Friday: 8 AM to 6 PM
    Saturday: 8 AM to 4:30 PM
    """
    print("is_headed_document:", is_headed_document(sample))
    for c in chunk_by_headings(sample, "sample.txt"):
        print("-" * 40)
        print(c["text"])
