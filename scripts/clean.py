"""
SIESVAI - Step 2: Data Cleaning
Removes the repeated site navigation and footer boilerplate from each
scraped raw file, keeping only the actual page content. Saves cleaned
output to data/processed/.

Usage:
    python scripts/clean.py

Notes:
- The nav block and footer block are identified by fixed marker lines
  observed in the scraped output (see NAV_END_MARKER / FOOTER_START_MARKER
  below). If the site's structure changes, these markers may need updating.
- This script is intentionally simple and rule-based rather than "smart" —
  for a small, fixed set of pages, explicit markers are more reliable than
  guessing at HTML structure.
"""

import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

NAV_END_MARKER = "Eligibility Criteria for Admission to FY BMS"

FOOTER_START_MARKER = "Important Links"

LINES_TO_DROP = {"-", "TOP", "|"}

def strip_nav_and_footer(lines: list[str]) -> list[str]:
    """
    Remove everything up to and including NAV_END_MARKER, and everything
    from FOOTER_START_MARKER onward. Returns the remaining "body" lines.
    """
    start_index = 0
    end_index = len(lines)

    for i, line in enumerate(lines):
        if NAV_END_MARKER in line:
            start_index = i + 1
            break

    for i, line in enumerate(lines):
        if FOOTER_START_MARKER in line:
            end_index = i
            break

    if start_index >= end_index:
        print("  WARNING: nav/footer markers not found as expected. "
              "Returning full text for manual review.")
        return lines

    return lines[start_index:end_index]

def remove_noise_lines(lines: list[str]) -> list[str]:
    """Drop stray filler lines like lone hyphens or 'TOP'."""
    return [line for line in lines if line.strip() not in LINES_TO_DROP]

def collapse_duplicate_blank_runs(lines: list[str]) -> list[str]:
    """Remove consecutive duplicate lines (e.g. repeated 'Examinations' twice in a row)."""
    result = []
    for line in lines:
        if not result or result[-1] != line:
            result.append(line)
    return result

def clean_file(filename: str) -> None:
    raw_path = os.path.join(RAW_DIR, filename)
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    body_lines = strip_nav_and_footer(lines)
    body_lines = remove_noise_lines(body_lines)
    body_lines = collapse_duplicate_blank_runs(body_lines)

    cleaned_text = "\n".join(body_lines).strip()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, filename)
    with open(processed_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"[{filename}] {len(lines)} raw lines -> {len(body_lines)} cleaned lines "
          f"-> Saved to {processed_path}")

def clean_all_files() -> None:
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: {RAW_DIR} does not exist. Run scrape.py first.")
        return

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]

    if not raw_files:
        print(f"No .txt files found in {RAW_DIR}. Run scrape.py first.")
        return

    print(f"Cleaning {len(raw_files)} file(s)...\n")
    for filename in raw_files:
        clean_file(filename)

    print("\nDone. Review data/processed/ files manually before moving to chunking.")

if __name__ == "__main__":
    clean_all_files()
