import os
import re

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FILENAME = "prospectus.txt"

def is_doubled_letter_artifact(line: str) -> bool:
    """
    Detect the "SSIIEESS ((NNEERRUULL))" style artifact: every character
    doubled. Checks by collapsing consecutive duplicate characters and
    comparing to the original - a genuine doubled-letter artifact shrinks
    to roughly half its length; normal text with occasional doubled
    letters (e.g. "Committee", "Re-accredited") does not.
    """
    stripped = line.strip()
    if len(stripped) < 10:
        return False

    collapsed = re.sub(r"(.)\1", r"\1", stripped)
    return len(collapsed) < len(stripped) * 0.6

def clean_prospectus_text(text: str) -> str:
    """Remove doubled-letter artifact lines and collapse extra blank lines."""
    lines = text.split("\n")

    kept_lines = [line for line in lines if not is_doubled_letter_artifact(line)]

    cleaned_lines = []
    blank_run = 0
    for line in kept_lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def clean_and_copy_prospectus() -> None:
    raw_path = os.path.join(RAW_DIR, FILENAME)

    if not os.path.isfile(raw_path):
        print(f"ERROR: {raw_path} not found. Run extract_pdf.py first.")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_prospectus_text(raw_text)
    removed_chars = len(raw_text) - len(cleaned_text)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, FILENAME)
    with open(processed_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"[{FILENAME}] {len(raw_text)} chars -> {len(cleaned_text)} chars "
          f"({removed_chars} chars removed) -> Saved to {processed_path}")
    print("\nDone. Please spot-check data/processed/prospectus.txt, "
          "especially any pages with tables (fees, eligibility), before "
          "moving to build_vector_db.py.")

if __name__ == "__main__":
    clean_and_copy_prospectus()
