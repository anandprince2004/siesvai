"""
SIESVAI - Step 1: Data Collection
Scrapes plain-text content from key SIES ASCN college pages and saves
each page as a clean .txt file in data/raw/.

Usage:
    python scripts/scrape.py

Notes:
- Only handles HTML pages (not PDFs) — PDF extraction is a separate step.
- Strips navigation, scripts, styles, and footer boilerplate as best as
  possible, keeping the actual body content.
- If a page's structure changes or scraping fails, the script logs an error
  and continues to the next page instead of crashing.
"""

import os
import time
import requests
from bs4 import BeautifulSoup

PAGES = {
    "admissions": "https://siesascn.edu.in/admissions",
    "courses_syllabus": "https://siesascn.edu.in/courses-syllabus",
    "contact": "https://siesascn.edu.in/contact-us",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TAGS_TO_REMOVE = ["script", "style", "nav", "footer", "header", "noscript", "form"]

def fetch_page(url: str) -> str:
    """Fetch raw HTML for a given URL. Raises on network/HTTP errors."""
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.text

def extract_clean_text(html: str) -> str:
    """
    Parse HTML and return clean, readable text.
    Removes nav/footer/script/style clutter and collapses excess whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    raw_text = soup.get_text(separator="\n")

    lines = [line.strip() for line in raw_text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    cleaned_text = "\n".join(non_empty_lines)

    return cleaned_text

def save_text(filename: str, content: str) -> None:
    """Save text content to data/raw/<filename>.txt"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved -> {filepath}")

def scrape_all_pages() -> None:
    """Loop through PAGES, scrape each, and save results."""
    print(f"Starting scrape of {len(PAGES)} page(s)...\n")

    for name, url in PAGES.items():
        print(f"[{name}] Fetching {url}")
        try:
            html = fetch_page(url)
            text = extract_clean_text(html)

            if len(text) < 100:
                print(f"  WARNING: extracted text is very short ({len(text)} chars). "
                      f"Page may use JS rendering or a different structure. "
                      f"Saving anyway for manual inspection.")

            save_text(name, text)

        except requests.exceptions.RequestException as e:
            print(f"  ERROR: failed to fetch {url} -> {e}")

        time.sleep(1)

    print("\nDone. Check data/raw/ and manually review each file for quality.")

if __name__ == "__main__":
    scrape_all_pages()
