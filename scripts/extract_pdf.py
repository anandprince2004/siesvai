import os
import re
import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO

HOMEPAGE_URL = "https://siesascn.edu.in/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILENAME = "prospectus.txt"

PROSPECTUS_LINK_TEXT = "download prospectus"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def find_prospectus_url() -> str:
    """
    Fetch the homepage and find the href of the link labeled
    "DOWNLOAD PROSPECTUS". Raises RuntimeError if no such link is found,
    rather than guessing or falling back to a hardcoded URL - a missing
    link means the site structure changed and needs a human to check.
    """
    print(f"Fetching homepage: {HOMEPAGE_URL}")
    response = requests.get(HOMEPAGE_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a", href=True):
        link_text = link.get_text(strip=True).lower()
        if PROSPECTUS_LINK_TEXT in link_text:
            prospectus_url = link["href"]
            if prospectus_url.startswith("/"):
                prospectus_url = "https://siesascn.edu.in" + prospectus_url
            print(f"Found prospectus link: {prospectus_url}")
            return prospectus_url

    raise RuntimeError(
        f"Could not find a link containing '{PROSPECTUS_LINK_TEXT}' on the "
        f"homepage. The site's structure may have changed - check "
        f"{HOMEPAGE_URL} manually and update this script if needed."
    )

def download_pdf(url: str) -> bytes:
    """Download the PDF and return its raw bytes."""
    print(f"Downloading PDF...")
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower():
        print(f"  WARNING: response Content-Type is '{content_type}', not "
              f"PDF. The download may have failed or returned an error "
              f"page instead of the actual file.")

    print(f"  Downloaded {len(response.content) / 1024:.1f} KB")
    return response.content

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from the PDF, page by page, using pdfplumber.
    Each page's text is separated with a clear marker so it's easy to see
    in the output where page boundaries fall - useful for manual review
    and for later chunking decisions.
    """
    print("Extracting text from PDF...")
    all_pages_text = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        print(f"  PDF has {total_pages} page(s)")

        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                all_pages_text.append(f"--- Page {i} ---\n{page_text.strip()}")
            else:
                print(f"  WARNING: page {i} extracted no text "
                      f"(may be an image/scanned page).")

    full_text = "\n\n".join(all_pages_text)
    return full_text

def save_text(content: str) -> None:
    """Save extracted text to data/raw/prospectus.txt"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved -> {filepath}")

def main():
    try:
        prospectus_url = find_prospectus_url()
        pdf_bytes = download_pdf(prospectus_url)
        text = extract_text_from_pdf(pdf_bytes)

        if len(text) < 200:
            print(f"\nWARNING: extracted text is very short "
                  f"({len(text)} chars). The PDF may be scanned/image-based "
                  f"rather than text-based, in which case pdfplumber cannot "
                  f"extract it directly - would need OCR instead.")

        save_text(text)
        print(f"\nDone. Extracted {len(text)} characters from "
              f"{len(text.split(chr(10) + chr(10)))} page section(s). "
              f"Please manually review data/raw/{OUTPUT_FILENAME} for quality.")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: network request failed -> {e}")
    except RuntimeError as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
