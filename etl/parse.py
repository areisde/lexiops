from pathlib import Path
from typing import List, Dict, Any
import regex as re
from pdfminer.high_level import extract_text

# Heuristics to split legal PDFs into articles/sections.
# Supports English/French/German "Article", "Art.", "Art".
ARTICLE_PATTERN = re.compile(
    r'(?mi)^(?:\s*)'                     # start of line, optional space
    r'((?:Article|Art\.?|Artikel)\s+\d+[a-zA-Z\-]*)'  # heading with number
    r'[\s:.-]*\n?'                       # punctuation/newline after heading
)

def extract_articles(raw_path: Path, doc_type: str, lang: str) -> List[Dict[str, Any]]:
    if doc_type.lower() != "pdf":
        # Fallback: treat as HTML/plain (optional: improve later)
        text = raw_path.read_text(encoding="utf-8", errors="ignore")
    else:
        # Extract full text from PDF
        text = extract_text(str(raw_path)) or ""

    text = text.replace("\r", "")
    if not text.strip():
        return []

    # Split by article-like headings; keep the heading in each chunk
    parts = []
    last_idx = 0
    for m in ARTICLE_PATTERN.finditer(text):
        start = m.start()
        if start > 0:
            parts.append(text[last_idx:start])
        last_idx = start
    parts.append(text[last_idx:])  # tail

    articles: List[Dict[str, Any]] = []
    for i, block in enumerate(parts):
        block = block.strip()
        if not block:
            continue
        # Title = first line (often "Article 1 ..." / "Art. 1 ...")
        first_line = block.splitlines()[0].strip()
        title = first_line[:200]
        articles.append({
            "id": i,
            "title": title,
            "text": block,
            "lang": lang,
            "source": str(raw_path),
        })
    # If the split failed (e.g., no headings detected), fall back to one big "article"
    if not articles:
        articles = [{
            "id": 0,
            "title": "Document",
            "text": text.strip(),
            "lang": lang,
            "source": str(raw_path),
        }]
    return articles