from typing import List, Dict, Any

def _split_text(t: str, chunk_size: int, overlap: int) -> List[str]:
    # simple char-based splitter for now; replace with token-based later
    chunks = []
    i = 0
    while i < len(t):
        chunks.append(t[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def make_chunks(articles: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    out = []
    for a in articles:
        for j, piece in enumerate(_split_text(a["text"], chunk_size, overlap)):
            out.append({
                "article_id": a["id"],
                "chunk_id": j,
                "text": piece,
                "lang": a["lang"],
                "source": a["source"],
            })
    return out