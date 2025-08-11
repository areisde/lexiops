from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/index/index.faiss")
STORE_PATH = Path("data/index/docstore.json")
EMBED_MODEL = None
INDEX = None
DOCS = None

class Hit(BaseModel):
    score: float
    title: str | None
    text: str
    source: str
    lang: str

class QueryResponse(BaseModel):
    query: str
    hits: List[Hit]

def load_resources():
    global EMBED_MODEL, INDEX, DOCS
    if not INDEX_PATH.exists() or not STORE_PATH.exists():
        raise RuntimeError("Index not found. Build it first.")
    INDEX = faiss.read_index(str(INDEX_PATH))
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    DOCS = store["chunks"]
    model_name = store.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    EMBED_MODEL = SentenceTransformer(model_name)

app = FastAPI(on_startup=[load_resources])

@app.get("/healthz")
def healthz():
    return {"status": "ok", "chunks": len(DOCS)}

@app.get("/search", response_model=QueryResponse)
def search(q: str = Query(..., min_length=3), k: int = 5):
    try:
        qv = EMBED_MODEL.encode([q], normalize_embeddings=True).astype("float32")
        scores, idxs = INDEX.search(qv, k)
        hits: List[Hit] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            d = DOCS[idx]
            hits.append(Hit(
                score=float(score),
                title=d.get("title"),
                text=d["text"],
                source=d["source"],
                lang=d["lang"],
            ))
        return QueryResponse(query=q, hits=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))