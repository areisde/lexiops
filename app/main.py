from __future__ import annotations
from pathlib import Path
import json, time, os
from typing import List, Dict, Any

import faiss
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .prompt_registry import load_prompt, active_prompt_version
from .prompt import build_user_prompt
from .llm import get_client_and_model, chat_complete

# --- Paths to your built index ---
INDEX_PATH = Path("data/index/index.faiss")
STORE_PATH = Path("data/index/docstore.json")

# --- Globals populated at startup ---
EMBED_MODEL: SentenceTransformer | None = None
INDEX = None
DOCS: List[Dict[str, Any]] = []
LLM_CLIENT = None
LLM_MODEL = None
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Pydantic models ----------
class Hit(BaseModel):
    score: float
    title: str | None
    text: str
    source: str
    lang: str

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    hits: List[Hit]

# ---------- App & startup ----------
def load_resources():
    global EMBED_MODEL, INDEX, DOCS, LLM_CLIENT, LLM_MODEL, EMBED_MODEL_NAME
    if not INDEX_PATH.exists() or not STORE_PATH.exists():
        raise RuntimeError("Index not found. Build it first (etl + index).")
    INDEX = faiss.read_index(str(INDEX_PATH))
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    DOCS = store["chunks"]
    EMBED_MODEL_NAME = store.get("embed_model", EMBED_MODEL_NAME)
    EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    LLM_CLIENT, LLM_MODEL = get_client_and_model()

app = FastAPI(on_startup=[load_resources])

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "chunks": len(DOCS), "embed_model": EMBED_MODEL_NAME, "llm_model": LLM_MODEL}

@app.get("/ask", response_model=AskResponse)
def ask(
    q: str = Query(..., min_length=3, description="User question"),
    k: int = Query(5, ge=1, le=20, description="Top-k chunks to retrieve"),
    pv: str | None = Query(None, description="Prompt version override (e.g., legal_v1.1.0)")
):
    t0 = time.time()
    try:
        # --- 1) Retrieval ---
        t_retr_start = time.time()
        qv = EMBED_MODEL.encode([q], normalize_embeddings=True).astype("float32")
        scores, idxs = INDEX.search(qv, k)
        raw_hits: List[Dict[str, Any]] = []
        citations: List[str] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            d = DOCS[idx]
            raw_hits.append({
                "score": float(score),
                "title": d.get("title"),
                "text": d["text"],
                "source": d["source"],
                "lang": d["lang"],
            })
            human_cit = (d.get("title") or "Untitled")
            src_tail = d.get("source", "")
            if len(src_tail) > 80:
                src_tail = "…" + src_tail[-80:]
            citations.append(f"{human_cit} — {src_tail}")
        t_retr = time.time() - t_retr_start

        # --- 2) Prompt versioning & assembly ---
        prompt_version = pv or active_prompt_version()
        system_text, user_tpl, prompt_sha = load_prompt(prompt_version)

        # Build the user prompt body (question + context sections)
        user_body = build_user_prompt(q, raw_hits)  # returns "Question: ...\n\nContext:\n..."
        # Extract just the context part to fill {{context}}
        ctx_part = user_body.split("Context:\n", 1)[1] if "Context:\n" in user_body else "NO CONTEXT"
        user_text = (
            user_tpl
            .replace("{{question}}", q)
            .replace("{{context}}", ctx_part)
        )

        # --- 3) LLM call ---
        t_llm_start = time.time()
        answer = chat_complete(LLM_CLIENT, LLM_MODEL, system_text, user_text)
        t_llm = time.time() - t_llm_start

        # --- 4) Log a compact trace to MLflow ---
        mlflow.set_experiment("inference")
        with mlflow.start_run(run_name="ask"):
            mlflow.log_params({
                "k": k,
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL_NAME,
                "prompt_version": prompt_version,
                "prompt_sha256": prompt_sha,
            })
            mlflow.log_metrics({
                "t_retrieval_ms": int(t_retr * 1000),
                "t_llm_ms": int(t_llm * 1000),
                "t_total_ms": int((time.time() - t0) * 1000),
            })
            trace = {
                "query": q,
                "prompt_version": prompt_version,
                "prompt_sha256": prompt_sha,
                "system_prompt": system_text,
                "user_prompt": user_text,
                "hits": raw_hits,
                "citations": citations,
                "answer": answer,
                "timings_ms": {
                    "retrieval": int(t_retr * 1000),
                    "llm": int(t_llm * 1000),
                    "total": int((time.time() - t0) * 1000),
                },
            }
            tmp = Path("trace.json")
            tmp.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(tmp), artifact_path="trace")
            tmp.unlink(missing_ok=True)

        # --- 5) Response ---
        hits_for_response = [Hit(**h) for h in raw_hits]
        return AskResponse(query=q, answer=answer, citations=citations, hits=hits_for_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))