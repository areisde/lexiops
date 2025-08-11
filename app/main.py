from __future__ import annotations
from pathlib import Path
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .prompt import build_user_prompt
from .llm import get_client_and_model, chat_complete
from .utils import (
    INDEX_PATH, STORE_PATH, EMBED_MODEL, EMBED_MODEL_NAME, INDEX, DOCS, 
    LLM_CLIENT, LLM_MODEL, RERANKER,
    _ensure_index_exists, _init_embeddings, _init_llm, _init_reranker_from_env,
    _encode_query, _faiss_search, _build_candidates, _apply_rerank_if_any,
    _citations_from_hits, active_prompt_version, load_prompt
)
from .version_tracking import get_version_manager, initialize_system_version

# Optional cross-encoder reranker (loaded only if available)
try:
    from .rerank import CrossEncoderReranker  # type: ignore
except Exception:
    CrossEncoderReranker = None  # type: ignore

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class Hit(BaseModel):
    score: float
    title: Optional[str]
    text: str
    source: str
    lang: str

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    hits: List[Hit]

# --------------------------------------------------------------------------------------
# App / Startup
# --------------------------------------------------------------------------------------
def load_resources():
    """Load all application resources and initialize version tracking"""
    global INDEX, DOCS
    
    print("🔄 Loading application resources...")
    
    # Load index and document store
    _ensure_index_exists()
    INDEX = faiss.read_index(str(INDEX_PATH))
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    DOCS = store["chunks"]
    
    # Initialize models and components
    _init_embeddings(store)
    _init_llm()
    _init_reranker_from_env()
    
    # Initialize MLflow GenAI version tracking
    try:
        system_version = initialize_system_version()
        print(f"✅ System version initialized: {system_version}")
    except Exception as e:
        print(f"⚠️  Version tracking initialization failed: {e}")
    
    print(f"✅ Resources loaded: {len(DOCS)} documents, {EMBED_MODEL_NAME}")

app = FastAPI(on_startup=[load_resources])

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    """Health check endpoint with version information"""
    version_info = {}
    try:
        manager = get_version_manager()
        version_info = {
            "system_version": manager.get_active_version(),
            "git_commit": manager.get_git_commit(),
        }
    except Exception:
        version_info = {"system_version": "unknown", "git_commit": "unknown"}
    
    return {
        "status": "ok",
        "chunks": len(DOCS),
        "embed_model": EMBED_MODEL_NAME,
        "llm_model": LLM_MODEL,
        "reranker_enabled": bool(RERANKER),
        **version_info,
    }

@app.get("/ask", response_model=AskResponse)
def ask(
    q: str = Query(..., min_length=3, description="User question"),
    k: int = Query(5, ge=1, le=20, description="Top-k chunks to retrieve"),
    pv: Optional[str] = Query(None, description="Prompt version override (e.g., legal_v1.1.0)"),
    rv: Optional[str] = Query(None, description="Retrieval version override (e.g., v1.1.0 for rerank)"),
):
    t0 = time.time()
    try:
        # --- 1) Retrieval (versioned) ---
        retrieval_version = rv or os.getenv("RETRIEVAL_VERSION", "v1.0.0")
        t_retr_start = time.time()
        qv = _encode_query(q)

        # Decide candidate pool & whether to apply reranking
        apply_rerank = (retrieval_version != "v1.0.0") and (RERANKER is not None)
        candidate_k = max(k * 6, 30) if apply_rerank else k

        scores, idxs = _faiss_search(qv, candidate_k)
        candidates = _build_candidates(scores, idxs)
        raw_hits = _apply_rerank_if_any(q, candidates, k, apply_rerank)
        citations = _citations_from_hits(raw_hits)

        t_retr_ms = int((time.time() - t_retr_start) * 1000)

        # --- 2) Prompt versioning & assembly ---
        prompt_version = pv or active_prompt_version()
        system_text, user_tpl, prompt_sha = load_prompt(prompt_version)

        # Build the user prompt body (question + context sections)
        user_body = build_user_prompt(q, raw_hits)  # returns "Question: ...\n\nContext:\n..."
        ctx_part = user_body.split("Context:\n", 1)[1] if "Context:\n" in user_body else "NO CONTEXT"
        user_text = user_tpl.replace("{{question}}", q).replace("{{context}}", ctx_part)

        # --- 3) LLM call ---
        t_llm_start = time.time()
        answer = chat_complete(LLM_CLIENT, LLM_MODEL, system_text, user_text)  # type: ignore
        t_llm_ms = int((time.time() - t_llm_start) * 1000)
        t_total_ms = int((time.time() - t0) * 1000)

        # --- 4) Response ---
        hits_for_response = [Hit(**h) for h in raw_hits]
        return AskResponse(query=q, answer=answer, citations=citations, hits=hits_for_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))