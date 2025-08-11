import json
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
from sentence_transformers import SentenceTransformer

from .llm import get_client_and_model

# Optional cross-encoder reranker
try:
    from .rerank import CrossEncoderReranker  # type: ignore
except Exception:
    CrossEncoderReranker = None  # type: ignore

# --------------------------------------------------------------------------------------
# Paths / Globals
# --------------------------------------------------------------------------------------
INDEX_PATH = Path("data/index/index.faiss")
STORE_PATH = Path("data/index/docstore.json")

EMBED_MODEL: Optional[SentenceTransformer] = None
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX = None
DOCS: List[Dict[str, Any]] = []

LLM_CLIENT = None
LLM_MODEL: Optional[str] = None

RERANKER = None  # type: ignore

def _ensure_index_exists() -> None:
    if not INDEX_PATH.exists() or not STORE_PATH.exists():
        raise RuntimeError("Index not found. Build it first (etl + index).")

def _init_embeddings(store: Dict[str, Any]) -> None:
    global EMBED_MODEL, EMBED_MODEL_NAME
    EMBED_MODEL_NAME = store.get("embed_model", EMBED_MODEL_NAME)
    EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

def _init_llm() -> None:
    global LLM_CLIENT, LLM_MODEL
    LLM_CLIENT, LLM_MODEL = get_client_and_model()

def _init_reranker_from_env() -> None:
    """Initialize cross-encoder reranker if installed and enabled via USE_RERANKER env."""
    global RERANKER
    use_reranker = os.getenv("USE_RERANKER", "").lower() in {"1", "true", "yes"}
    if use_reranker and CrossEncoderReranker is not None:
        try:
            RERANKER = CrossEncoderReranker()
        except Exception:
            RERANKER = None
    else:
        RERANKER = None

def _encode_query(q: str):
    return EMBED_MODEL.encode([q], normalize_embeddings=True).astype("float32")  # type: ignore

def _faiss_search(qv, k: int) -> Tuple[List[float], List[int]]:
    scores, idxs = INDEX.search(qv, k)  # type: ignore
    return scores[0].tolist(), idxs[0].tolist()

def _build_candidates(scores: List[float], idxs: List[int]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for score, idx in zip(scores, idxs):
        d = DOCS[idx]
        candidates.append({
            "score": float(score),
            "title": d.get("title"),
            "text": d["text"],
            "source": d["source"],
            "lang": d["lang"],
        })
    return candidates

def _apply_rerank_if_any(query: str, candidates: List[Dict[str, Any]], k: int, use_rerank: bool) -> List[Dict[str, Any]]:
    if use_rerank and RERANKER is not None:
        return RERANKER.rerank(query, candidates, top_k=k)  # type: ignore
    return candidates[:k]

def _citations_from_hits(hits: List[Dict[str, Any]]) -> List[str]:
    """Extract citations from search hits for response formatting"""
    citations: List[str] = []
    for h in hits:
        title = h.get("title") or "Untitled"
        src_tail = h.get("source", "")
        if len(src_tail) > 80:
            src_tail = "…" + src_tail[-80:]
        citations.append(f"{title} — {src_tail}")
    return citations

def active_prompt_version() -> str:
    """Get the active prompt version from environment or default"""
    return os.getenv("PROMPT_VERSION", "legal_v1.0.0")

def load_prompt(version: str) -> Tuple[str, str, str]:
    """Load prompt templates from configs/prompts.yaml and return (system_prompt, user_template, sha256)
    
    Args:
        version: Prompt version identifier (e.g., "legal_v1.0.0")
        
    Returns:
        Tuple of (system_prompt, user_template, content_hash)
    """
    prompts_path = Path("configs/prompts.yaml")
    
    # Load prompts configuration
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts configuration not found: {prompts_path}")
    
    with open(prompts_path, 'r') as f:
        prompts_config = yaml.safe_load(f)
    
    # Get available prompt versions
    available_prompts = prompts_config.get("prompts", {})
    if not available_prompts:
        raise ValueError("No prompts found in prompts.yaml")
    
    # Try to find the requested version, fallback to first available
    if version in available_prompts:
        prompt_config = available_prompts[version]
        used_version = version
    else:
        # Use first available version as fallback
        used_version = list(available_prompts.keys())[0]
        prompt_config = available_prompts[used_version]
        print(f"Warning: Prompt version '{version}' not found, using '{used_version}'")
    
    # Extract system and user prompts
    system_prompt = prompt_config.get("system", "").strip()
    user_template = prompt_config.get("user_template", "").strip()
    
    if not system_prompt or not user_template:
        raise ValueError(f"Invalid prompt configuration in {used_version}: missing 'system' or 'user_template'")
    
    # Generate content hash for version tracking
    content = f"{system_prompt}{user_template}"
    prompt_sha = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    print(f"✅ Loaded prompts from config version: {used_version}")
    return system_prompt, user_template, prompt_sha