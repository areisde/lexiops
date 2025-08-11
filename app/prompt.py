# app/prompt.py
from __future__ import annotations
from typing import List, Dict, Tuple
import yaml
import hashlib
import os
from pathlib import Path

PROMPT_FILE = Path("configs/prompts.yaml")

def build_user_prompt(question: str, contexts: List[Dict], max_ctx_chars: int = 6000) -> str:
    ctx_parts, total = [], 0
    for c in contexts:
        header = f"Source: {c.get('title') or 'Untitled'} | {c.get('source')}"
        block = f"{header}\n{c['text'].strip()}\n"
        if total + len(block) > max_ctx_chars:
            break
        ctx_parts.append(block)
        total += len(block)
    context = "\n---\n".join(ctx_parts) if ctx_parts else "NO CONTEXT"
    return f"Question: {question}\n\nContext:\n{context}\n"

def load_prompt(version: str) -> Tuple[str, str, str]:
    """
    Returns (system, user_template, sha256)
    """
    cfg = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))
    p = cfg["prompts"].get(version)
    if not p:
        raise ValueError(f"Prompt '{version}' not found in {PROMPT_FILE}")
    # Stable hash of the exact text shipped to the model
    blob = (p["system"] + "\n---\n" + p["user_template"]).encode("utf-8")
    sha = hashlib.sha256(blob).hexdigest()
    return p["system"], p["user_template"], sha

def active_prompt_version() -> str:
    # env-driven for easy A/B; fallback to default
    return os.getenv("PROMPT_VERSION", "legal_v1.0.0")