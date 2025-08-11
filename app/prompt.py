# app/prompt.py
from __future__ import annotations
from typing import List, Dict, Tuple

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