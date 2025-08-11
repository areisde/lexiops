from __future__ import annotations
from pathlib import Path
from typing import Tuple
import yaml, hashlib, os

PROMPT_FILE = Path("configs/prompts.yaml")

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