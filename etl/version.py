from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from .hash import _load_config_file, _canonical_serialize


def create_snapshot_manifest(
    snapshot_id: str,
    fingerprint: str,
    corpus_digest: str,
    config_digest: str,
    git_sha: str,
    doc_records: List[Dict[str, Any]],
    chunks_df: pd.DataFrame,
    root_dir: Path
) -> None:
    """
    Create and save a snapshot manifest as the source of truth.
    
    Parameters:
      - snapshot_id: unique snapshot identifier
      - fingerprint: combined fingerprint hash
      - corpus_digest: digest from document records
      - config_digest: digest from configuration files
      - git_sha: git commit hash (or "nogit")
      - doc_records: list of document records
      - chunks_df: DataFrame with chunked data
      - root_dir: project root directory
      
    Saves:
      - data/snapshots/{snapshot_id}.json
    """
    # Load current configs
    chunker_config = _load_config_file(root_dir / "config" / "chunker.yaml")
    embedder_config = _load_config_file(root_dir / "config" / "embedder.yaml")
    index_config = _load_config_file(root_dir / "config" / "index.yaml")
    
    # Compute individual config hashes
    chunker_sha256 = hashlib.sha256(_canonical_serialize(chunker_config).encode()).hexdigest()
    embedder_sha256 = hashlib.sha256(_canonical_serialize(embedder_config).encode()).hexdigest()
    index_sha256 = hashlib.sha256(_canonical_serialize(index_config).encode()).hexdigest()
    
    # Prepare sources list
    sources = []
    for record in doc_records:
        sources.append({
            "file": record.get("file_path", ""),
            "doc_sha": record.get("doc_sha", ""),
            "size_bytes": record.get("file_size", 0),
            "mtime": record.get("mtime", 0)
        })
    
    # Calculate stats
    total_docs = len(doc_records)
    total_chunks = len(chunks_df)
    avg_len_chars = int(chunks_df["length_chars"].mean()) if total_chunks > 0 else 0
    
    # Get embedder dimension (assume it's available from config or calculate)
    model_name = embedder_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    # For common models, we can infer dimensions, but this is a simplification
    dimension_map = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }
    embedding_dim = dimension_map.get(model_name, 384)  # default fallback
    
    # Create manifest
    print(f"🐛 DEBUG - Creating manifest with fingerprint: {fingerprint} (type: {type(fingerprint)})")
    manifest = {
        "identity": {
            "id": snapshot_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "git_sha": git_sha,
            "fingerprint": fingerprint,
            "corpus_digest": corpus_digest,
            "config_digest": config_digest
        },
        "sources": sources,
        "chunking": {
            "method": "character_window",
            "size": chunker_config.get("chunk_size", 1000),
            "overlap": chunker_config.get("overlap", 200),
            "extra_normalization_flags": chunker_config.get("normalization", {})
        },
        "embedder": {
            "name": model_name,
            "version": embedder_config.get("version", "latest"),
            "params": {k: v for k, v in embedder_config.items() if k not in ["model_name", "version"]},
            "normalized": embedder_config.get("normalize", True),
            "dim": embedding_dim
        },
        "index": {
            "type": index_config.get("type", "IndexFlatIP"),
            "params": {k: v for k, v in index_config.items() if k != "type"},
            "path": f"data/index/{snapshot_id}/faiss.index",
            "mapping": f"data/index/{snapshot_id}/mapping.jsonl"
        },
        "chunks": {
            "parquet": f"data/chunks/chunks@{snapshot_id}.parquet"
        },
        "stores": {
            "raw": str(root_dir / "data" / "raw"),
            "chunks": str(root_dir / "data" / "chunks"),
            "index": str(root_dir / "data" / "index")
        },
        "stats": {
            "docs": total_docs,
            "chunks": total_chunks,
            "avg_len_chars": avg_len_chars
        },
        "config_shas": {
            "chunker_sha256": chunker_sha256,
            "embedder_sha256": embedder_sha256,
            "index_sha256": index_sha256
        }
    }
    
    # Save manifest
    snapshots_dir = root_dir / "data" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = snapshots_dir / f"{snapshot_id}.json"
    
    # Debug the manifest before saving
    print(f"🐛 DEBUG - Manifest fingerprint before save: {manifest['identity']['fingerprint']}")
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Update pointer files - overwrite data/LATEST with snapshot_id
    latest_file = root_dir / "data" / "LATEST"
    with open(latest_file, "w") as f:
        f.write(snapshot_id)
    
    print(f"📋 Snapshot manifest saved: {manifest_path}")
    print(f"🔗 Updated LATEST pointer: {snapshot_id}")