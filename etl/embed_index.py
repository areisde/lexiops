from __future__ import annotations
from pathlib import Path
import json, os
from typing import List, Dict, Any
import numpy as np
import mlflow
from sentence_transformers import SentenceTransformer
import faiss

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_index(chunks_dir: Path, out_dir: Path, model_name: str="sentence-transformers/all-MiniLM-L6-v2"):
    mlflow.set_experiment("embedding-index")
    with mlflow.start_run(run_name="build_index"):
        mlflow.log_param("embed_model", model_name)

        # load all chunk files
        all_chunks: List[Dict[str, Any]] = []
        for src_dir in chunks_dir.iterdir():
            f = src_dir / "chunks.jsonl"
            if f.exists():
                all_chunks.extend(load_jsonl(f))
        mlflow.log_metric("chunk_count", len(all_chunks))
        if not all_chunks:
            raise RuntimeError("No chunks found. Run ETL first.")

        # embed
        model = SentenceTransformer(model_name)
        texts = [c["text"] for c in all_chunks]
        vecs = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype="float32")

        # faiss index
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine equivalent because we normalized
        index.add(vecs)

        # persist
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = out_dir / "index.faiss"
        meta_path = out_dir / "docstore.json"
        faiss.write_index(index, str(faiss_path))
        save_json({"chunks": all_chunks, "embed_model": model_name}, meta_path)

        # log artifacts
        mlflow.log_artifact(str(faiss_path), artifact_path="index")
        mlflow.log_artifact(str(meta_path), artifact_path="index")
        return faiss_path, meta_path