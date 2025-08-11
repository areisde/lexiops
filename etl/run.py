# etl/run.py
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import mlflow
import yaml

from .fetch import download  # (name:str, url:str, dest_dir:Path) -> Path
from .parse import extract_articles  # (raw_path:Path, doc_type:str, lang:str) -> List[Dict[str, Any]]
from .chunk import make_chunks  # (articles:List[dict], chunk_size:int, overlap:int) -> List[Dict[str, Any]]

# ---------- small helpers ----------
def write_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# ---------- main ETL ----------
def run(config_path: Path, data_root: Path, chunk_size: int, overlap: int, experiment: str) -> None:
    # MLflow setup
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"etl_{now_stamp()}"):
        # log generic params
        mlflow.log_params({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "config_path": str(config_path),
        })

        # load config
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        sources = cfg.get("sources", [])
        mlflow.log_param("source_count", len(sources))

        # ensure folders
        raw_dir = data_root / "raw"
        clean_dir = data_root / "clean"
        chunks_dir = data_root / "chunks"
        for d in (raw_dir, clean_dir, chunks_dir):
            d.mkdir(parents=True, exist_ok=True)

        total_articles = 0
        total_chunks = 0

        for s in sources:
            name = s["name"]
            url = s["url"]
            doc_type = s.get("type", "pdf")
            lang = s.get("lang", "en")

            # 1) FETCH
            dest_dir = raw_dir / name / now_stamp()
            raw_path = download(name=name, url=url, dest_dir=dest_dir)

            # 2) PARSE -> article-level JSONL
            articles = extract_articles(raw_path=raw_path, doc_type=doc_type, lang=lang)
            total_articles += len(articles)
            articles_path = clean_dir / name / "articles.jsonl"
            write_jsonl(articles, articles_path)

            # 3) CHUNK
            chunks = make_chunks(articles=articles, chunk_size=chunk_size, overlap=overlap)
            total_chunks += len(chunks)
            chunks_path = chunks_dir / name / "chunks.jsonl"
            write_jsonl(chunks, chunks_path)

            # 4) MLflow logging per-source
            mlflow.log_params({
                f"{name}_lang": lang,
                f"{name}_type": doc_type,
            })
            mlflow.log_metrics({
                f"{name}_articles": len(articles),
                f"{name}_chunks": len(chunks),
            })
            # Log artifacts (small, representative files; raw can be big—log the folder once)
            if dest_dir.exists():
                mlflow.log_artifacts(str(dest_dir), artifact_path=f"raw/{name}")
            mlflow.log_artifact(str(articles_path), artifact_path=f"clean/{name}")
            mlflow.log_artifact(str(chunks_path), artifact_path=f"chunks/{name}")

        # run-level metrics
        mlflow.log_metrics({
            "total_articles": total_articles,
            "total_chunks": total_chunks,
        })

def cli():
    p = argparse.ArgumentParser(description="lexiops ETL (fetch -> parse -> chunk -> MLflow)")
    p.add_argument("--config", default="configs/sources.yaml", type=Path)
    p.add_argument("--data-root", default="data", type=Path)
    p.add_argument("--chunk-size", default=1000, type=int)
    p.add_argument("--overlap", default=150, type=int)
    p.add_argument("--experiment", default="etl-index", type=str)
    args = p.parse_args()
    run(args.config, args.data_root, args.chunk_size, args.overlap, args.experiment)

if __name__ == "__main__":
    cli()