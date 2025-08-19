import pandas as pd
import fitz
import numpy as np
import faiss
import json
import time
from sentence_transformers import SentenceTransformer
from typing import Union, List, Dict, Any, Optional
from .hash import _load_config_file
from pathlib import Path
import hashlib

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start += chunk_size - overlap
    
    return chunks


def chunk_pdfs(doc_records: List[Dict[str, Any]], root_dir: Path) -> 'pd.DataFrame':
    """
    Chunk PDF files deterministically based on manifest records.
    
    Parameters:
      - doc_records: list of document records with structure like
          {"file_path": ..., "doc_sha": ..., "file_size": ..., "mtime": ..., "storage_path": ...}
      - root_dir: project root directory for loading chunker config
      
    Returns:
      - DataFrame with columns: chunk_id, doc_sha, file_name, page, chunk_idx, length_chars, text
    """
    # Load chunking config
    chunker_config = _load_config_file(root_dir / "config" / "chunker.yaml")
    chunk_size = chunker_config.get("chunk_size", 1000)
    overlap = chunker_config.get("overlap", 200)
    
    # Sort records by doc_sha, then file_path for deterministic order
    sorted_records = sorted(doc_records, key=lambda x: (x.get("doc_sha", ""), x.get("file_path", "")))
    
    all_rows = []
    
    for record in sorted_records:
        doc_sha = record.get("doc_sha")
        storage_path = record.get("storage_path")
        file_path = record.get("file_path", "")
        file_name = Path(file_path).name if file_path else "unknown"
        
        if not doc_sha or not storage_path:
            continue
            
        try:
            # Extract text from PDF using PyMuPDF
            with fitz.open(storage_path) as pdf_doc:
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    page_text = page.get_text()
                    
                    # Skip empty pages
                    if not page_text.strip():
                        continue
                    
                    # Chunk the page text
                    chunks = chunk_text(page_text, chunk_size, overlap)
                    
                    for chunk_idx, chunk_text_content in enumerate(chunks):
                        # Create deterministic chunk_id
                        chunk_id = hashlib.sha256(chunk_text_content.encode("utf-8")).hexdigest()
                        
                        all_rows.append({
                            "chunk_id": chunk_id,
                            "doc_sha": doc_sha,
                            "file_name": file_name,
                            "page": page_num,
                            "chunk_idx": chunk_idx,
                            "length_chars": len(chunk_text_content),
                            "text": chunk_text_content
                        })
                        
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            continue
    
    # Create DataFrame and sort by (doc_sha, page, chunk_idx)
    if not all_rows:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["chunk_id", "doc_sha", "file_name", "page", "chunk_idx", "length_chars", "text"])
    
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["doc_sha", "page", "chunk_idx"]).reset_index(drop=True)
    
    print(f"✂️ Chunking complete: {len(df)} chunks from {len(sorted_records)} documents")
    return df


def save_chunks_table(snapshot_id: str, df: pd.DataFrame, root_dir: Path, sample_size: int = 5000) -> None:
    """
    Persist the chunks DataFrame as Parquet files with Snappy compression.

    Parameters:
        - df: DataFrame containing chunked text data.
        - root_dir: Project root directory for determining output paths.
        - sample_size: Number of rows to include in the sample file for UI preview (default: 5000).

    Saves:
        - Full chunks table at data/chunks/chunks@{snapshot_id}.parquet
        - Sample table at data/chunks/chunks_sample@{snapshot_id}.parquet
    """
    chunks_dir = root_dir / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_path = root_dir / "data" / "chunks" / f"chunks@{snapshot_id}.parquet"
    df.to_parquet(chunks_path, compression="snappy", index=False)
    
    print(f"💾 Chunks saved: {chunks_path} ({len(df)} rows)")


def create_embeddings_and_index(df: pd.DataFrame, snapshot_id: str, root_dir: Path) -> None:
    """
    Create embeddings and FAISS index from chunks DataFrame.
    
    Parameters:
      - df: DataFrame with columns: chunk_id, doc_sha, file_name, page, chunk_idx, length_chars, text
      - snapshot_id: unique snapshot identifier for output directory
      - root_dir: project root directory
      
    Saves:
      - data/index/{snapshot_id}/faiss.index
      - data/index/{snapshot_id}/mapping.jsonl
      - data/index/{snapshot_id}/index_stats.json
    """
    start_time = time.time()
    
    # Load embedder config
    embedder_config = _load_config_file(root_dir / "config" / "embedder.yaml")
    model_name = embedder_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    normalize = embedder_config.get("normalize", True)
    
    print(f"🔢 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Create output directory
    index_dir = root_dir / "data" / "index" / snapshot_id
    index_dir.mkdir(parents=True, exist_ok=True)
    
    if len(df) == 0:
        print("⚠️ No chunks to embed")
        return
    
    # Generate embeddings
    print(f"🔢 Generating embeddings for {len(df)} chunks...")
    texts = df["text"].tolist()
    embeddings = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    
    # Save FAISS index
    faiss_path = index_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    
    # Save mapping (one line per row)
    mapping_path = index_dir / "mapping.jsonl"
    with open(mapping_path, "w") as f:
        for idx, (row_id, row) in enumerate(df.iterrows()):
            mapping_entry = {
                "row_id": idx,
                "chunk_id": str(row["chunk_id"]),
                "doc_sha": str(row["doc_sha"]),
                "file_name": str(row["file_name"]),
                "page": int(row["page"]),
                "chunk_idx": int(row["chunk_idx"]),
                "length_chars": int(row["length_chars"])
            }
            f.write(json.dumps(mapping_entry) + "\n")
    
    # Calculate stats
    build_time = time.time() - start_time
    faiss_size = faiss_path.stat().st_size
    mapping_size = mapping_path.stat().st_size
    
    # Save index stats
    stats = {
        "num_chunks": len(df),
        "embedding_dimension": int(dimension),
        "index_type": "IndexFlatIP",
        "build_time_seconds": round(build_time, 2),
        "model_name": model_name,
        "normalize": normalize,
        "file_sizes": {
            "faiss_index_bytes": faiss_size,
            "mapping_jsonl_bytes": mapping_size
        }
    }
    
    stats_path = index_dir / "index_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"🔍 Index complete: {len(df)} chunks, {dimension}D embeddings, {build_time:.1f}s")
    print(f"📁 Saved to: {index_dir}")

    