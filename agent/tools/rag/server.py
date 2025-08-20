"""
RAG MCP Server
"""

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from langchain_core.tools import tool
import pandas as pd
import faiss
import yaml
from typing import Union
from sentence_transformers import SentenceTransformer
import logging
import numpy as np

# Initialize the FastMCP server
mcp = FastMCP("rag-server")

# Load configuration
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config/agent/rag.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_ROOT_URI = ROOT_DIR / "data"

# Resolve SNAPSHOT_ID
SNAPSHOT_ID = config["snapshot"]
if SNAPSHOT_ID == "LATEST":
    with open(DATA_ROOT_URI / "LATEST", "r") as f:
        SNAPSHOT_ID = f.read().strip()

# Paths
chunks_path = DATA_ROOT_URI / config["paths"]["chunks_parquet"].format(SNAPSHOT_ID=SNAPSHOT_ID)
faiss_index_path = DATA_ROOT_URI / config["paths"]["faiss_index"].format(SNAPSHOT_ID=SNAPSHOT_ID)
mapping_path = DATA_ROOT_URI / config["paths"]["mapping"].format(SNAPSHOT_ID=SNAPSHOT_ID)
LATEST_SNAPSHOT = DATA_ROOT_URI / "snapshots" / f"{SNAPSHOT_ID}.json"
with open(LATEST_SNAPSHOT, "r") as f:
    latest_snapshot = json.load(f)
embedding_model = latest_snapshot["embedder"]["name"]
embedding_normalized = latest_snapshot["embedder"]["normalized"]

# Load data
chunks = pd.read_parquet(chunks_path)
faiss_index = faiss.read_index(str(faiss_index_path))
with open(mapping_path, "r") as f:
    mapping = {line["row_id"]: line for line in map(json.loads, f)}

@tool(parse_docstring=True)
@mcp.tool()
async def rag_search(query: str, k: int, filters: Union[dict, None] = None) -> str:
    """Search the FAISS index and return top-k results.

    Args:
        query (str): The search query string.
        k (int): The number of top results to return.
        filters (Union[dict, None], optional): Additional filters for the search. Defaults to None.

    Returns:
        str: A JSON string containing the search results, including hits, snapshot ID, and retriever metadata.
    """
    try:
        # Embed query
        model = SentenceTransformer(embedding_model)
        query_vector = model.encode(query, normalize_embeddings=embedding_normalized)

        # Convert query vector to NumPy array
        query_vector = np.array(query_vector)

        # Reshape query vector to 2D array
        query_vector = query_vector.reshape(1, -1)

        # Search FAISS index
        distances, indices = faiss_index.search(query_vector, k)

        # Map results
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Extract the correct row index from the mapping
            row_index = mapping[idx]["row_id"]
            row = chunks.iloc[row_index]
            hits.append({
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "doc_sha": row["doc_sha"],
                "file_name": row["file_name"],
                "page": int(row["page"]),  # Convert np.int64 to int
                "score": float(dist),
            })

        return json.dumps({
            "hits": hits,
            "snapshot_id": SNAPSHOT_ID,
            "retriever": {"k": k, "rerank": config["retriever"]["rerank"]},
        })

    except Exception as e:
        return json.dumps({"error": str(e)})

@tool(parse_docstring=True)
@mcp.tool()
async def rag_citations(chunk_ids: list) -> str:
    """Retrieve citation metadata for given chunk IDs.

    Args:
        chunk_ids (list): A list of chunk IDs for which to retrieve citation metadata.

    Returns:
        str: A JSON string containing the citation metadata, including chunk ID, URI, and page number.
    """
    citations = []
    for chunk_id in chunk_ids:
        row = chunks[chunks["chunk_id"] == chunk_id].iloc[0]
        citations.append({
            "chunk_id": chunk_id,
            "file_name": row["file_name"],
            "page": int(row["page"]),  # Convert np.int64 to int
        })
    return json.dumps({"citations": citations})

if __name__ == "__main__":
    mcp.run()
