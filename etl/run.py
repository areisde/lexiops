"""
ETL Pipeline for Lexiops Legal RAG System
"""
from .hash import hash_pdf_files, config_change, create_snapshot_id
from .embed import chunk_pdfs, save_chunks_table, create_embeddings_and_index
from .version import create_snapshot_manifest
from pathlib import Path
from typing import Union, Tuple

# Ensure script runs from project root (one level up from this etl/ directory)
ROOT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    print("🔗 Running ETL pipeline...")
    doc_records = hash_pdf_files(source_directory=ROOT_DIR / Path("data/raw"), output_directory=ROOT_DIR / Path("data"))
    
    change_result = config_change(root_dir=ROOT_DIR, doc_records=doc_records)

    print(f"🔍 Config change return: {change_result}")

    if type(change_result) is bool: # Return is False
        print("📌 No configuration changes detected. Pipeline complete.")
    else:
        fingerprint, corpus_digest, config_digest = change_result
        print(f"🔍 Config change detected: {fingerprint}")
        snapshot_id = create_snapshot_id(str(fingerprint))
        print(f"📸 Created new snapshot: {snapshot_id}")

        chunked_df = chunk_pdfs(doc_records, root_dir=ROOT_DIR)
        save_chunks_table(snapshot_id, chunked_df, root_dir=ROOT_DIR)
        print(f"📦 Created chunks: {len(chunked_df)}")

        create_embeddings_and_index(chunked_df, snapshot_id, root_dir=ROOT_DIR)
        print("🔢 Created embeddings and index")

        create_snapshot_manifest(
            snapshot_id=snapshot_id,
            fingerprint=str(fingerprint),
            corpus_digest=corpus_digest,
            config_digest=config_digest,
            git_sha="nogit",  # Placeholder if no git integration
            doc_records=doc_records,
            chunks_df=chunked_df,
            root_dir=ROOT_DIR
        )
        print("📊 Snapshot manifest created")

    print(f"📊 ETL pipeline complete: {len(doc_records)} documents processed.")