from __future__ import annotations
import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import json
import yaml
from typing import Union, List, Dict, Any, Optional


def hash_pdf_files(source_directory: Path, output_directory: Path = Path("data")) -> List[Dict[str, Any]]:
    """
    Process all PDF files in a directory using content-addressable storage.
    
    For each PDF file:
    - Compute doc_sha = sha256(raw_bytes)
    - Write raw bytes to data/raw/sha256/<h[:2]>/<h[2:4]>/<doc_sha> if not already present
    - Record (file_path, doc_sha, file_size, mtime) for the manifest
    
    Args:
        source_directory: Directory containing PDF files to process
        data_root: Root data directory (default: "data")
        
    Returns:
        List of manifest entries with file metadata
    """
    source_dir = Path(source_directory)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create base directory structure
    raw_base = output_directory / "raw" / "sha256"
    raw_base.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    processed_count = 0
    skipped_count = 0
    
    # Find all PDF files
    pdf_files = list(source_dir.glob("**/*.pdf")) + list(source_dir.glob("**/*.PDF"))
    
    print(f"📁 Processing {len(pdf_files)} PDF files from {source_dir}")
    
    for pdf_file in pdf_files:
        try:
            # Read file and compute hash
            with open(pdf_file, 'rb') as f:
                raw_bytes = f.read()
            
            doc_sha = hashlib.sha256(raw_bytes).hexdigest()
            file_size = len(raw_bytes)
            mtime = pdf_file.stat().st_mtime
            
            # Create storage path: data/raw/sha256/<first_2>/<next_2>/<full_hash>
            h = doc_sha
            storage_path = raw_base / h[:2] / h[2:4] / doc_sha
            
            # Check if file already exists
            if storage_path.exists():
                print(f"⏭️  Skipping {pdf_file.name} (already stored as {doc_sha[:8]}...)")
                skipped_count += 1
            else:
                # Create directory structure and write file
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                with open(storage_path, 'wb') as f:
                    f.write(raw_bytes)
                print(f"✅ Stored {pdf_file.name} as {doc_sha[:8]}... ({file_size:,} bytes)")
                processed_count += 1
            
            # Add to manifest
            manifest.append({
                "file_path": str(pdf_file),
                "doc_sha": doc_sha,
                "file_size": file_size,
                "mtime": mtime,
                "storage_path": str(storage_path)
            })
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            continue
    
    print(f"📊 Processing complete: {processed_count} stored, {skipped_count} skipped")
    return manifest


def _canonical_serialize(obj: Any) -> str:
    """Serialize object to canonical JSON string with sorted keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load a single YAML config file, returning empty dict on any error."""
    try:
        if config_path.exists() and yaml is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _load_effective_configs(root_dir: Path) -> str:
    """Load all config files and return canonical serialized string."""
    config_names = ["chunker", "embedder", "index"]
    effective_configs = {
        name: _load_config_file(root_dir / "config" / f"{name}.yaml")
        for name in config_names
    }
    return _canonical_serialize(effective_configs)


def _compute_corpus_digest(doc_records: List[Dict[str, Any]]) -> str:
    """Compute deterministic digest from document records."""
    items = []
    for entry in doc_records:
        doc_sha = entry.get("doc_sha")
        # Use only doc_sha for corpus digest, not storage_path which can vary
        if doc_sha:
            items.append(doc_sha)
    
    items_sorted = sorted(items)
    hasher = hashlib.sha256()
    for doc_sha in items_sorted:
        hasher.update(doc_sha.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _compute_fingerprint(corpus_digest: str, config_digest: str) -> str:
    """Combine corpus and config digests into a single fingerprint."""
    hasher = hashlib.sha256()
    hasher.update(corpus_digest.encode("utf-8"))
    hasher.update(b"-")
    hasher.update(config_digest.encode("utf-8"))
    return hasher.hexdigest()


def _get_last_fingerprint(root_dir: Path) -> Optional[str]:
    """Get the last saved fingerprint from snapshot, if any."""
    latest_file = root_dir / "data" / "LATEST"
    print(f"🔍 Looking for LATEST file at: {latest_file}")
    
    if not latest_file.exists():
        print(f"❌ LATEST file does not exist")
        return None
    
    try:
        latest_name = latest_file.read_text(encoding="utf-8").strip()
        print(f"📄 LATEST file contents: '{latest_name}'")
        
        snapshot_path = root_dir / "data" / "snapshots" / f"{latest_name}.json"
        print(f"🔍 Looking for snapshot at: {snapshot_path}")
        
        if snapshot_path.exists():
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            fingerprint = snap.get("identity", {}).get("fingerprint")
            print(f"✅ Found saved fingerprint: {fingerprint}")
            return fingerprint
        else:
            print(f"❌ Snapshot file does not exist: {snapshot_path}")
    except Exception as e:
        print(f"❌ Error reading fingerprint: {e}")
    return None


def get_git_commit_hash() -> str:
    """Retrieve the current Git commit hash."""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return git_sha[:8]  # Use short hash
    except Exception as e:
        print(f"❌ Error retrieving Git commit hash: {e}")
        return "no_git"


def create_snapshot_id(fingerprint: str) -> str:
    """
    Create a unique snapshot identifier.
    
    Format: SNAP_YYYYMMDDThhmmss_git_sha[:7]_fingerprint[:8]

    Parameters:
      - fingerprint: the computed fingerprint string

    Returns:
      - snapshot_id: formatted snapshot identifier
    """ 
    # Get current UTC timestamp
    utc_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    
    # Get git commit hash
    git_sha = get_git_commit_hash()
    
    # Create snapshot ID
    snapshot_id = f"SNAP_{utc_timestamp}_{git_sha}_{fingerprint[:8]}"
    return snapshot_id


def config_change(root_dir: Path, doc_records: List[Dict[str, Any]], force: bool = False) -> Union[bool, Tuple[str, str, str]]:
    """
    Compute a fingerprint for the current corpus + effective configs.

    Parameters:
      - root_dir: project root directory
      - doc_records: list of document records with structure like
          {"file_path": ..., "doc_sha": ..., "file_size": ..., "mtime": ..., "storage_path": ...}
      - force: when True always returns fingerprint rather than comparing against LATEST

    Returns:
      - False if no change detected (and force is False)
      - (fingerprint, corpus_digest, config_digest) tuple when a change is detected or force=True
    """
    # Compute digests
    config_digest = _load_effective_configs(root_dir)
    corpus_digest = _compute_corpus_digest(doc_records)
    fingerprint = _compute_fingerprint(corpus_digest, config_digest)

    # Debug output
    print(f"🔍 Current fingerprint: {fingerprint}")
    print(f"📊 Corpus digest: {corpus_digest}")
    print(f"⚙️ Config digest: {config_digest}")

    # Compare with last snapshot unless forced
    if not force:
        last_fingerprint = _get_last_fingerprint(root_dir)
        print(f"📋 Last fingerprint: {last_fingerprint}")
        
        if last_fingerprint == fingerprint:
            print("✅ No changes detected; skipping pipeline")
            return False
        else:
            print("🔄 Changes detected - proceeding with pipeline")

    return (fingerprint, corpus_digest, config_digest)
