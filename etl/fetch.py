from pathlib import Path
import requests
from urllib.parse import urlparse

def download(name: str, url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    fname = Path(parsed.path).name or f"{name}.pdf"
    if "." not in fname:
        fname += ".pdf"
    out = dest_dir / fname
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out