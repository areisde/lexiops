from pathlib import Path
from .embed_index import build_index

if __name__ == "__main__":
    build_index(Path("data/chunks"), Path("data/index"))