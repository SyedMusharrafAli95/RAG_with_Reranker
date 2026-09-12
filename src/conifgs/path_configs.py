from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data" / "fiqa"
BM25_INDEXES_DIR = ROOT_DIR / "model" / "bm25_indexes"
DENSE_INDEXES_DIR = ROOT_DIR / "model" / "dense_indexes"
CACHE_DIR = ROOT_DIR / "model_cache"
