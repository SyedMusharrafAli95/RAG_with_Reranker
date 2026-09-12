import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.conifgs.ai_configs import dense_retreiver_model
from src.conifgs.path_configs import CACHE_DIR, DATA_DIR, DENSE_INDEXES_DIR

DENSE_INDEXES_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(
    dense_retreiver_model, cache_folder=CACHE_DIR, local_files_only=True, device="cuda"
)


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


class DenseRetriever:
    def __init__(self) -> None:
        corpus = load_corpus()
        self._doc_ids = corpus["_id"].tolist()
        self._doc_texts = [
            t.strip() or "[empty document]" for t in corpus["text"].tolist()
        ]

        self.model = SentenceTransformer(
            dense_retreiver_model,
            cache_folder=CACHE_DIR,
            local_files_only=True,
            device="cuda",
        )
        self._load_embeddings()

    def _build_index(self, doc_texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed the full corpus in batches with a progress bar."""
        chunks = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding"):
            chunks.append(self.model.encode(doc_texts[i : i + batch_size]))
        return np.vstack(chunks)  # stack batches into one (N, 384) matrix

    def _load_embeddings(self):
        embeddings_path = DENSE_INDEXES_DIR / "embeddings.npy"
        if embeddings_path.exists():
            print(f"Loading cached embeddings from {embeddings_path}")
            embeddings_raw = np.load(embeddings_path)
        else:
            print(f"Embedding {len(self._doc_texts)} docs ")
            embeddings_raw = self._build_index(self._doc_texts)
            np.save(embeddings_path, embeddings_raw)

        self._embeddings = embeddings_raw / np.linalg.norm(
            embeddings_raw, axis=1, keepdims=True
        )

    def _embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(query)
        vec = np.array(embedding, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]


# ------------------------------------------------------------

if __name__ == "__main__":
    dense_retriever = DenseRetriever()
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")

    print(dense_retriever.search(query))
