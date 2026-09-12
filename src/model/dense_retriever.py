
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.conifgs.path_configs import CACHE_DIR, DATA_DIR, DENSE_INDEXES_DIR
from src.conifgs.ai_configs import dense_retreiver_model

DENSE_INDEXES_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(dense_retreiver_model, cache_folder=CACHE_DIR, local_files_only=True, device="cuda")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts and return a (len(texts), 384) array."""
    embeddings = model.encode(texts)
    return embeddings


def build_index(doc_texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed the full corpus in batches with a progress bar."""
    chunks = []
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding"):
        chunks.append(embed_batch(doc_texts[i : i + batch_size]))
    return np.vstack(chunks)  # stack batches into one (N, 384) matrix


# --------------------------------------------------------------
# Step 2: Build or load the cached embedding matrix
# --------------------------------------------------------------

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].tolist()

# embedding error because of empty strings in the embeddings endpoint. ~38 FiQA docs have
# blank text; we swap in a placeholder so the row order stays aligned with
# the BM25 index (which tolerates empty text just fine).
doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]

embeddings_path = DENSE_INDEXES_DIR / "embeddings.npy"
if embeddings_path.exists():
    print(f"Loading cached embeddings from {embeddings_path}")
    doc_embeddings = np.load(embeddings_path)
else:
    print(f"Embedding {len(doc_texts)} docs ")
    doc_embeddings = build_index(doc_texts)
    np.save(embeddings_path, doc_embeddings)

# Pre-normalize once so cosine similarity becomes a single dot product later.
doc_embeddings_normed = doc_embeddings / np.linalg.norm(
    doc_embeddings, axis=1, keepdims=True
)

# --------------------------------------------------------------
# Step 3: Query by cosine similarity
# --------------------------------------------------------------


def search_dense(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k (doc_id, similarity) pairs for a query."""
    query_vec = embed_batch([query])[0]
    query_vec /= np.linalg.norm(query_vec)
    scores = doc_embeddings_normed @ query_vec
    top_k = np.argsort(-scores)[:k]
    return [(doc_ids[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")
    for i, (doc_id, score) in enumerate(search_dense(query, k=5), 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}. [{score:.3f}] {doc_id} {text}\n")