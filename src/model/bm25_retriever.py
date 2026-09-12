import bm25s
import pandas as pd

from src.conifgs.path_configs import DATA_DIR, BM25_INDEXES_DIR


# LOADING THE CORPUS
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].tolist()
doc_texts = corpus["text"].tolist()

print(f"Indexing {len(doc_texts)} documents with BM25...")


# indexing using bm25
tokens = bm25s.tokenize(doc_texts, stopwords="en")

print(tokens.ids[:1])  # list[list[int]] -- one inner list per doc
print(list(tokens.vocab.items())[:10])  # dict[str, int] -- token string -> integer ID

retriever = bm25s.BM25()  # method='lucene' by default
retriever.index(tokens)


# Save the index plus the doc_ids in matching order, so we can map back later.
BM25_INDEXES_DIR.mkdir(parents=True, exist_ok=True)
retriever.save(str(BM25_INDEXES_DIR))
(BM25_INDEXES_DIR / "doc_ids.txt").write_text("\n".join(doc_ids))


def search_bm25(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k (doc_id, score) pairs for a query."""
    query_tokens = bm25s.tokenize([query], stopwords="en")
    indices, scores = retriever.retrieve(query_tokens, k=k)
    # indices[0] is a numpy array of integer positions in doc_ids.
    return [
        (doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())
    ]


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")
    for i, (doc_id, score) in enumerate(search_bm25(query, k=5), 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}. [{score:6.2f}] {doc_id}  {text[:80]}")