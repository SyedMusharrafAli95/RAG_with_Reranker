from collections import defaultdict

from src.model.bm25_retriever import BM25Retriever
from src.model.dense_retriever import DenseRetriever
from src.utils.corpus import load_corpus


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_candidates(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    bm25_ids = [doc_id for doc_id, _ in bm25.search(query, k=candidate_k)]
    dense_ids = [doc_id for doc_id, _ in dense.search(query, k=candidate_k)]
    return reciprocal_rank_fusion([bm25_ids, dense_ids])[:candidate_k]



def show(label: str, results: list[tuple[str, float]]) -> None:
    print(f"\n{label}")
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")

if __name__ == "__main__":
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    corpus = load_corpus()

    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    show("BM25 only", bm25.search(query, k=5))
    show("Dense only", dense.search(query, k=5))
    show("Hybrid (RRF)", hybrid_candidates(query, bm25, dense,  candidate_k=5))
