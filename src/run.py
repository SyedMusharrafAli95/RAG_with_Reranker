from src.conifgs.ai_configs import reranker_model
from src.conifgs.path_configs import CACHE_DIR
from src.model.bm25_retriever import BM25Retriever
from src.model.dense_retriever import DenseRetriever
from src.model.reranker import Reranker
from src.utils.corpus import load_corpus

if __name__ == "__main__":
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    corpus_by_id = load_corpus().set_index("_id")
    reranker = Reranker(reranker_model, CACHE_DIR)

    query = "What should I buy as a stock"
    print(f"Query: {query}")

    # show("Hybrid (RRF) only", hybrid_candidates(query, bm25, dense, candidate_k=50)[:5])
    results = reranker.search_reranked(query, bm25, dense, corpus_by_id)
    for res in results:
        print(res)
