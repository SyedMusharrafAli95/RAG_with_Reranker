import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.conifgs.ai_configs import reranker_model
from src.conifgs.path_configs import CACHE_DIR
from src.model.bm25_retriever import BM25Retriever
from src.model.dense_retriever import DenseRetriever
from src.model.fusion import hybrid_candidates
from src.utils.corpus import load_corpus


class Reranker:
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "cuda"

        self._load_reranker_model()

    def _load_reranker_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, cache_dir=CACHE_DIR
        ).to(self.device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def _reranker_inference(self, pairs):
        score_idx = []
        for idx, pair in enumerate(pairs):
            with torch.no_grad():
                inputs = self.tokenizer(
                    [pair],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to("cuda")
                scores = (
                    self.model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
                score_idx.append((idx, scores.item()))
        score_idx.sort(key=lambda x: x[1], reverse=True)
        return score_idx

    def _rerank_with_bgi(
        self,
        query: str,
        candidate_ids: list[str],
        corpus_by_id: pd.DataFrame,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        documents = [[query, corpus_by_id.loc[d, "text"]] for d in candidate_ids]
        response = self._reranker_inference(documents)[:k]
        return [(candidate_ids[r[0]], r[1]) for r in response]

    def search_reranked(
        self,
        query: str,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        corpus_by_id: pd.DataFrame,
        k: int = 10,
        candidate_k: int = 50,
    ) -> list[tuple[str, float]]:
        candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
        candidate_ids = [doc_id for doc_id, _ in candidates]
        ranked_candidates = self._rerank_with_bgi(
            query, candidate_ids, corpus_by_id, k=k
        )
        return self._get_documents(ranked_candidates, corpus_by_id)

    def _get_documents(
        self, results: list[tuple[str, float]], corpus_by_id: pd.DataFrame
    ) -> list[str]:
        retrieved_doc = []
        for i, (doc_id, score) in enumerate(results[:], 1):
            text = corpus_by_id.loc[doc_id, "text"]
            retrieved_doc.append((score, doc_id, text))

        return retrieved_doc


if __name__ == "__main__":
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    corpus_by_id = load_corpus().set_index("_id")
    reranker = Reranker(reranker_model, CACHE_DIR)

    query = "What should I buy as a stock"
    print(f"Query: {query}")

    # show("Hybrid (RRF) only", hybrid_candidates(query, bm25, dense, candidate_k=50)[:5])
    results = reranker.search_reranked(query, bm25, dense, corpus_by_id)
