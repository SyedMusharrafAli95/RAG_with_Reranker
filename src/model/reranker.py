import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.conifgs.ai_configs import reranker_model
from src.conifgs.path_configs import CACHE_DIR
from src.model.bm25_retriever import BM25Retriever
from src.model.dense_retriever import DenseRetriever
from src.model.fusion import hybrid_candidates
from src.utils.corpus import load_corpus

bm25 = BM25Retriever()
dense = DenseRetriever()
corpus_by_id = load_corpus().set_index("_id")

tokenizer = AutoTokenizer.from_pretrained(reranker_model, cache_dir=CACHE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    reranker_model, cache_dir=CACHE_DIR
).to("cuda")
model.eval()

# pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
# with torch.no_grad():
#     inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
#     scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
#     print(scores)


def reranker_inference(pairs):
    score_idx = []
    for idx, pair in enumerate(pairs):
        with torch.no_grad():
            inputs = tokenizer(
                [pair],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to("cuda")
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            score_idx.append((idx, scores.item()))
    score_idx.sort(key=lambda x: x[1], reverse=True)
    return score_idx


def rerank_with_bgi(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:
    documents = [[query, corpus_by_id.loc[d, "text"]] for d in candidate_ids]
    response = reranker_inference(documents)
    return [(candidate_ids[r[0]], r[1]) for r in response]


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    return rerank_with_bgi(query, candidate_ids, corpus_by_id, k=k)


def show(label: str, results: list[tuple[str, float]]) -> None:
    print(f"\n{label}")
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = corpus_by_id.loc[doc_id, "text"]
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")


if __name__ == "__main__":
    query = "What should I buy as a stock"
    print(f"Query: {query}")

    # show("Hybrid (RRF) only", hybrid_candidates(query, bm25, dense, candidate_k=50)[:5])
    show(
        "Hybrid + bgi rerank",
        search_reranked(query, bm25, dense, corpus_by_id),
    )
