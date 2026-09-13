import bm25s

from src.conifgs.path_configs import BM25_INDEXES_DIR
from src.utils.corpus import load_corpus


class BM25Retriever:
    def __init__(self, initialize_corpus=False) -> None:
        corpus = load_corpus()
        self.doc_ids = corpus["_id"].tolist()
        self.doc_texts = corpus["text"].tolist()

        if initialize_corpus:
            self._initialize_corpus()
        self._initialize_inference()

    def _initialize_corpus(self) -> None:

        tokens = bm25s.tokenize(self.doc_texts, stopwords="en")
        retriever = bm25s.BM25()  # method='lucene' by default
        retriever.index(tokens)

        # Save the index plus the doc_ids in matching order, so we can map back later.
        BM25_INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        retriever.save(str(BM25_INDEXES_DIR))
        (BM25_INDEXES_DIR / "doc_ids.txt").write_text("\n".join(self.doc_ids))

    def _initialize_inference(self) -> None:

        self._retriever = bm25s.BM25.load(str(BM25_INDEXES_DIR))
        self._doc_ids = (BM25_INDEXES_DIR / "doc_ids.txt").read_text().splitlines()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(tokens, k=k)
        return [
            (self._doc_ids[i], float(scores[0][j]))
            for j, i in enumerate(indices[0].tolist())
        ]


if __name__ == "__main__":
    # LOADING THE CORPUS
    bm25_retriever = BM25Retriever(initialize_corpus=False)

    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")

    print(bm25_retriever.search(query))
