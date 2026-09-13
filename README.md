# RAG Retrieval Evaluation Harness

A benchmarking harness for retrieval pipelines that compares four strategies — BM25, dense, hybrid (Reciprocal Rank Fusion), and hybrid + neural reranking — on the **FiQA-2018** financial QA dataset. The metric is **NDCG@10**, the standard BEIR benchmark measure.

The goal is to show that each stage of the pipeline earns a measurable improvement over the previous one.

## Pipeline Overview

```
FiQA corpus (~57k financial docs)
         │
         ├──► BM25Retriever        sparse keyword matching (bm25s, Lucene variant)
         │
         └──► DenseRetriever       semantic search (intfloat/multilingual-e5-small)
                    │
                    └──► Hybrid (RRF)    Reciprocal Rank Fusion of BM25 + Dense
                                │
                                └──► Reranker    cross-encoder scoring (BAAI/bge-reranker-v2-m3)
                                                         │
                                                         ▼
                                                    NDCG@10 scores
```

At query time, BM25 and dense retrieval each return 50 candidates. Those are fused via RRF into a unified candidate pool. The reranker then scores every `[query, document]` pair with a cross-encoder and returns the final top-10.

## Expected Results (FiQA-2018 NDCG@10)

| Method | NDCG@10 |
|--------|---------|
| BM25 | ~24 |
| Dense (multilingual-e5-small) | ~31 |
| Hybrid (RRF) | ~33 |
| Hybrid + Rerank (bge-reranker-v2-m3) | ~40+ |

Public BEIR baselines: [BEIR Leaderboard](https://github.com/beir-cellar/beir/wiki/Leaderboard)

## Project Structure

```
src/
├── conifgs/                  # Configuration
│   ├── ai_configs.py         # Model names for dense retriever and reranker
│   └── path_configs.py       # All filesystem paths (data, indexes, cache)
├── data/
│   ├── downloading_fiqa.py   # One-time dataset download from HuggingFace
│   ├── exploring_data.py     # EDA and test query extraction
│   └── fiqa/                 # Parquet files: corpus, queries, qrels, test_queries
├── model/
│   ├── bm25_retriever.py     # Sparse BM25 retriever (index build + search)
│   ├── dense_retriever.py    # Dense retriever (embedding build + cosine search)
│   ├── fusion.py             # Reciprocal Rank Fusion (hybrid candidates)
│   ├── reranker.py           # Cross-encoder reranker (full pipeline convenience fn)
│   ├── bm25_indexes/         # Persisted BM25 index files
│   └── dense_indexes/        # Persisted corpus embeddings (.npy)
├── model_cache/              # HuggingFace model weights on disk
├── utils/
│   └── corpus.py             # load_corpus() helper (reads corpus.parquet)
└── evaluate.py               # Main entry point — runs the benchmark
```

## Components

### BM25 Retriever

Uses [`bm25s`](https://github.com/xhluca/bm25s) (fast, pure-Python BM25) with the Lucene BM25 variant and English stopword removal. The index is built once and saved to `model/bm25_indexes/`. Subsequent runs load directly from disk.

To rebuild the index from scratch, instantiate with `BM25Retriever(initialize_corpus=True)`.

### Dense Retriever

Uses `intfloat/multilingual-e5-small` (384-dim multilingual sentence embeddings) via `sentence-transformers`. Corpus embeddings are built in batches of 256 on GPU, L2-normalized, and saved to `model/dense_indexes/embeddings.npy`. Search is brute-force cosine similarity (dot product on normalized vectors) — adequate for ~57k documents.

Embeddings are built automatically on first run; subsequent runs load from cache.

### Hybrid Fusion (RRF)

Combines BM25 and dense rankings using **Reciprocal Rank Fusion** with `k=60`. Each document at rank `r` in a ranking gets score `1.0 / (60 + r)`. Scores are summed across both rankings. Documents that appear highly ranked in both retrievers naturally accumulate higher fused scores.

### Reranker

Uses `BAAI/bge-reranker-v2-m3`, a multilingual cross-encoder loaded via HuggingFace `transformers`. It scores each `[query, document]` pair as a single sequence-classification logit. The reranker takes the top-50 hybrid candidates and re-scores all 50 pairs to produce the final top-10.

## Requirements

- Python >= 3.10
- CUDA-capable GPU (both the dense retriever and reranker are hard-coded to `device="cuda"`)
- [`uv`](https://docs.astral.sh/uv/) package manager

## Installation

```bash
uv sync
```

PyTorch is sourced from the PyTorch CUDA 12.4 index automatically via `pyproject.toml`.

## Running

### Step 1 — Download the FiQA dataset (one-time)

```bash
uv run python src/data/downloading_fiqa.py
```

Downloads corpus, queries, and qrels from HuggingFace Hub and saves them as Parquet files to `src/data/fiqa/`.

### Step 2 — Build the BM25 index (one-time)

Temporarily set `initialize_corpus=True` in `bm25_retriever.py`'s `__main__` block and run:

```bash
uv run python src/model/bm25_retriever.py
```

Saves the index to `src/model/bm25_indexes/`. Only needs to be done once.

### Step 3 — Run the evaluation

```bash
uv run python src/evaluate.py
```

On the first run, the dense retriever will embed the full corpus (~57k docs) and cache the embeddings. Subsequent runs skip straight to evaluation.

By default, the benchmark samples **50 queries** (controlled by `RERANK_SAMPLE_SIZE = 50` in `evaluate.py`) to keep the reranker runtime reasonable. Set it to `None` to evaluate all 648 test queries.

## Dataset

**FiQA-2018** (Financial Opinion Mining and Question Answering) is a [BEIR](https://github.com/beir-cellar/beir) benchmark dataset for financial QA retrieval. It contains ~57k documents, ~6k queries, and relevance judgments for 648 test queries.

Downloaded from HuggingFace Hub: `BeIR/fiqa` and `BeIR/fiqa-qrels`.

## Evaluation Metric

**NDCG@10** (Normalized Discounted Cumulative Gain at 10) rewards putting relevant documents high in the top-10 ranking and penalizes putting them low. A perfect ranking scores 1.0.

```
DCG  = Σ rel(d) / log2(rank + 2)   for top-10 predicted docs
IDCG = DCG of the ideal ranking
NDCG = DCG / IDCG
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `bm25s` | Fast BM25 sparse retrieval |
| `sentence-transformers` | Dense embedding model |
| `transformers` | Cross-encoder reranker |
| `torch` (cu124) | CUDA 12.4 GPU inference |
| `numpy` | Embedding matrix operations |
| `pandas` | Parquet data loading |
| `datasets` | HuggingFace dataset download |
| `langfuse` | Observability / tracing |
