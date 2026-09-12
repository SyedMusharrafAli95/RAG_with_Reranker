from src.conifgs.path_configs import DATA_DIR
import pandas as pd


# loading and inspecting 3 parquet files

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
print(corpus["text"].iloc[0])

queries = pd.read_parquet(DATA_DIR / "queries.parquet")
print(queries["text"].iloc[0])

qrels = pd.read_parquet(DATA_DIR / "qrels.parquet")
print(qrels.iloc[0])

# filtering the data to only include test query
test_query_ids = set(qrels["query-id"].astype(str))
test_queries = queries[queries["_id"].astype(str).isin(test_query_ids)]
print(f"Queries (test, with qrels): {len(test_queries)}")

# inspecting one query and its relebvnat doc
query = test_queries.iloc[0]
print("\nExample query")
print(f"  _id:  {query['_id']}")
print(f"  text: {query['text']}")

relevant = qrels[qrels["query-id"].astype(str) == str(query["_id"])]
print(f"\n  Relevant docs for this question: {list(relevant['corpus-id'])}")

# storing test queries

test_queries.to_parquet(DATA_DIR / "test_queries.parquet")