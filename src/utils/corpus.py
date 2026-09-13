import pandas as pd

from src.conifgs.path_configs import DATA_DIR


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "corpus.parquet")
