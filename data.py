# data.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

DATA_DIR = Path("hotel_ui_data")

def _read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def load_hotels_df() -> pd.DataFrame:
    csv_p = DATA_DIR / "hotels.csv"
    json_p = DATA_DIR / "hotels.json"
    if csv_p.exists():
        return pd.read_csv(csv_p)
    if json_p.exists():
        return pd.read_json(json_p)
    raise FileNotFoundError("Thiếu hotels.csv / hotels.json trong hotel_ui_data/")

def load_id_mapping() -> dict:
    mp = _read_json(DATA_DIR / "hotel_id_mapping.json", {})
    return {str(k): int(v) for k, v in mp.items()}

def load_metrics() -> dict:
    return _read_json(DATA_DIR / "performance_metrics.json", {"methods": {"doc2vec": {}}})

def load_doc2vec_model() -> Doc2Vec:
    p = DATA_DIR / "doc2vec_model.model"
    if not p.exists():
        raise FileNotFoundError("Thiếu doc2vec_model.model trong hotel_ui_data/")
    return Doc2Vec.load(str(p))

def load_doc2vec_similarity() -> np.ndarray:
    p = DATA_DIR / "doc2vec_similarity.npy"
    if not p.exists():
        raise FileNotFoundError("Thiếu doc2vec_similarity.npy trong hotel_ui_data/")
    return np.load(p)
