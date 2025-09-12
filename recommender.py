# recommender.py
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec

DEFAULT_FILTERS = {
    "min_stars": 1, "min_rating": 5.0, "min_comments": 0,
    "min_service": 0.0, "min_location": 0.0,
    "near_beach": False, "business": False, "spa": False, "family": False,
}

def preprocess_query(text: str) -> list[str]:
    if not text: return []
    q = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    q = re.sub(r"\s+", " ", q)
    return q.split()

def _row_to_dict(row, sim):
    return {
        "hotel_id": row["Hotel_ID"],
        "name": row["Hotel_Name"],
        "similarity": float(sim),
        "stars": row["Hotel_Rank_numeric"],
        "total": row["Total_Score_clean"],
        "loc": row["Location_clean"],
        "clean": row["Cleanliness_clean"],
        "serv": row["Service_clean"],
        "fac": row["Facilities_clean"],
        "value": row["Value_for_money_clean"],
        "comfort": row["Comfort_and_room_quality_clean"],
        "comments": int(row["comments_count"]) if "comments_count" in row else 0,
        "addr": row["Hotel_Address"],
        "desc": row.get("Hotel_Description", ""),
    }

def search_by_query_doc2vec(query_text: str, model: Doc2Vec, hotels_df: pd.DataFrame, top_k=20):
    tokens = preprocess_query(query_text)
    if not tokens: return []
    qv = model.infer_vector(tokens)
    # Giả định doc tags là '0','1',... theo index của hotels_df
    dv = np.array([model.dv[str(i)] for i in range(len(hotels_df))])
    sim = cosine_similarity([qv], dv).ravel()
    top_idx = np.argsort(sim)[::-1][:top_k]
    return [_row_to_dict(hotels_df.iloc[i], sim[i]) for i in top_idx]

def similar_by_hotel_doc2vec(hotel_id, hotels_df, sim_matrix, id_mapping, top_k=20):
    key = str(hotel_id)
    if key not in id_mapping: return []
    idx = id_mapping[key]
    scores = sim_matrix[idx]
    order = np.argsort(scores)[::-1]
    order = order[order != idx][:top_k]  # bỏ chính nó
    return [_row_to_dict(hotels_df.iloc[i], scores[i]) for i in order]

def apply_filters(items: list[dict], f: dict):
    if not items: return []
    out = []
    for r in items:
        if r["stars"] < f["min_stars"]: continue
        if r["total"] < f["min_rating"]: continue
        if r["comments"] < f["min_comments"]: continue
        if f["min_service"] and r["serv"] < f["min_service"]: continue
        if f["min_location"] and r["loc"] < f["min_location"]: continue
        desc = (r.get("desc") or "").lower()
        if f["near_beach"] and not any(w in desc for w in ["beach","biển","sea","ocean"]): continue
        if f["business"] and not any(w in desc for w in ["business","meeting","conference"]): continue
        if f["spa"] and not any(w in desc for w in ["spa","massage","wellness","relax"]): continue
        if f["family"] and not any(w in desc for w in ["family","kids","children","playground"]): continue
        out.append(r)
    return out

def count_active_filters(f: dict) -> int:
    b = DEFAULT_FILTERS
    return sum([
        f["min_stars"] > b["min_stars"],
        f["min_rating"] > b["min_rating"],
        f["min_comments"] > b["min_comments"],
        f["min_service"] > b["min_service"],
        f["min_location"] > b["min_location"],
        f["near_beach"], f["business"], f["spa"], f["family"]
    ])
