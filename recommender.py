# recommender.py
import re
import hashlib
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

def _get_docvec(model: Doc2Vec, key):
    """Safely fetch a document vector by tag, accepting int or str tags."""
    try:
        return model.dv[key]
    except KeyError:
        return model.dv[str(key)]

def _infer_vec(model: Doc2Vec, tokens: list[str], epochs: int = 80, alpha: float = 0.025, seed_value: int | None = None):
    """Deterministic infer_vector with temporary RNG seeding."""
    if seed_value is None:
        seed_value = 0
    rng_state = model.random.get_state()
    try:
        model.random.seed(seed_value)
        return model.infer_vector(tokens, epochs=epochs, alpha=alpha)
    finally:
        model.random.set_state(rng_state)

def _stable_seed_from_text(text: str) -> int:
    b = hashlib.sha256(text.encode("utf-8")).digest()[:4]
    return int.from_bytes(b, "little")

def _hotel_vector(model: Doc2Vec, row: pd.Series, doc_index_or_none):
    """Return doc vector by tag if available; otherwise infer from description."""
    if doc_index_or_none is not None:
        return _get_docvec(model, doc_index_or_none)
    desc_tokens = (row.get("Hotel_Description") or "").lower().split()
    # dùng seed ổn định theo hotel_id để fallback cũng deterministic
    seed_value = _stable_seed_from_text(str(row.get("Hotel_ID", "")))
    return _infer_vec(model, desc_tokens, epochs=40, alpha=0.025, seed_value=seed_value)

def search_by_query_doc2vec(query_text: str, model: Doc2Vec, hotels_df: pd.DataFrame, id_mapping: dict, top_k=20):
    tokens = preprocess_query(query_text)
    if not tokens:
        return []

    # Deterministic inference by stable seed from normalized query
    seed_source = " ".join(tokens)
    seed_value = _stable_seed_from_text(seed_source)
    query_vector = _infer_vec(model, tokens, epochs=80, alpha=0.025, seed_value=seed_value)

    # Build document matrix aligned to hotels_df rows, include ALL hotels
    doc_matrix = []
    valid_row_indices = []
    for row_index, row in hotels_df.iterrows():
        hid_key = str(row.get("Hotel_ID"))
        doc_index = id_mapping.get(hid_key) if hid_key in id_mapping else None
        vec = _hotel_vector(model, row, doc_index)  # fallback if missing tag
        doc_matrix.append(vec)
        valid_row_indices.append(row_index)

    if not doc_matrix:
        return []

    doc_matrix = np.asarray(doc_matrix)
    similarities = cosine_similarity([query_vector], doc_matrix).ravel()

    # Stable ranking: sort by similarity desc, tie-break by numeric Hotel_ID asc
    hid_series = pd.to_numeric(hotels_df.loc[valid_row_indices, "Hotel_ID"], errors="coerce")
    hotel_ids_for_tie = hid_series.fillna(10**12).astype(np.int64).to_numpy()
    order = np.lexsort((hotel_ids_for_tie, -similarities))  # primary: -sim, tie: Hotel_ID asc
    top_order = order[:top_k]

    results = []
    for pos in top_order:
        row_idx = valid_row_indices[int(pos)]
        sim_score = float(similarities[int(pos)])
        results.append(_row_to_dict(hotels_df.iloc[row_idx], sim_score))
    return results

def similar_by_hotel_doc2vec(hotel_id, hotels_df, sim_matrix, id_mapping, top_k=20):
    key = str(hotel_id)
    if key not in id_mapping:
        return []

    target_doc_index = id_mapping[key]
    scores = np.asarray(sim_matrix[target_doc_index]).ravel()  # ensure 1D

    # Build mapping from doc index -> row index for correct alignment
    doc_index_to_row_index = {}
    for row_index, row in hotels_df.iterrows():
        hid = str(row.get("Hotel_ID"))
        if hid in id_mapping:
            di = id_mapping[hid]
            doc_index_to_row_index[di] = row_index

    # Prepare hotel ids array aligned to 'scores' indices for tie-breaks
    max_hid = int(1e12)
    hotel_ids_for_tie = np.full_like(scores, fill_value=max_hid, dtype=np.int64)
    # vector hoá để lấy Hotel_ID dạng số nếu có
    # (vì cần theo doc index, đi từng phần tử)
    for di, row_index in doc_index_to_row_index.items():
        try:
            hid_val = pd.to_numeric(hotels_df.iloc[row_index]["Hotel_ID"], errors="coerce")
            hotel_ids_for_tie[di] = int(hid_val) if pd.notna(hid_val) else max_hid
        except Exception:
            pass

    # Exclude self, sort by score desc with stable tie-break on Hotel_ID asc
    effective_scores = scores.copy()
    effective_scores[target_doc_index] = -np.inf  # exclude self
    order = np.lexsort((hotel_ids_for_tie, -effective_scores))
    top_indices = [di for di in order if di != target_doc_index][:top_k]

    results = []
    for di in top_indices:
        if di in doc_index_to_row_index:
            row_idx = doc_index_to_row_index[di]
            results.append(_row_to_dict(hotels_df.iloc[row_idx], float(scores[di])))
    return results

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
