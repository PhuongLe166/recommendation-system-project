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

def _get_docvec(model: Doc2Vec, key):
    """Safely fetch a document vector by tag, accepting int or str tags."""
    try:
        return model.dv[key]
    except KeyError:
        return model.dv[str(key)]

def search_by_query_doc2vec(query_text: str, model: Doc2Vec, hotels_df: pd.DataFrame, id_mapping: dict, top_k=20):
    tokens = preprocess_query(query_text)
    if not tokens:
        return []

    # Deterministic inference by seeding RNG from the normalized query
    seed_source = " ".join(tokens)
    seed_value = abs(hash(seed_source)) % (2**32 - 1)
    rng_state = model.random.get_state()
    try:
        model.random.seed(seed_value)
        query_vector = model.infer_vector(tokens)
    finally:
        # Restore RNG so other parts of the app remain unaffected
        model.random.set_state(rng_state)

    # Build document matrix aligned to hotels_df rows using the provided id mapping
    doc_matrix = []
    valid_row_indices = []
    for row_index, row in hotels_df.iterrows():
        hotel_id = row.get("Hotel_ID")
        key = str(hotel_id)
        if key not in id_mapping:
            continue
        doc_index = id_mapping[key]
        doc_matrix.append(_get_docvec(model, doc_index))
        valid_row_indices.append(row_index)

    if not doc_matrix:
        return []

    doc_matrix = np.asarray(doc_matrix)
    similarities = cosine_similarity([query_vector], doc_matrix).ravel()

    # Stable ranking: sort by similarity desc, tie-break by Hotel_ID asc
    hotel_ids_for_tie = hotels_df.loc[valid_row_indices, "Hotel_ID"].to_numpy()
    order = np.lexsort((hotel_ids_for_tie, -similarities))  # ascending on -sim == desc on sim
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
    scores = sim_matrix[target_doc_index]

    # Build mapping from doc index -> row index for correct alignment
    doc_index_to_row_index = {}
    hotel_ids_in_doc_order = []
    for row_index, row in hotels_df.iterrows():
        hid = str(row.get("Hotel_ID"))
        if hid in id_mapping:
            di = id_mapping[hid]
            doc_index_to_row_index[di] = row_index
    # Prepare hotel ids array aligned to scores indices for tie-breaks; fill with large values if missing
    max_hid = int(1e12)
    hotel_ids_for_tie = np.full_like(scores, fill_value=max_hid, dtype=np.int64)
    for di, row_index in doc_index_to_row_index.items():
        try:
            hotel_ids_for_tie[di] = int(hotels_df.iloc[row_index]["Hotel_ID"])
        except Exception:
            # Fallback keeps max value, ensuring missing ones go last on ties
            pass

    # Exclude self, sort by score desc with stable tie-break on Hotel_ID asc
    mask_not_self = np.ones_like(scores, dtype=bool)
    mask_not_self[target_doc_index] = False
    effective_scores = scores.copy()
    effective_scores[~mask_not_self] = -np.inf
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
