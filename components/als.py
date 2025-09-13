# als.py — Streamlit UI for ALS (Light, polished)
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st

# ---- Base paths (đọc đúng cấu trúc bạn đã export) ----
ALS_BASE = Path("output/ui/als")
CSV_BASE = ALS_BASE / "csv"

RECS_DIR = CSV_BASE / "recs_scored_top10"
SUMMARY_DIR = CSV_BASE / "hotel_summary"
META_DIR = CSV_BASE / "hotel_meta"
ALL_USERS_DIR = CSV_BASE / "all_users"
ALL_USERS_JSON_DIR = ALS_BASE / "all_users_json"
KPIS_DIR = ALS_BASE / "kpis_json"  # Spark JSON lines folder

# ---- Helpers ----
def _read_spark_json_lines(folder: Path) -> List[Dict]:
    """
    Đọc folder JSON Lines do Spark ghi (part-*.json hoặc *.json).
    Trả về list các dict.
    """
    if not folder.exists():
        return []
    files = sorted(list(folder.glob("*.json"))) or sorted(folder.glob("part-*"))
    if not files:
        # đôi khi Spark lồng 1 cấp con
        files = sorted(folder.glob("*/part-*"))
        if not files:
            return []
    out = []
    with open(files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


@st.cache_data(show_spinner=False)
def load_kpis() -> Dict:
    rows = _read_spark_json_lines(KPIS_DIR)
    return rows[0] if rows else {}


@st.cache_data(show_spinner=False)
def load_all_users_df() -> pd.DataFrame:
    """
    Load the full user list with optional stats.
    Priority:
      1) JSON lines at output/ui/als/all_users_json (Spark-style)
      2) CSV at output/ui/als/csv/all_users
      3) Extract unique users from recs_scored_top10
    """
    # 1) JSON lines
    if ALL_USERS_JSON_DIR.exists():
        files = sorted(list(ALL_USERS_JSON_DIR.glob("*.json"))) or sorted(ALL_USERS_JSON_DIR.glob("part-*"))
        if files:
            rows = []
            with open(files[0], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        rows.append(obj)
                    except Exception:
                        pass
            if rows:
                df = pd.DataFrame(rows)
                if "userId" not in df.columns and len(df.columns) > 0:
                    # Best-effort: assume first column is userId
                    first_col = df.columns[0]
                    df = df.rename(columns={first_col: "userId"})
                if "userId" in df.columns:
                    df["userId"] = df["userId"].astype(str)
                    df = df.drop_duplicates(subset=["userId"]).reset_index(drop=True)
                    return df

    # 2) CSV fallback
    if ALL_USERS_DIR.exists():
        files = sorted(ALL_USERS_DIR.glob("*.csv"))
        if files:
            df = pd.read_csv(files[0])
            if "userId" not in df.columns:
                df.columns = ["userId"] + list(df.columns[1:])
            df["userId"] = df["userId"].astype(str)
            df = df.drop_duplicates(subset=["userId"]).reset_index(drop=True)
            return df

    # 3) From recs
    if RECS_DIR.exists():
        files = sorted(RECS_DIR.glob("*.csv"))
        if files:
            user_col = "userId"
            ids = pd.read_csv(files[0], usecols=[user_col], dtype=str)
            ids = ids[user_col].dropna().astype(str).unique().tolist()
            return pd.DataFrame({"userId": sorted(ids)})
    return pd.DataFrame(columns=["userId"])


@st.cache_data(show_spinner=True)
def load_recs_for_user(user_id: str, top_k: int = 10) -> pd.DataFrame:
    """
    Đọc recs_scored_top10 theo chunks để tránh ngốn RAM, filter theo userId.
    """
    files = sorted(RECS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    usecols = [
        "userId", "hotelID", "Hotel_Name",
        "Average_Rating", "Predicted_Rating",
        "Hybrid_Score", "Normalized_Hybrid_Score",
        "Rank", "Number_of_Reviewers",
    ]

    frames = []
    for chunk in pd.read_csv(files[0], chunksize=50_000, usecols=usecols, dtype=str):
        matched = chunk[chunk["userId"] == str(user_id)]
        if not matched.empty:
            frames.append(matched)
    if not frames:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(frames, ignore_index=True)
    # Cast types
    for c in ["Average_Rating", "Predicted_Rating", "Hybrid_Score", "Normalized_Hybrid_Score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Rank", "Number_of_Reviewers"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # Rank asc, Hybrid desc
    df = df.sort_values(["Rank", "Normalized_Hybrid_Score"], ascending=[True, False]).head(top_k)
    return df


@st.cache_data(show_spinner=False)
def load_hotel_summary() -> pd.DataFrame:
    files = sorted(SUMMARY_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[0])
    # Chuẩn tên cột cho dễ đọc
    rename = {
        "Hotel ID": "hotelID",
        "Hotel Name": "Hotel_Name",
        "Avg Normalized Hybrid Score": "Avg_Normalized_Hybrid_Score",
        "Number of Visitors": "Number_of_Visitors",
        "Times Recommended": "Times_Recommended",
    }
    for k, v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df


@st.cache_data(show_spinner=False)
def load_hotel_meta() -> pd.DataFrame:
    files = sorted(META_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[0])
    return df


def _inject_css():
    # No-op: remove ALS page CSS
    return


def _kpi_row(k: Dict, mode_label: str):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            try:
                st.metric("RMSE (ALS)", f"{float(k.get('rmse', 0)):.3f}")
            except Exception:
                st.metric("RMSE (ALS)", k.get("rmse", "-"))
    with col2:
        with st.container(border=True):
            try:
                st.metric("Sparsity", f"{float(k.get('sparsity', 0)):.3f}")
            except Exception:
                st.metric("Sparsity", k.get("sparsity", "-"))
    with col3:
        with st.container(border=True):
            st.metric("# Users", k.get("num_users", "-"))
    with col4:
        with st.container(border=True):
            st.metric("# Hotels", k.get("num_items", "-"))
    # Remove detailed ALS params from display as requested


def _inject_card_css_once():
    """Inject minimal CSS for ALS cards (once per session)."""
    key = "_als_card_css_injected"
    if st.session_state.get(key):
        return
    st.session_state[key] = True
    st.markdown(
        """
        <style>
        .als-card { background:#ffffff; border:1px solid #e9edf5; border-radius:14px; padding:14px; box-shadow:0 8px 22px rgba(17,24,39,.06); margin-bottom:12px; }
        .als-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
        .als-title { font-weight:800; color:#0f172a; }
        .als-badge { background: linear-gradient(135deg,#667eea,#764ba2); color:#fff; border-radius:999px; padding:.2rem .6rem; font-weight:700; font-size:.85rem; }
        .als-meta { color:#6b7280; font-size:13px; margin-bottom:8px; }
        .als-chips { display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
        .als-chip { display:inline-block; padding:.35rem .6rem; border-radius:10px; font-size:.85rem; border:1px solid #e9edf5; }
        .als-chip.good{ background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
        .als-chip.warn{ background:#fffbeb; color:#92400e; border-color:#fde68a; }
        .als-chip.info{ background:#eef2ff; color:#3730a3; border-color:#c7d2fe; }
        .als-bar{ background:#eef2ff; border-radius:999px; height:8px; overflow:hidden; }
        .als-fill{ background:linear-gradient(135deg,#667eea,#764ba2); height:100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_cards(df: pd.DataFrame):
    """Render ALS result cards with styled HTML/CSS while preserving original fields."""
    _inject_card_css_once()
    for _, r in df.iterrows():
        norm = float(r.get("Normalized_Hybrid_Score", 0.0))
        avg = float(r.get("Average_Rating", 0.0))
        pred = float(r.get("Predicted_Rating", 0.0))
        reviewers = int(r.get("Number_of_Reviewers", 0))
        title = str(r.get("Hotel_Name", "Hotel"))
        sid = str(r.get("hotelID", ""))
        pct = max(0.0, min(norm, 10.0)) * 10.0

        html = f"""
<div style="background:#ffffff;border:1px solid #e9edf5;border-radius:14px;padding:14px;box-shadow:0 8px 22px rgba(17,24,39,.06);margin-bottom:12px;min-height:160px;display:flex;flex-direction:column;gap:8px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div style="font-weight:800;color:#0f172a;max-width:78%;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.3;">{title}</div>
    <span style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-radius:999px;padding:.2rem .6rem;font-weight:700;font-size:.85rem;white-space:nowrap;">Norm: {norm:.2f}</span>
  </div>
  <div style="color:#6b7280;font-size:13px;">ID: {sid}</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="display:inline-block;padding:.35rem .6rem;border-radius:10px;font-size:.85rem;border:1px solid #e9edf5;background:#eef2ff;color:#3730a3;">Avg: {avg:.2f}</span>
    <span style="display:inline-block;padding:.35rem .6rem;border-radius:10px;font-size:.85rem;border:1px solid #a7f3d0;background:#ecfdf5;color:#065f46;">Pred: {pred:.2f}</span>
    <span style="display:inline-block;padding:.35rem .6rem;border-radius:10px;font-size:.85rem;border:1px solid #fde68a;background:#fffbeb;color:#92400e;">Reviewers: {reviewers}</span>
  </div>
  <div style="flex:1"></div>
  <div style="background:#eef2ff;border-radius:999px;height:8px;overflow:hidden;"><div style="width:{pct:.0f}%;background:linear-gradient(135deg,#667eea,#764ba2);height:100%;"></div></div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)


def _render_table(df: pd.DataFrame):
    show = df.copy()
    if "Normalized_Hybrid_Score" in show.columns:
        show["Hybrid %"] = show["Normalized_Hybrid_Score"].clip(lower=0, upper=10) * 10.0
    keep = [
        c for c in [
            "Rank",
            "Hotel_Name",
            "Average_Rating",
            "Predicted_Rating",
            "Hybrid %",
            "Number_of_Reviewers",
        ]
        if c in show.columns
    ]
    st.dataframe(show[keep], hide_index=True, use_container_width=True)


def render_als_ui():
    st.header("ALS Recommendations")

    tab1, tab2, tab3 = st.tabs(["ALS Recommendations", "Top Hotels", "Insights"])

    # === TAB 1: ALS Recs ===
    with tab1:
        kpis = load_kpis()
        # Toolbar (inline controls like original design)
        # Session defaults
        if "als_show_filters" not in st.session_state:
            st.session_state["als_show_filters"] = True
        if "als_min_avg" not in st.session_state:
            st.session_state["als_min_avg"] = 7.0
        if "als_min_rev" not in st.session_state:
            st.session_state["als_min_rev"] = 5
        if "als_view" not in st.session_state:
            st.session_state["als_view"] = "Cards"

        # Handle reset trigger before widgets are created (Streamlit restriction)
        if st.session_state.get("als_reset", False):
            st.session_state["als_min_avg"] = 7.0
            st.session_state["als_min_rev"] = 5
            st.session_state["als_view"] = "Cards"
            st.session_state["als_reset"] = False

        colA, colB, colC, colD = st.columns([2.8, 2.0, 1.4, 1.0], gap="medium")
        with colA:
            users_df = load_all_users_df()
            user_ids = users_df.get("userId", pd.Series(dtype=str)).astype(str).tolist()
            user_ids = sorted(set(user_ids))
            user_options = ["— Select user —"] + user_ids
            selected_user = st.selectbox("Select User ID (reviewerID)", user_options, index=0)
        with colB:
            rank_mode = st.radio("Ranking mode", ["Normalized Hybrid", "Predicted"], horizontal=True)
        with colC:
            top_choice = st.radio("Top K", ["Top 5", "Top 10"], index=1, horizontal=True)
        with colD:
            if st.button("Filters"):
                st.session_state["als_show_filters"] = not st.session_state["als_show_filters"]

        topn = 5 if top_choice == "Top 5" else 10
        user_id = "" if selected_user == "— Select user —" else str(selected_user).strip()

        # Filters
        # Filters row under toolbar
        if st.session_state["als_show_filters"]:
            f1, f2, f3, f4 = st.columns([1.6, 1.2, 1.4, 0.7], gap="medium")
            with f1:
                st.slider("Minimum average rating", 0.0, 10.0, step=0.1, key="als_min_avg")
            with f2:
                st.select_slider("Minimum reviewers", options=[0, 5, 10, 20, 50, 100], key="als_min_rev")
            with f3:
                st.radio("View mode", ["Cards", "Table"], horizontal=True, key="als_view")
            with f4:
                if st.button("Reset"):
                    # Set trigger and rerun so we reset BEFORE widgets instantiate
                    st.session_state["als_reset"] = True
                    st.rerun()

        # Read filters from session state
        min_avg = st.session_state["als_min_avg"]
        min_visitors = st.session_state["als_min_rev"]
        view_mode = st.session_state["als_view"]

        # KPI hàng trên
        mode_label = "mode: Hybrid" if rank_mode.startswith("Normalized") else "mode: Predicted"
        _kpi_row(kpis, mode_label)

        # Results
        results_df = pd.DataFrame()
        if user_id:
            with st.spinner("Fetching recommendations..."):
                results_df = load_recs_for_user(user_id, int(topn))

            # Thêm thông tin stars nếu có trong meta (hoặc dự phòng không có)
            # (Nếu bạn có file sao riêng, có thể merge ở đây)
            # Apply filters
            def _pass(row):
                ok = True
                try:
                    ok &= float(row.get("Average_Rating", 0)) >= float(min_avg)
                except Exception:
                    pass
                try:
                    ok &= int(row.get("Number_of_Reviewers", 0)) >= int(min_visitors)
                except Exception:
                    pass
                # min_stars bỏ qua nếu không có cột sao
                return ok

            if not results_df.empty:
                results_df = results_df[results_df.apply(_pass, axis=1)]

                # Sort by selected mode
                key = "Normalized_Hybrid_Score" if rank_mode.startswith("Normalized") else "Predicted_Rating"
                if key in results_df.columns:
                    results_df = results_df.sort_values(key, ascending=False)

                # Display
                st.subheader("Recommendations")
                if view_mode == "Cards":
                    _render_cards(results_df)
                else:
                    _render_table(results_df)

                st.download_button(
                    "Xuất CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"als_recs_{user_id}.csv",
                    mime="text/csv",
                    use_container_width=False,
                )
            else:
                st.info("No recommendations (user may not be in recs_scored_top10 or filters too strict).")
        elif not user_id:
            st.info("Please select a user to view recommendations.")

    # === TAB 2: Top Hotels ===
    with tab2:
        df = load_hotel_summary()
        if df.empty:
            st.info("No hotel_summary data yet.")
        else:
            st.markdown("#### Top Hotels Summary")
            show_cols = [
                "hotelID", "Hotel_Name",
                "Average Rating", "Predicted Rating",
                "Avg_Normalized_Hybrid_Score",
                "Number_of_Visitors", "Times_Recommended",
            ]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    # === TAB 3: Insights (KPIs) ===
    with tab3:
        st.markdown("#### KPIs (raw JSON)")
        st.json(load_kpis(), expanded=False)
    
    # End
