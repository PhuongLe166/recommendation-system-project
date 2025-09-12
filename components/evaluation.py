import streamlit as st
import pandas as pd
import os, json, base64

def render_evaluation():
    st.header("Evaluation & Report")

    # Minimal CSS to wrap each image in a bordered card with shadow
    st.markdown(
        """
        <style>
        :root { --card-img-h: 230px; }
        .img-card { background:#ffffff; border:1px solid #e9edf5; border-radius:12px; padding:10px; box-shadow:0 4px 14px rgba(17,24,39,0.08); }
        .img-card .img-wrap { height: var(--card-img-h); display:flex; align-items:center; justify-content:center; overflow:hidden; border-radius:8px; }
        .img-card img { width:100%; height:100%; object-fit: contain; display:block; }
        .img-card .img-cap { text-align:center; color:#6b7280; font-size:0.9rem; margin-top:6px; height:24px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _render_subtitle(title: str, icon: str | None = None, anchor: str | None = None):
        prefix = f"{icon} " if icon else ""
        st.subheader(prefix + title)

    # --- Load metrics ---
    try:
        with open("hotel_ui_data/performance_metrics.json", "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        st.error(f"Không thể load performance_metrics.json: {e}")
        return

    methods = metrics_data.get("methods", {})
    total_hotels = metrics_data.get("total_hotels", 0)
    generated_at = metrics_data.get("generated_at", "")

    # In this style, we focus on sections and cards; metrics can be added later if needed.

    # --- Bảng so sánh ---
    df_metrics = pd.DataFrame([
        {
            "Method": m.upper(),
            "Build Time (s)": round(v["build_time"], 2),
            "Precision@5": round(v["precision_5"], 4),
            "MAP": round(v["map_score"], 4),
            "Correlation": round(v["correlation"], 4),
        }
        for m, v in methods.items()
    ])
    st.subheader("Model comparison")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # --- Chọn lọc ảnh chính ---
    img_dir = "hotel_ui_data/images"
    selected_images = [
        ("methods_precision.png", "Precision@K Comparison"),
        ("methods_quality.png", "MAP & Correlation"),
        ("methods_radar.png", "Overall Performance Radar")
    ]

    if os.path.exists(img_dir):
        # Helper to render an image inside a styled card
        def _card_from_path(img_path: str, caption: str) -> str:
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return f"""
<div class=\"img-card\">
  <div class=\"img-wrap\"><img src=\"data:image/png;base64,{b64}\" /></div>
  <div class=\"img-cap\">{caption}</div>
</div>
"""
            except Exception:
                return ""

        # Section: Main charts
        st.subheader("Illustration charts")
        cols = st.columns(3)
        idx = 0
        for filename, caption in selected_images:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                html = _card_from_path(path, caption)
                if html:
                    with cols[idx % len(cols)]:
                        st.markdown(html, unsafe_allow_html=True)
                    idx += 1

        # --- Similarity distributions for 3 models ---
        def _show_card(img_path: str, caption: str):
            html = _card_from_path(img_path, caption)
            if html:
                st.markdown(html, unsafe_allow_html=True)

        sim_images = [
            ("tfidf_similarity_distribution.png", "TF-IDF Similarity Distribution"),
            ("gensim_similarity_distribution.png", "LSI (Gensim) Similarity Distribution"),
            ("doc2vec_similarity_distribution.png", "Doc2Vec Similarity Distribution"),
        ]
        st.subheader("Similarity distributions by model")
        cols2 = st.columns(3)
        i2 = 0
        for filename, caption in sim_images:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                with cols2[i2 % len(cols2)]:
                    _show_card(path, caption)
                i2 += 1

        # --- Wordclouds for 3 models ---
        wc_images = [
            ("tfidf_wordcloud.png", "TF-IDF Wordcloud"),
            ("gensim_wordcloud.png", "LSI (Gensim) Wordcloud"),
            ("doc2vec_wordcloud.png", "Doc2Vec Wordcloud"),
        ]
        st.subheader("Wordclouds by model")
        cols3 = st.columns(3)
        i3 = 0
        for filename, caption in wc_images:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                with cols3[i3 % len(cols3)]:
                    _show_card(path, caption)
                i3 += 1
    else:
        st.warning("⚠️ Images folder not found. Please re-run the plot export step.")
