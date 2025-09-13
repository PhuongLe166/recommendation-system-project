import streamlit as st
import pandas as pd
import os, json, base64

def render_evaluation():
    st.header("Evaluation & Report")

    # ==============================
    # CSS for image cards
    # ==============================
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

    # ==============================
    # Load Content-Based metrics
    # ==============================
    try:
        with open("hotel_ui_data/performance_metrics.json", "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        st.error(f"Unable to load performance_metrics.json: {e}")
        return

    methods = metrics_data.get("methods", {})

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

    # ==============================
    # Content-Based Evaluation
    # ==============================
    st.subheader("Content-Based Model Comparison")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Interpretation (Content-Based Models):**
        - **Doc2Vec** achieves the highest precision and MAP, confirming stronger ranking quality, but requires longer build time.  
        - **TF-IDF** remains competitive while being lightweight and efficient.  
        - **LSI (Gensim)** offers balance but does not outperform Doc2Vec in ranking relevance.  
        """
    )

    # ==============================
    # Visualization Cards Helper
    # ==============================
    img_dir = "hotel_ui_data/images"

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

    def _render_image_section(title, images, n_cols=3):
        st.subheader(title)
        cols = st.columns(n_cols)
        idx = 0
        for filename, caption in images:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                html = _card_from_path(path, caption)
                if html:
                    with cols[idx % n_cols]:
                        st.markdown(html, unsafe_allow_html=True)
                    idx += 1

    # --- Charts: Content-Based Comparison ---
    selected_images = [
        ("methods_precision.png", "Precision@K Comparison"),
        ("methods_quality.png", "MAP & Correlation"),
        ("methods_radar.png", "Overall Performance Radar")
    ]
    _render_image_section("Illustration Charts (Content-Based Models)", selected_images)

    st.markdown(
        """
        **Insights from Charts:**  
        - **Precision@K:** All three methods perform well, Doc2Vec slightly ahead → stronger top-K recommendation quality.  
        - **MAP & Correlation:** Doc2Vec gives the highest MAP, showing better ranking of relevant hotels.  
        - **Radar Chart:** Doc2Vec leads in accuracy, TF-IDF is lightweight/fast, LSI balances interpretability.  
        """
    )

    # --- Similarity Distributions ---
    sim_images = [
        ("tfidf_similarity_distribution.png", "TF-IDF Similarity Distribution"),
        ("gensim_similarity_distribution.png", "LSI (Gensim) Similarity Distribution"),
        ("doc2vec_similarity_distribution.png", "Doc2Vec Similarity Distribution"),
    ]
    _render_image_section("Similarity Distributions by Model", sim_images)

    st.markdown(
        """
        **Insights from Distributions:**  
        - **TF-IDF:** Narrow distribution, struggles to separate similar vs dissimilar hotels.  
        - **LSI (Gensim):** Broader distribution, better clustering of hotels by topics.  
        - **Doc2Vec:** Balanced bell-shaped distribution with mean ≈ 0.4 → strongest semantic separation.  
        """
    )

    # --- Wordclouds ---
    wc_images = [
        ("tfidf_wordcloud.png", "TF-IDF Wordcloud"),
        ("gensim_wordcloud.png", "LSI (Gensim) Wordcloud"),
        ("doc2vec_wordcloud.png", "Doc2Vec Wordcloud"),
    ]
    _render_image_section("Wordclouds by Model", wc_images)

    st.markdown(
        """
        **Insights from Wordclouds:**  
        - **TF-IDF:** Focuses on frequent tokens (*“biển”, “new property”*, etc.) → surface-level representation.  
        - **LSI (Gensim):** Highlights latent topics (*“dịch vụ”, “khách hàng”, “trải nghiệm”*) → better topic grouping.  
        - **Doc2Vec:** Captures richer semantics (*“tận hưởng”, “tuyệt vời”, “gần biển”*) → reflects customer intent best.  
        """
    )

    # ==============================
    # ALS Collaborative Filtering
    # ==============================
    st.subheader("Collaborative Filtering (ALS) Evaluation")

    st.markdown(
        """
        | Metric              | Value (approx.) | Interpretation |
        |---------------------|-----------------|----------------|
        | **Mean Prediction** | 8.72            | Predictions mostly around 9, limited variance. |
        | **Std. Dev. Pred.** | 0.81            | Concentrated predictions, weak separation. |
        | **Mean Residual**   | 0.49            | On average, ~0.5 points lower than actual ratings. |
        | **Std. Dev. Resid.**| 1.18            | Errors typically spread ±1 point. |
        | **Residual Min/Max**| -3.70 / +5.06   | Some cases poorly predicted (>3 points error). |
        | **RMSE**            | > 1.0           | Moderate error on a 10-point scale. |
        """
    )

    st.markdown(
        """
        **Insights (ALS):**  
        - Predictions cluster near 9 → tendency to **overestimate ratings**.  
        - RMSE > 1 means the model is not highly accurate in distinguishing hotels.  
        - Bias toward **popular hotels**, niche preferences are underrepresented.  
        - Despite limitations, ALS provides **collaborative personalization** that complements content-based methods.  
        """
    )

    # --- ALS Diagnostic Charts ---
    als_images = [
        ("als_prediction_summary.png", "ALS Prediction Distribution"),
        ("als_residual_summary.png", "ALS Residuals Distribution"),
    ]
    _render_image_section("ALS Diagnostic Charts", als_images, n_cols=2)

    # ==============================
    # Final Summary
    # ==============================
    st.subheader("Overall Evaluation Summary")
    st.markdown(
        """
        - **Content-Based Models:**  
          - Doc2Vec delivers the most semantically accurate recommendations.  
          - TF-IDF provides a fast, lightweight option.  
          - LSI balances topic interpretability and efficiency.  

        - **Collaborative Filtering (ALS):**  
          - Adds personalization from user–item interactions.  
          - Prediction errors are moderate (RMSE > 1), with bias towards popular hotels.  
          - Works best when combined with content-based methods.  

        **Recommendation for Agoda:**  
        - Use **Doc2Vec** for semantic search and content-based personalization.  
        - Use **ALS** to incorporate collaborative signals from historical ratings.  
        - Deploy a **hybrid recommender (Doc2Vec + ALS)** to maximize personalization and accuracy.  
        """
    )
