import streamlit as st
import pandas as pd
import os, json, base64

def render_evaluation():
    st.header("Evaluation & Report")

    # CSS for image cards
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

    # --- Load metrics ---
    try:
        with open("hotel_ui_data/performance_metrics.json", "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        st.error(f"Unable to load performance_metrics.json: {e}")
        return

    methods = metrics_data.get("methods", {})

    # --- Model comparison table ---
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

    st.subheader("Model Comparison")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Interpretation:**
        - **Doc2Vec** achieves the highest precision and MAP, indicating stronger ranking quality, though it requires significantly more build time.
        - **TF-IDF** provides competitive accuracy with much lower computation cost, making it efficient for quick experimentation.
        - **Gensim LSI** balances performance and efficiency but does not outperform Doc2Vec in ranking quality.
        """
    )

    # --- Visualization Cards Helper ---
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

    # --- Charts: Performance comparison ---
    selected_images = [
        ("methods_precision.png", "Precision@K Comparison"),
        ("methods_quality.png", "MAP & Correlation"),
        ("methods_radar.png", "Overall Performance Radar")
    ]
    _render_image_section("Illustration Charts", selected_images)

    st.markdown(
        """
        **Insights:**
        - The **Precision@K chart** shows that all three methods perform consistently, but Doc2Vec slightly edges ahead for top-K recommendations.  
        - The **MAP & Correlation chart** highlights Doc2Vec’s superior ability to rank relevant hotels higher while maintaining stable correlation with ground truth.  
        - The **Radar chart** visualizes multi-metric trade-offs: Doc2Vec leads in precision and MAP, while TF-IDF remains lightweight with faster runtime.
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
        **Observations:**
        - **TF-IDF** shows a narrow similarity distribution, meaning it struggles to clearly separate very similar vs. dissimilar hotels.  
        - **LSI (Gensim)** offers a broader distribution, capturing latent topics and improving hotel clustering.  
        - **Doc2Vec** produces the most informative spread, reflecting stronger semantic separation, crucial for nuanced hotel recommendations (e.g., distinguishing “beach resort” vs. “city business hotel”).
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
        **Key Takeaways:**
        - **TF-IDF** emphasizes frequent keywords like *“new property”* or *“good facilities”*, focusing on surface-level features.  
        - **LSI (Gensim)** identifies broader topics such as *customer service* and *travel experience*, offering more context than raw frequency.  
        - **Doc2Vec** captures richer semantic cues, linking words such as *“family-friendly”*, *“cleanliness”*, and *“near beach”*, which better reflect user intent and hotel attributes.
        """
    )

    # --- Final Summary ---
    st.subheader("Overall Evaluation Summary")
    st.markdown(
        """
        - **Best Overall Model:** *Doc2Vec* delivers the most accurate and semantically meaningful recommendations, though it comes with higher computational cost.  
        - **Best Lightweight Alternative:** *TF-IDF* offers fast performance with reasonably high accuracy, suitable for real-time applications or when resources are limited.  
        - **Balanced Approach:** *LSI (Gensim)* sits in the middle, offering topic-level interpretability and efficiency.  

        **Recommendation for Agoda:**  
        - Use **Doc2Vec** for the main personalized recommendation pipeline (quality-focused).  
        - Deploy **TF-IDF** as a fallback or lightweight semantic search option (speed-focused).  
        - Continue A/B testing hybrid approaches (Doc2Vec + ALS collaborative filtering) to maximize both personalization and scalability.
        """
    )
