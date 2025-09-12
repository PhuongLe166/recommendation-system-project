# components/insights.py
import streamlit as st

def render_insights(doc2vec_metrics: dict):
    st.markdown("### 🧠 AI Insights")
    st.caption("**Doc2Vec** hiểu ngữ nghĩa mô tả để gợi ý khách sạn tương đồng.")
    p5 = doc2vec_metrics.get("precision_5", 0)
    map_s = doc2vec_metrics.get("map_score", 0)
    build = doc2vec_metrics.get("build_time", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Độ chính xác (Top-5)", f"{p5:.1%}" if isinstance(p5, (int,float)) else str(p5))
        st.metric("Build time", f"{build:.1f}s" if isinstance(build, (int,float)) else str(build))
    with col2:
        st.metric("MAP", f"{map_s:.3f}" if isinstance(map_s, (int,float)) else str(map_s))
        st.metric("Model", "Doc2Vec")
