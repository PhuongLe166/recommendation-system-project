import streamlit as st

def render_insights():
    st.markdown("### 🧠 AI Insights")
    st.caption("**Doc2Vec** hiểu ngữ nghĩa mô tả để gợi ý khách sạn tương đồng.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Độ chính xác (Top-5)", "89.2%")
        st.metric("Tốc độ phản hồi", "< 2s")
    with col_m2:
        st.metric("MAP", "0.324")
        st.metric("Model", "v1.0")
