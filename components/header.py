import streamlit as st

def render_header():
    st.markdown(
        """
<div class="main-header" style="display:flex; align-items:center; justify-content:center; text-align:center;">
  <h1 style="margin:0; font-size:2rem;">🏨 Hotel Recommender System</h1>
</div>
        """,
        unsafe_allow_html=True,
    )

