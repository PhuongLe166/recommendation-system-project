import streamlit as st
from typing import Tuple, Optional, Dict

def render_filters(
    filters: Dict,
    active_count: int
) -> Tuple[Dict, Optional[str]]:
    """
    Render panel bộ lọc.
    Trả về:
      - filters mới (copy đã cập nhật từ UI)
      - action: 'apply' | 'reset' | None
    """
    f = dict(filters)  # làm việc trên bản copy
    action = None

    # Wrapper card and header
    with st.container(border=False):
        st.markdown(
            f"""
            <div class="filters-header">
                <div class="filters-title">🔎 Filters</div>
                <div class="filters-badge">{active_count} active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("General", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                f["min_stars"] = st.select_slider(
                    "⭐ Minimum stars", options=[1, 2, 3, 4, 5], value=int(f["min_stars"]),
                    format_func=lambda x: f"{x}★"
                )
                f["min_comments"] = st.number_input(
                    "💬 Min. reviews", min_value=0, value=int(f["min_comments"]), step=10
                )
            with col2:
                f["min_rating"] = st.slider(
                    "📊 Min. review score", 5.0, 10.0, float(f["min_rating"]), 0.1
                )

        with st.expander("Amenities", expanded=True):
            st.caption("Select your preferences")
            col_tag1, col_tag2 = st.columns(2)
            with col_tag1:
                f["near_beach"] = st.checkbox("🏖️ Near beach", value=bool(f["near_beach"]))
                f["business"]   = st.checkbox("💼 Business", value=bool(f["business"]))
            with col_tag2:
                f["spa"]    = st.checkbox("🧖 Spa", value=bool(f["spa"]))
                f["family"] = st.checkbox("👨‍👩‍👧‍👦 Family", value=bool(f["family"]))

        with st.expander("Scores", expanded=True):
            col_service, col_location = st.columns(2)
            with col_service:
                f["min_service"] = st.slider(
                    "🛎️ Min. service", 0.0, 10.0, float(f["min_service"]), 0.1
                )
            with col_location:
                f["min_location"] = st.slider(
                    "📍 Min. location", 0.0, 10.0, float(f["min_location"]), 0.1
                )

        st.markdown("<div class='filters-actions' />", unsafe_allow_html=True)
        col_apply, col_reset = st.columns(2)
        with col_apply:
            if st.button("Apply filters", type="primary", use_container_width=True):
                action = "apply"
        with col_reset:
            if st.button("Reset", use_container_width=True):
                action = "reset"

    return f, action
