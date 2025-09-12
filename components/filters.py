import streamlit as st
from typing import Tuple, Optional, Dict

def render_filters(
    filters: Dict,
    active_count: int,
    layout: str = "vertical",
) -> Tuple[Dict, Optional[str]]:
    """
    Render filter panel.
    Returns:
      - updated filters (copy)
      - action: 'apply' | 'reset' | None
    """
    f = dict(filters)  # làm việc trên bản copy
    action = None

    # Wrapper card and header
    with st.container(border=False):
        if layout == "horizontal":
            with st.container(border=True):
                st.subheader("🔎 Filters")
                st.caption(f"{active_count} active")

                # Row 1: numeric and slider controls
                r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
                with r1c1:
                    f["min_stars"] = st.select_slider(
                        "⭐ Min stars", options=[1, 2, 3, 4, 5], value=int(f["min_stars"]),
                        format_func=lambda x: f"{x}★"
                    )
                with r1c2:
                    f["min_rating"] = st.slider("📊 Min rating", 5.0, 10.0, float(f["min_rating"]), 0.1)
                with r1c3:
                    f["min_comments"] = st.number_input("💬 Min reviews", min_value=0, value=int(f["min_comments"]), step=10)
                with r1c4:
                    f["min_service"] = st.slider("🛎️ Min service", 0.0, 10.0, float(f["min_service"]), 0.1)
                with r1c5:
                    f["min_location"] = st.slider("📍 Min location", 0.0, 10.0, float(f["min_location"]), 0.1)

                st.divider()

                # Row 2: amenities + actions aligned right
                c_spa, c_beach, c_family, c_business, c_actions = st.columns([1, 1, 1, 1, 2])
                with c_spa:
                    f["spa"] = st.checkbox("🧖 Spa", value=bool(f["spa"]))
                with c_beach:
                    f["near_beach"] = st.checkbox("🏖️ Beach", value=bool(f["near_beach"]))
                with c_family:
                    f["family"] = st.checkbox("👨‍👩‍👧‍👦 Family", value=bool(f["family"]))
                with c_business:
                    f["business"] = st.checkbox("💼 Business", value=bool(f["business"]))
                with c_actions:
                    spacer, col_apply, col_reset = st.columns([1.0, 1.1, 1.0])
                    with col_apply:
                        if st.button("Apply", type="primary", use_container_width=True):
                            action = "apply"
                    with col_reset:
                        if st.button("Reset", use_container_width=True):
                            action = "reset"
        else:
            st.subheader("🔎 Filters")
            st.caption(f"{active_count} active")

        if layout != "horizontal":
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

        st.divider()
        if layout != "horizontal":
            col_apply, col_reset = st.columns(2)
            with col_apply:
                if st.button("Apply", type="primary", use_container_width=True):
                    action = "apply"
            with col_reset:
                if st.button("Reset", use_container_width=True):
                    action = "reset"

    return f, action
