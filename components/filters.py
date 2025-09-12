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
    st.markdown("### 🎯 Bộ lọc thông minh")
    st.caption(f"Đang bật: **{active_count}** bộ lọc")

    f = dict(filters)  # làm việc trên bản copy

    with st.expander("Cài đặt bộ lọc", expanded=True):
        f["min_stars"] = st.select_slider(
            "⭐ Hạng sao tối thiểu", options=[1, 2, 3, 4, 5], value=int(f["min_stars"]),
            format_func=lambda x: f"{x} sao"
        )
        f["min_rating"] = st.slider(
            "📊 Điểm đánh giá tối thiểu", 5.0, 10.0, float(f["min_rating"]), 0.1
        )
        f["min_comments"] = st.number_input(
            "💬 Số đánh giá tối thiểu", min_value=0, value=int(f["min_comments"]), step=10
        )

        st.markdown("**🏨 Tiện nghi ưu tiên**")
        col_tag1, col_tag2 = st.columns(2)
        with col_tag1:
            f["near_beach"] = st.checkbox("🏖️ Gần biển", value=bool(f["near_beach"]))
            f["business"]   = st.checkbox("💼 Business", value=bool(f["business"]))
        with col_tag2:
            f["spa"]    = st.checkbox("🌊 Spa", value=bool(f["spa"]))
            f["family"] = st.checkbox("👨‍👩‍👧‍👦 Gia đình", value=bool(f["family"]))

        col_service, col_location = st.columns(2)
        with col_service:
            f["min_service"] = st.slider(
                "Điểm dịch vụ tối thiểu", 0.0, 10.0, float(f["min_service"]), 0.1
            )
        with col_location:
            f["min_location"] = st.slider(
                "Điểm vị trí tối thiểu", 0.0, 10.0, float(f["min_location"]), 0.1
            )

        col_apply, col_reset = st.columns(2)
        action = None
        with col_apply:
            if st.button("✅ Áp dụng", type="primary", use_container_width=True):
                action = "apply"
        with col_reset:
            if st.button("🗑️ Xóa", use_container_width=True):
                action = "reset"

    return f, action
