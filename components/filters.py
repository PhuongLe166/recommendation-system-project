import streamlit as st
from utils import bump_filtered_count, reset_filtered_count

def render_filters():
    st.markdown("### 🎯 Bộ lọc thông minh")
    st.caption("Đang bật: **3** bộ lọc")

    with st.expander("Cài đặt bộ lọc", expanded=True):
        st.select_slider("⭐ Hạng sao tối thiểu", options=[1,2,3,4,5], value=4, format_func=lambda x: f"{x} sao")
        st.slider("📊 Điểm đánh giá tối thiểu", 5.0, 10.0, 8.5, 0.1)
        st.number_input("💬 Số đánh giá tối thiểu", 0, 5000, 100, 50)

        st.markdown("**🏨 Tiện nghi ưu tiên**")
        col_tag1, col_tag2 = st.columns(2)
        with col_tag1:
            st.checkbox("🏖️ Gần biển", value=True)
            st.checkbox("💼 Business")
        with col_tag2:
            st.checkbox("🌊 Spa", value=True)
            st.checkbox("👨‍👩‍👧‍👦 Gia đình")

        col_service, col_location = st.columns(2)
        with col_service:
            st.slider("Điểm dịch vụ tối thiểu", 0.0, 10.0, 8.0, 0.1)
        with col_location:
            st.slider("Điểm vị trí tối thiểu", 0.0, 10.0, 8.5, 0.1)

        col_apply, col_reset = st.columns(2)
        applied = False
        reseted = False
        with col_apply:
            if st.button("Áp dụng", type="primary", use_container_width=True):
                applied = True
                bump_filtered_count()
                st.rerun()
        with col_reset:
            if st.button("Xóa", use_container_width=True):
                reseted = True
                reset_filtered_count()
                st.rerun()

    return applied if 'applied' in locals() else False, reseted if 'reseted' in locals() else False
