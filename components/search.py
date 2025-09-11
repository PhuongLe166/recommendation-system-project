import streamlit as st

def render_search():
    col1, col2 = st.columns([3, 1])

    mode = "description"
    find_clicked = False
    similar_clicked = False
    search_query = ""
    selected_hotel = ""

    with col1:
        tab1, tab2 = st.tabs(["🔍 Tìm theo mô tả", "🏨 Chọn theo khách sạn"])
        with tab1:
            search_query = st.text_input(
                "Mô tả khách sạn lý tưởng của bạn",
                placeholder="luxury beach resort spa, family hotel with pool...",
                label_visibility="collapsed",
            )
            mode = "description"
        with tab2:
            selected_hotel = st.selectbox(
                "Chọn khách sạn mẫu",
                [""] ,  # có thể thay bằng danh sách thật nếu bạn có
                label_visibility="collapsed",
            )
            mode = "hotel" if selected_hotel else "description"

    with col2:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            find_clicked = st.button("🔍 Tìm kiếm bằng AI", type="primary", use_container_width=True)
        with col_btn2:
            st.button("⚙️ Bộ lọc", use_container_width=True)

    return mode, search_query, selected_hotel, find_clicked, similar_clicked
