import streamlit as st
from theme import inject_css
from data import get_hotels, hotels_df_for_chart
from components.header import render_header
from components.search import render_search
from components.filters import render_filters
from components.cards import render_cards_grid
from components.charts import render_scatter
from components.insights import render_insights

st.set_page_config(
    page_title="Hotel Recommender System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
inject_css()

# Init state
if "filtered_hotels" not in st.session_state:
    st.session_state.filtered_hotels = 12

# ---- Header
render_header()

# ---- Search
mode, search_query, selected_hotel, find_clicked, similar_clicked = render_search()

# # Mock data (giữ giống bản demo)
# hotels_data = get_hotels()
# df_chart = hotels_df_for_chart(hotels_data)

# # ---- Layout: sidebar + main
# col_sidebar, col_main = st.columns([1, 2.5], gap="large")

# with col_sidebar:
#     # Filters
#     filters_applied, filters_reset = render_filters()

#     # Insights
#     render_insights()

# with col_main:
#     # Result summary line
#     st.success(
#         f"✅ **{st.session_state.filtered_hotels} khách sạn** sau khi áp dụng **3 bộ lọc** (từ 20 ứng viên ban đầu)"
#     )

#     # Cards grid
#     render_cards_grid(hotels_data)

#     # Chart
#     render_scatter(df_chart)

# st.markdown("---")
# st.caption("🏨 Hệ thống khuyến nghị khách sạn thông minh - Powered by Doc2Vec & Content-Based Filtering")
