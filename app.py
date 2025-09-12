import streamlit as st
import time
from theme import inject_css
from components.header import render_header
from components.search import render_search
from components.filters import render_filters
from components.results import render_results
from data import load_hotels_df, load_id_mapping, load_metrics, load_doc2vec_model, load_doc2vec_similarity
from recommender import (
    DEFAULT_FILTERS, apply_filters, count_active_filters,
    search_by_query_doc2vec, similar_by_hotel_doc2vec
)

st.set_page_config(
    page_title="Hotel Recommender System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global CSS
inject_css()

# --- State Management ---
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "filters": DEFAULT_FILTERS.copy(),
        "all_recs": [],
        "recs": [],
        "last_query": "",
        "last_hotel_id": None,
        "processing": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Load Resources ---
@st.cache_resource(show_spinner=False)
def load_resources():
    """Load all resources with caching"""
    try:
        hotels_df = load_hotels_df()
        id_map = load_id_mapping()
        d2v = load_doc2vec_model()
        d2v_sim = load_doc2vec_similarity()
        metrics = load_metrics().get("methods", {}).get("doc2vec", {})
        return hotels_df, id_map, d2v, d2v_sim, metrics
    except Exception as e:
        st.error(f"Lỗi nạp dữ liệu/model: {e}")
        st.stop()

hotels_df, id_map, d2v, d2v_sim, metrics = load_resources()

# --- Header ---
render_header()

# --- Search Interface ---
search_result = render_search(hotels_df)

# --- Process Search with Loading State ---
processing_placeholder = st.empty()

if search_result["query_submitted"] and search_result["query_text"]:
    st.session_state.processing = True
    
if search_result["similar_submitted"] and search_result["chosen_hotel_id"]:
    st.session_state.processing = True

# Show processing message
if st.session_state.processing:
    with processing_placeholder.container():
        with st.spinner("🔍 Đang xử lý..."):
            time.sleep(1.0)  # Simulate processing
            
            if search_result["query_submitted"] and search_result["query_text"]:
                results = search_by_query_doc2vec(
                    search_result["query_text"], 
                    d2v, 
                    hotels_df, 
                    top_k=20
                )
                st.session_state.all_recs = results
                st.session_state.recs = []
                st.session_state.last_query = search_result["query_text"]
                st.session_state.last_hotel_id = None
                
            elif search_result["similar_submitted"] and search_result["chosen_hotel_id"]:
                results = similar_by_hotel_doc2vec(
                    search_result["chosen_hotel_id"], 
                    hotels_df, 
                    d2v_sim, 
                    id_map, 
                    top_k=20
                )
                st.session_state.all_recs = results
                st.session_state.recs = []
                st.session_state.last_hotel_id = search_result["chosen_hotel_id"]
                st.session_state.last_query = ""
            
            st.session_state.processing = False
            st.rerun()

# --- Layout: Main + Filters on Right ---
col_main, col_filters = st.columns([2.5, 1], gap="large")

with col_main:
    # Display results
    display_recs = st.session_state.recs if st.session_state.recs else st.session_state.all_recs
    total_all = len(st.session_state.all_recs)
    
    if display_recs:
        st.markdown("""
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .hotel-card {
            animation: fadeIn 0.5s ease-out;
        }
        </style>
        """, unsafe_allow_html=True)
    
    render_results(
        display_recs, 
        st.session_state.filters, 
        total_all=total_all
    )

with col_filters:
    # Always show filters
    new_filters, action = render_filters(
        st.session_state.filters,
        active_count=count_active_filters(st.session_state.filters),
    )

    # Handle filter actions
    if action == "apply":
        st.session_state.filters = new_filters
        if st.session_state.all_recs:
            filtered = apply_filters(st.session_state.all_recs, st.session_state.filters)
            st.session_state.recs = filtered
            
    elif action == "reset":
        st.session_state.filters = DEFAULT_FILTERS.copy()
        st.session_state.recs = []

# --- Footer ---
st.markdown("---")
st.caption("🏨 Hotel Recommender System - Powered by Doc2Vec & Content-Based Filtering")