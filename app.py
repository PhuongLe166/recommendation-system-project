import streamlit as st
import time
from theme import inject_css
from components.header import render_header
from components.sidebar import render_sidebar
from components.search import render_search
from components.filters import render_filters
from components.results import render_results
from components.evaluation import render_evaluation
from data import load_hotels_df, load_id_mapping, load_metrics, load_doc2vec_model, load_doc2vec_similarity
from components.business_problem import render_business_problem
from recommender import (
    DEFAULT_FILTERS, apply_filters, count_active_filters,
    search_by_query_doc2vec, similar_by_hotel_doc2vec
)

st.set_page_config(
    page_title="Hotel Recommender System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS with fixed positioning
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
        "current_page": "Recommendation",
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Render Sidebar and get current page ---
current_page = render_sidebar()

# --- Header (fixed at top) ---
render_header()

# --- Main Content Area based on current page ---
if current_page == "Business Problem":
    render_business_problem()

elif current_page == "Evaluation & Report":
    # Evaluation & Report Page
    render_evaluation()

elif current_page == "Recommendation":
    # Original Recommendation Page
    
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
            st.error(f"Failed to load data/model: {e}")
            st.stop()

    hotels_df, id_map, d2v, d2v_sim, metrics = load_resources()

    # --- Search Interface (scrolls with content) ---
    search_result = render_search(hotels_df)
    # --- Horizontal Filters below Search ---
    new_filters_h, action_h = render_filters(
        st.session_state.filters,
        active_count=count_active_filters(st.session_state.filters),
        layout="horizontal",
    )
    if action_h == "apply":
        st.session_state.filters = new_filters_h
    elif action_h == "reset":
        st.session_state.filters = DEFAULT_FILTERS.copy()

    # --- Process Search with Loading State ---
    processing_placeholder = st.empty()

    if search_result["query_submitted"] and search_result["query_text"]:
        st.session_state.processing = True
        
    if search_result["similar_submitted"] and search_result["chosen_hotel_id"]:
        st.session_state.processing = True

    # Show processing message
    if st.session_state.processing:
        with processing_placeholder.container():
            with st.spinner("🔍 Processing..."):
                time.sleep(1.0)  # Simulate processing
                
                if search_result["query_submitted"] and search_result["query_text"]:
                    results = search_by_query_doc2vec(
                        search_result["query_text"], 
                        d2v, 
                        hotels_df, 
                        id_map,
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

    # --- Layout: Main only (filters moved under search) ---
    col_main = st.container()
    with col_main:
        # Display results
        display_recs = st.session_state.recs if st.session_state.recs else st.session_state.all_recs
        total_all = len(st.session_state.all_recs)
        
        if display_recs:
            pass
        
        render_results(
            display_recs, 
            st.session_state.filters, 
            total_all=total_all
        )

    # Apply filters to current results if needed
    if st.session_state.all_recs:
        filtered = apply_filters(st.session_state.all_recs, st.session_state.filters)
        st.session_state.recs = filtered


# --- Footer ---
st.markdown("---")
st.caption("🏨 Hotel Recommender System - Powered by Doc2Vec & Content-Based Filtering")