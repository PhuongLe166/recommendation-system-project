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

st.markdown("""
<style>
    /* Remove Streamlit default top header (white bar) so it doesn't overlap */
    header[data-testid="stHeader"] { display: none; }

    /* Ensure sidebar stays fixed and visible */
    section[data-testid="stSidebar"] {
        position: fixed; top: 0; left: 0; height: 100vh;
        width: 21rem; min-width: 21rem;
        transform: none !important; visibility: visible !important; opacity: 1 !important;
        z-index: 1000;
    }

    /* Sticky header */
    .main-header { position: fixed; top: 0; left: 21rem; right: 0; z-index: 999; margin: 0; }

    /* Prevent overlap */
    .main .block-container { padding-top: 140px; padding-right: 420px; }
    .main { margin-left: 21rem; }

    /* Fixed filters column */
    div[data-testid="column"]:last-child > div:first-child {
        position: fixed; top: 150px; right: 24px; width: 360px;
        max-height: calc(100vh - 170px); overflow-y: auto; padding: 0; z-index: 997;
    }
    /* Filter card */
    div[data-testid="column"]:last-child > div:first-child > div {
        background: #ffffff; border: 1px solid #e9edf5; border-radius: 16px;
        padding: 18px 18px 12px 18px; box-shadow: 0 12px 28px rgba(17,24,39,0.08);
        position: relative; overflow: hidden;
    }
    /* Accent bar */
    div[data-testid="column"]:last-child > div:first-child > div::after {
        content: ""; position: absolute; left: 0; right: 0; top: 0; height: 6px;
        background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    }

    /* Filters header */
    .filters-header { display:flex; align-items:center; justify-content:space-between; margin-bottom: .5rem; }
    .filters-title { font-weight: 800; font-size: 1.1rem; color:#0f172a; }
    .filters-badge { background: rgba(102,126,234,.12); color:#4f46e5; border:1px solid rgba(102,126,234,.24);
        padding:.15rem .5rem; border-radius: 999px; font-size:.8rem; font-weight:700; }

    /* Expander */
    div[data-testid="column"]:last-child [data-testid="stExpander"] > details {
        border: 1px solid #e9edf5; border-radius: 12px; background: #fafbff; overflow: hidden;
    }
    div[data-testid="column"]:last-child [data-testid="stExpander"] > details > summary {
        padding: 14px 16px; font-weight: 600; color: #1f2937; background: linear-gradient(180deg,#ffffff,#f6f7fb);
    }
    div[data-testid="column"]:last-child [data-testid="stExpander"] > details[open] {
        border-color: #d9def0; background: #ffffff; box-shadow: inset 0 0 0 1px rgba(102,126,234,0.12);
    }

    /* Controls spacing */
    div[data-testid="column"]:last-child .stSlider, 
    div[data-testid="column"]:last-child .stNumberInput, 
    div[data-testid="column"]:last-child .stCheckbox { margin-top: 6px; margin-bottom: 10px; }

    /* Inputs accents */
    div[data-testid="column"]:last-child input[type="range"] { accent-color: #667eea; }
    div[data-testid="column"]:last-child input[type="range"]::-webkit-slider-runnable-track { background: #e8edf6; height: 6px; border-radius: 999px; }
    div[data-testid="column"]:last-child input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; margin-top: -6px; background: linear-gradient(135deg,#667eea,#764ba2); border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }
    div[data-testid="column"]:last-child input[type="range"]::-moz-range-track { background: #e8edf6; height: 6px; border-radius: 999px; }
    div[data-testid="column"]:last-child input[type="range"]::-moz-range-thumb { background: linear-gradient(135deg,#667eea,#764ba2); width: 18px; height: 18px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }
    div[data-testid="column"]:last-child input[type="number"] { border-radius: 10px; }
    div[data-testid="column"]:last-child .st-bc { accent-color: #667eea; }

    /* Actions spacing */
    .filters-actions { height: 4px; margin-bottom: .25rem; }

    /* Buttons */
    div[data-testid="column"]:last-child .stButton > button {
        width: 100%; border-radius: 10px; border: none;
        box-shadow: 0 6px 16px rgba(102,126,234,0.18);
        background: linear-gradient(135deg,#667eea,#764ba2);
    }
    div[data-testid="column"]:last-child .stButton > button[kind="secondary"] {
        background: #eef2ff; color:#3730a3; box-shadow:none; border:1px solid #c7d2fe;
    }

    /* Sticky tabs below header */
    .stTabs { position: sticky; top: 140px; z-index: 998; background: #f6f7fb; padding-top: 0.5rem; margin-top: 0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
</style>
""", unsafe_allow_html=True)

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
            st.error(f"Lỗi nạp dữ liệu/model: {e}")
            st.stop()

    hotels_df, id_map, d2v, d2v_sim, metrics = load_resources()

    # --- Search Interface (scrolls with content) ---
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