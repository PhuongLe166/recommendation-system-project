import streamlit as st

def inject_css():
    st.markdown(
        """
<style>
    .stApp { background: #f6f7fb; }
    /* Global font & base colors */
    :root {
        --brand-500: #667eea;
        --brand-600: #5a6fe0;
        --brand-700: #4f46e5;
        --brand-800: #4338ca;
        --ink-900: #0f172a;
        --ink-700: #334155;
        --muted-500: #64748b;
        --panel: #ffffff;
        --panel-border: #e9edf5;
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class^="css"], .stApp { font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol' !important; }
    
    /* Header styling */
    .main-header { 
        background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        padding: 2rem; 
        border-radius: 15px; 
        color: white; 
        margin-bottom: 2rem; 
        box-shadow:0 15px 40px rgba(102,126,234,.35);
    }
    
    .hotel-card { background: white; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #667eea;
        box-shadow:0 6px 20px rgba(0,0,0,0.06); margin-bottom: 1rem;}
    .metric-card { background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb; text-align: center;}
    .score-badge { background: linear-gradient(45deg,#667eea,#764ba2); color: white; padding:.5rem 1rem; border-radius: 20px; font-weight: bold; display:inline-block;}
    .tag-chip { background: rgba(102,126,234,0.08); border:1px solid rgba(102,126,234,0.24); padding:.3rem .6rem; border-radius:15px; font-size:.85rem; display:inline-block; margin:.2rem; color:#667eea;}
    .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
    .stTabs [data-baseweb="tab"] { height: 40px; background: white; border-radius: 20px; padding: 0 1rem; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }

    /* Filters panel wrapper as white block */
    .filters-card { background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow:0 6px 20px rgba(0,0,0,0.06); }
</style>
        """,
        unsafe_allow_html=True,
    )