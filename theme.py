import streamlit as st

def inject_css():
    st.markdown(
        """
<style>
    .stApp { background: #f6f7fb; }
    
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
</style>
        """,
        unsafe_allow_html=True,
    )