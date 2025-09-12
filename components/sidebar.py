# components/sidebar.py
import streamlit as st

def render_sidebar():
    """
    Render the sidebar navigation
    Returns the current selected page
    """
    with st.sidebar:
        st.markdown("""
        <style>
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }
        .nav-item {
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .nav-item:hover {
            background: rgba(102, 126, 234, 0.1);
            transform: translateX(5px);
        }
        .nav-item-active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">🏨 Navigation</div>', unsafe_allow_html=True)
        
        # Get current page from session state
        current_page = st.session_state.get("current_page", "Recommendation")
        
        # Navigation buttons
        if st.button("📊 Business Problem", 
                    key="nav_business", 
                    use_container_width=True, 
                    type="primary" if current_page == "Business Problem" else "secondary"):
            st.session_state.current_page = "Business Problem"
            st.rerun()
        
        if st.button("📈 Evaluation & Report", 
                    key="nav_evaluation", 
                    use_container_width=True,
                    type="primary" if current_page == "Evaluation & Report" else "secondary"):
            st.session_state.current_page = "Evaluation & Report"
            st.rerun()
        
        if st.button("🎯 Recommendation", 
                    key="nav_recommendation", 
                    use_container_width=True,
                    type="primary" if current_page == "Recommendation" else "secondary"):
            st.session_state.current_page = "Recommendation"
            st.rerun()
        
        # Add some space
        st.markdown("---")
        
        # Add info about current page
        st.info(f"📍 Current: **{current_page}**")
        
    return current_page