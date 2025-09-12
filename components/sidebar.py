# components/sidebar.py
import streamlit as st

def render_sidebar():
    """
    Render the sidebar navigation
    Returns the current selected page
    """
    with st.sidebar:
        st.subheader("🏨 Navigation")
        
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
        
        if st.button("🧮 ALS", 
                    key="nav_als", 
                    use_container_width=True,
                    type="primary" if current_page == "ALS" else "secondary"):
            st.session_state.current_page = "ALS"
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