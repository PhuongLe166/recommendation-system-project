# components/search.py
import streamlit as st
import pandas as pd

def render_search(hotels_df: pd.DataFrame):
    """
    Simplified search interface with inline buttons
    
    Returns dict with:
        - query_text: The search query text
        - query_submitted: Whether query search was triggered
        - chosen_hotel_id: Selected hotel ID
        - similar_submitted: Whether similar search was triggered
    """
    query_submitted = False
    similar_submitted = False
    chosen_hotel_id = None
    query_text = ""

    # Main search interface with better styling
    # st.markdown("""
    # <div style='background: white; padding: 1.5rem; border-radius: 15px; 
    #             box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 1.5rem;'>
    #     <h3 style='margin: 0 0 1rem 0; color: #1f2937;'>🔍 Tìm kiếm khách sạn</h3>
    # </div>
    # """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔍 Tìm theo mô tả", "🏨 Tìm theo khách sạn tương tự"])
    
    with tab1:
        # Query search - single row layout
        col_input, col_button = st.columns([3, 1])
        
        with col_input:
            query_text = st.text_input(
                "Mô tả khách sạn lý tưởng của bạn",
                placeholder="spa, luxury beach resort, family hotel...",
                help="Nhập mô tả chi tiết về khách sạn bạn muốn tìm",
                key="search_query_input",
                label_visibility="collapsed"
            )
        
        with col_button:
            query_submitted = st.button(
                "🔍 Search", 
                type="primary",
                use_container_width=True,
                key="query_search_button"
            )
        
        # Show last query if exists
        if st.session_state.get("last_query"):
            st.caption(f"🔍 Tìm kiếm gần nhất: *{st.session_state.last_query}*")
    
    with tab2:
        # Similar hotels - single row layout
        col_select, col_button2 = st.columns([3, 1])
        
        with col_select:
            # Hotel selection dropdown
            hotel_options = [("", "-- Chọn một khách sạn --")]
            hotel_options.extend([
                (row["Hotel_ID"], f"{row['Hotel_Name']} (ID: {row['Hotel_ID']})")
                for _, row in hotels_df.iterrows()
            ])
            
            selected = st.selectbox(
                "Chọn khách sạn để tìm các khách sạn tương tự",
                options=hotel_options,
                format_func=lambda x: x[1],
                key="hotel_selector_input",
                label_visibility="collapsed"
            )
            
            chosen_hotel_id = selected[0] if selected[0] else None
        
        with col_button2:
            similar_submitted = st.button(
                "🔍 Search", 
                type="primary",
                use_container_width=True,
                key="similar_search_button"
            )
        
        # Show last selection if exists
        if st.session_state.get("last_hotel_id"):
            last_hotel_name = next(
                (name for hid, name in hotel_options[1:] if hid == st.session_state.last_hotel_id), 
                None
            )
            if last_hotel_name:
                st.caption(f"🏨 Đã chọn gần nhất: *{last_hotel_name.split(' (ID:')[0]}*")

    # Check for empty inputs after button clicks
    if query_submitted and not query_text.strip():
        st.warning("⚠️ Vui lòng nhập mô tả khách sạn để tìm kiếm!")
        query_submitted = False
    
    if similar_submitted and not chosen_hotel_id:
        st.warning("⚠️ Vui lòng chọn một khách sạn để tìm khách sạn tương tự!")
        similar_submitted = False

    return {
        "query_text": query_text,
        "query_submitted": query_submitted,
        "chosen_hotel_id": chosen_hotel_id,
        "similar_submitted": similar_submitted,
    }