import streamlit as st

def _render_card(hotel: dict, index: int):
    st.markdown(
        f"""
        <div style='background:white; padding:1.5rem; border-radius:15px; border-left:4px solid #667eea; margin-bottom:1rem;'>
            <div style='display:flex; justify-content:space-between; align-items:start;'>
                <div>
                    <h4 style='margin:0;'>{index}. {hotel['name']}</h4>
                    <p style='color:#f59e0b; margin:.2rem 0;'>{'⭐' * hotel['stars']} ({hotel['stars']} sao)</p>
                </div>
                <span class='score-badge'>
                    {hotel['match_score']}% phù hợp
                </span>
            </div>
            <p style='color:#6b7280; font-size:.9rem; margin:.5rem 0;'>
                📍 {hotel['address']} • 💬 {hotel['reviews']} đánh giá
            </p>
            <p style='margin:.5rem 0;'>{hotel['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tags
    tags_html = " ".join([f"<span class='tag-chip'>{tag}</span>" for tag in hotel["tags"]])
    st.markdown(f"<div style='margin:.5rem 0;'>{tags_html}</div>", unsafe_allow_html=True)

    # Rating bars
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.progress(hotel["location_score"] / 10)
        st.caption(f"Vị trí • {hotel['location_score']}")
    with col_r2:
        st.progress(hotel["cleanliness_score"] / 10)
        st.caption(f"Sạch sẽ • {hotel['cleanliness_score']}")
    with col_r3:
        st.progress(hotel["service_score"] / 10)
        st.caption(f"Dịch vụ • {hotel['service_score']}")

    # Facts
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric("Điểm tổng", f"{hotel['rating']}/10", label_visibility="visible")
    with col_f2:
        st.metric("Giá ước tính", hotel["price"], label_visibility="visible")
    with col_f3:
        st.metric("Hạng sao TB", f"{hotel['stars']}.0", label_visibility="visible")

def render_cards_grid(hotels: list[dict]):
    st.markdown("### 📋 Danh sách khách sạn được khuyến nghị")
    for i in range(0, len(hotels), 2):
        c1, c2 = st.columns(2)
        for col, idx in [(c1, i), (c2, i+1)]:
            if idx < len(hotels):
                with col:
                    _render_card(hotels[idx], idx + 1)
