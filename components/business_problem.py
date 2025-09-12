import streamlit as st

def render_business_problem():
    st.header("Business Problem")

    st.subheader("1. Context")
    st.write(
        "Agoda is a global online travel platform offering hotels, resorts, apartments, and homestays at competitive prices. "
        "Users can easily search, compare, and book rooms. However, choosing among thousands of hotels remains challenging, "
        "especially as each customer has very different preferences."
    )

    st.subheader("2. Challenges")
    st.markdown(
        "- Users spend too much time filtering for suitable hotels.\n"
        "- Traditional filters (price, stars, location) are not enough to reflect true preferences.\n"
        "- Hoteliers lack behavioral analytics, leading to suboptimal service and marketing strategies."
    )

    st.subheader("3. Business goals")
    st.write("For customers:")
    st.markdown("- Save time searching.\n- Receive personalized hotel recommendations based on descriptions, preferences, and past experiences.")
    st.write("For hotels/enterprises:")
    st.markdown("- Understand customer behavior to improve services.\n- Increase conversion from search → booking.\n- Improve satisfaction and loyalty.")

    st.subheader("4. Proposed solution")
    st.markdown(
        "- Build a Recommender System based on Content-Based Filtering (TF-IDF, Gensim, Doc2Vec).\n"
        "- Combine hotel descriptions, reviews, amenities, and user behavior data.\n"
        "- Support intelligent search: by text description and by similar hotels.\n"
        "- Provide a clear interface with sidebar filters and results table."
    )

    st.subheader("5. Expected outcomes")
    st.markdown(
        "- Customers: quickly find suitable hotels with personalized experiences.\n"
        "- Business: understand customers, optimize strategy, and increase bookings.\n"
        "- System: become a useful analytics and decision-support tool."
    )
