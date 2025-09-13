import streamlit as st

def render_business_problem():
    st.header("Business Problem")

    # 1. Context
    st.subheader("1. Context")
    st.write(
        "Agoda is a global online travel platform offering millions of hotels, resorts, apartments, "
        "and homestays at competitive prices. While users can easily search, compare, and book rooms, "
        "the overwhelming number of choices often makes decision-making difficult. Each traveler has "
        "unique preferences — families may prioritize child-friendly facilities, business travelers may "
        "seek central locations, while couples may look for beachside resorts. Standard filters such as "
        "price, star rating, and location are insufficient to capture these diverse needs."
    )

    # 2. Challenges
    st.subheader("2. Challenges")
    st.markdown(
        "- **For customers:**\n"
        "  - Spend excessive time browsing and applying multiple filters.\n"
        "  - Difficulty identifying hotels that match personal preferences and trip context.\n"
        "  - Limited personalization; textual reviews and descriptions are underutilized.\n\n"
        "- **For Agoda (business side):**\n"
        "  - Lack of advanced behavioral analytics to understand customer intent.\n"
        "  - Missed opportunities to optimize conversion rates.\n"
        "  - Limited insights for hotel partners into strengths, weaknesses, and customer sentiment."
    )

    # 3. Business Goals
    st.subheader("3. Business Goals")
    st.write("**For customers:**")
    st.markdown(
        "- Reduce time spent searching for suitable hotels.\n"
        "- Provide personalized recommendations aligned with user needs and past behavior.\n"
        "- Enhance booking experience and overall satisfaction."
    )
    st.write("**For Agoda and hotel partners:**")
    st.markdown(
        "- Increase conversion rates from search → booking.\n"
        "- Offer data-driven insights into customer behavior and sentiment.\n"
        "- Strengthen customer loyalty and repeat bookings."
    )

    # 4. Proposed Solution
    st.subheader("4. Proposed Solution")
    st.markdown(
        "- **Content-Based Filtering:**\n"
        "  - Apply TF-IDF, LSI (Gensim), and Doc2Vec to represent hotel descriptions and customer reviews.\n"
        "  - Enable semantic search, allowing queries such as *“quiet hotel near the beach with spa.”*\n"
        "  - Provide hotel-to-hotel recommendations (\"similar to this hotel\").\n\n"
        "- **Collaborative Filtering (ALS - Alternating Least Squares):**\n"
        "  - Use historical booking and rating data to uncover hidden patterns between users and hotels.\n"
        "  - Learn latent factors (user preferences, hotel attributes) to generate personalized suggestions.\n"
        "  - Support cold-start scenarios by leveraging community behavior.\n\n"
        "- **Agoda Insights Dashboard:**\n"
        "  - Benchmark hotels against system averages.\n"
        "  - Extract sentiment insights from reviews.\n"
        "  - Highlight strengths and weaknesses for hotel partners."
    )

    # 5. Expected Outcomes
    st.subheader("5. Expected Outcomes")
    st.markdown(
        "- **For customers:**\n"
        "  - Faster discovery of relevant hotels.\n"
        "  - Personalized, context-aware recommendations.\n\n"
        "- **For Agoda and partners:**\n"
        "  - Higher booking conversion rates.\n"
        "  - Deeper understanding of customer needs and satisfaction drivers.\n"
        "  - Stronger competitive positioning in the OTA market.\n\n"
        "- **For the system:**\n"
        "  - A hybrid recommendation engine combining Content-Based and ALS.\n"
        "  - A decision-support tool for both customers and business stakeholders."
    )
