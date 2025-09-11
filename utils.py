import streamlit as st
import random

def bump_filtered_count():
    st.session_state.filtered_hotels = random.randint(8, 15)

def reset_filtered_count():
    st.session_state.filtered_hotels = 12
