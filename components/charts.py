import streamlit as st
import plotly.express as px
import pandas as pd

def render_scatter(df: pd.DataFrame):
    st.markdown("### 📊 Phân bố điểm số")
    fig = px.scatter(
        df,
        x="Điểm tổng",
        y="Độ tương đồng (%)",
        size="Hạng sao",
        color="Hạng sao",
        hover_data=["name","price"],
        title="Scatter: Độ tương đồng (%) vs. Điểm tổng • kích thước ~ hạng sao",
        color_continuous_scale="viridis",
        size_max=30,
    )
    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        showlegend=True,
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
    st.plotly_chart(fig, use_container_width=True)
