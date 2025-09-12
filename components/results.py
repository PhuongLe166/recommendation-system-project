# components/results.py
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st

from recommender import count_active_filters
from components.cards import render_cards_grid
from components.charts import render_scatter


def _build_chart_df(recs: List[Dict]) -> pd.DataFrame:
    """
    Normalize data for scatter chart:
      - Column 'Total Score'
      - Column 'Similarity (%)'
      - Column 'Star Rating'
      - Keep 'name' and 'price' for hover
    """
    rows = []
    for r in recs:
        name = r.get("name", "N/A")
        stars = r.get("stars", 0) or 0
        total = r.get("total", 0.0) or 0.0
        sim_pct = (r.get("similarity", 0.0) or 0.0) * 100.0
        price = "$200" if stars >= 4 else "$120"

        rows.append(
            {
                "name": name,
                "price": price,
                "Total Score": float(total),
                "Similarity (%)": float(sim_pct),
                "Star Rating": float(stars),
            }
        )
    return pd.DataFrame(rows)


def render_results(
    recs: List[Dict],
    filters: Dict,
    total_all: Optional[int] = None,
    show_chart: bool = True,
) -> None:
    """
    Render the main area:
      1) Results banner
      2) Cards grid
      3) Scatter chart (if enough data)

    Params
    ------
    recs: list of hotels ready to display
    filters: active filters dict
    total_all: total initial candidates (for banner)
    show_chart: whether to show chart (default True)
    """
    if not recs:
        st.info("Enter a description or pick a hotel and click search to start.")
        return

    shown = len(recs)
    active_n = count_active_filters(filters)

    # Summary banner
    if total_all is not None and total_all >= shown:
        st.success(
            f"✅ **{shown} hotels** after applying **{active_n} filters** "
            f"(from {total_all} initial candidates)"
        )
    else:
        st.success(f"✅ **{shown} hotels** after applying **{active_n} filters**")

    # Cards grid
    render_cards_grid(recs)

    # Chart
    if show_chart and len(recs) >= 3:
        df_chart = _build_chart_df(recs)
        render_scatter(df_chart)
    elif show_chart and len(recs) > 0:
        st.caption("📊 Need ≥ 3 hotels to display the distribution chart.")
