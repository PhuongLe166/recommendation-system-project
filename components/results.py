# components/results.py
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st

from recommender import count_active_filters
from components.cards import render_cards_grid
from components.charts import render_scatter


def _build_chart_df(recs: List[Dict]) -> pd.DataFrame:
    """
    Chuẩn hoá dữ liệu cho biểu đồ scatter:
      - Cột 'Điểm tổng'
      - Cột 'Độ tương đồng (%)'
      - Cột 'Hạng sao'
      - Thêm 'name' và 'price' để hover trong chart
    """
    rows = []
    for r in recs:
        name = r.get("name", "N/A")
        stars = r.get("stars", 0) or 0
        total = r.get("total", 0.0) or 0.0
        sim_pct = (r.get("similarity", 0.0) or 0.0) * 100.0
        price = "2.000.000 ₫" if stars >= 4 else "1.200.000 ₫"

        rows.append(
            {
                "name": name,
                "price": price,
                "Điểm tổng": float(total),
                "Độ tương đồng (%)": float(sim_pct),
                "Hạng sao": float(stars),
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
    Render toàn bộ khu vực col_main:
      1) Banner tóm tắt kết quả
      2) Lưới thẻ khách sạn (cards)
      3) Biểu đồ scatter (nếu đủ dữ liệu)

    Params
    ------
    recs: list các khách sạn đã sẵn sàng hiển thị (đã lọc nếu có)
    filters: dict bộ lọc hiện hành (để tính số filter đang bật)
    total_all: tổng ứng viên ban đầu (để hiển thị câu 'từ X ứng viên ban đầu')
    show_chart: có hiển thị biểu đồ không (mặc định True)
    """
    if not recs:
        st.info("Hãy nhập mô tả hoặc chọn khách sạn và nhấn tìm kiếm để bắt đầu.")
        return

    shown = len(recs)
    active_n = count_active_filters(filters)

    # Banner tóm tắt
    if total_all is not None and total_all >= shown:
        st.success(
            f"✅ **{shown} khách sạn** sau khi áp dụng **{active_n} bộ lọc** "
            f"(từ {total_all} ứng viên ban đầu)"
        )
    else:
        st.success(f"✅ **{shown} khách sạn** sau khi áp dụng **{active_n} bộ lọc**")

    # Cards grid
    render_cards_grid(recs)

    # Chart
    if show_chart and len(recs) >= 3:
        df_chart = _build_chart_df(recs)
        render_scatter(df_chart)
    elif show_chart and len(recs) > 0:
        st.caption("📊 Cần ≥ 3 khách sạn để hiển thị biểu đồ phân bố.")
