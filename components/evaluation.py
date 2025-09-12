import streamlit as st
import pandas as pd
import os, json, base64

def render_evaluation():
    st.markdown("""
        <h1 class='ev-title'>Evaluation & Report</h1>
    """, unsafe_allow_html=True)

    # --- Inject page-specific CSS (inspired by provided design) ---
    st.markdown(
        """
        <style>
        .ev-title { margin-bottom: 25px; font-size: 26px; font-weight: 600; color: #222; }

        .ev-table {
            background: #ffffff; border: 1px solid #e9edf5; border-radius: 12px;
            padding: 14px 14px 8px 14px; box-shadow: 0 8px 22px rgba(17,24,39,0.08);
            margin-bottom: 18px;
        }
        .ev-table h3 { margin: 4px 6px 10px 6px; color: #1f2937; }
        .ev-table table { width: 100%; border-collapse: collapse; border-radius: 10px; overflow: hidden; }
        .ev-table thead th {
            background: #1abc9c; color: white; text-align: center; padding: 10px;
        }
        .ev-table tbody td { text-align: center; padding: 10px; border-bottom: 1px solid #eef2ff; }
        .ev-table tbody tr:nth-child(even) td { background: #fbfdff; }

        .ev-charts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        .ev-img-card { background:#ffffff; border:1px solid #e9edf5; border-radius:12px; padding:10px; box-shadow:0 8px 22px rgba(17,24,39,0.06); }
        .ev-img { width:100%; height: var(--img-h, 240px); object-fit: contain; background:#fff; border-radius:8px; }
        .ev-caption { text-align:center; color:#6b7280; font-size:0.9rem; margin-top:6px }
        
        /* New style matching index.html */
        .ev-section { margin-bottom: 35px; }
        .ev-h3 { font-size: 20px; font-weight: 600; margin: 6px 0 12px 0; border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; color:#111827; }
        .ev-card { background: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .ev-table table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 15px; }
        .ev-table th, .ev-table td { padding: 12px; text-align: center; }
        .ev-table th { background: #1976d2; color: #ffffff; font-weight: 600; }
        .ev-table tr:nth-child(even) td { background: #f9f9f9; }
        .ev-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
        .ev-chart-card { background: #ffffff; border-radius: 10px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-align: center; }
        .ev-chart-card h4 { font-size: 15px; margin-bottom: 12px; color: #444; font-weight: 500; }
        .ev-chart-card img { width: 100%; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _render_subtitle(title: str, icon: str | None = None, anchor: str | None = None):
        icon_html = f"<span class='ev-subtitle-icon'>{icon}</span>" if icon else ""
        anchor_html = f"<a class='ev-subtitle-anchor' href='#{anchor}'>↪</a>" if anchor else ""
        st.markdown(
            f"""
            <div class='ev-subtitle' id='{anchor or ''}'>
                {icon_html}
                <span class='ev-subtitle-text'>{title}</span>
                {anchor_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Load metrics ---
    try:
        with open("hotel_ui_data/performance_metrics.json", "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        st.error(f"Không thể load performance_metrics.json: {e}")
        return

    methods = metrics_data.get("methods", {})
    total_hotels = metrics_data.get("total_hotels", 0)
    generated_at = metrics_data.get("generated_at", "")

    # In this style, we focus on sections and cards; metrics can be added later if needed.

    # --- Bảng so sánh ---
    df_metrics = pd.DataFrame([
        {
            "Method": m.upper(),
            "Build Time (s)": round(v["build_time"], 2),
            "Precision@5": round(v["precision_5"], 4),
            "MAP": round(v["map_score"], 4),
            "Correlation": round(v["correlation"], 4),
        }
        for m, v in methods.items()
    ])
    # --- Styled comparison table ---
    headers = ["Method", "Build Time (s)", "Precision@5", "MAP", "Correlation"]
    rows_html = "".join([
        f"<tr>"
        f"<td>{row['Method']}</td>"
        f"<td>{row['Build Time (s)']:.2f}</td>"
        f"<td>{row['Precision@5']:.2f}</td>"
        f"<td>{row['MAP']:.2f}</td>"
        f"<td>{row['Correlation']:.2f}</td>"
        f"</tr>" for _, row in df_metrics.iterrows()
    ])
    table_html = (
        "<div class='ev-section'>"
        "<h3 class='ev-h3'>So sánh các mô hình</h3>"
        "<div class='ev-card ev-table'>"
        "<table>"
        "<thead><tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead>"
        "<tbody>" + rows_html + "</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # --- Chọn lọc ảnh chính ---
    img_dir = "hotel_ui_data/images"
    selected_images = [
        ("methods_precision.png", "Precision@K Comparison"),
        ("methods_quality.png", "MAP & Correlation"),
        ("methods_radar.png", "Overall Performance Radar")
    ]

    if os.path.exists(img_dir):
        # Section: Main charts
        charts_html = ["<div class='ev-section'>", "<h3 class='ev-h3'>Biểu đồ minh họa</h3>", "<div class='ev-charts'>"]
        for filename, caption in selected_images:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    charts_html.append(f"<div class='ev-chart-card'><h4>{caption}</h4><img src='data:image/png;base64,{b64}' /></div>")
                except Exception:
                    charts_html.append(f"<div class='ev-chart-card'><h4>{caption}</h4></div>")
        charts_html.append("</div></div>")
        st.markdown("".join(charts_html), unsafe_allow_html=True)

        # --- Similarity distributions for 3 models ---
        def _card(img_path: str, caption: str):
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return f"<div class='ev-chart-card'><h4>{caption}</h4><img src='data:image/png;base64,{b64}' /></div>"
            except Exception:
                return f"<div class='ev-chart-card'><h4>{caption}</h4></div>"

        sim_images = [
            ("tfidf_similarity_distribution.png", "TF-IDF Similarity Distribution"),
            ("gensim_similarity_distribution.png", "LSI (Gensim) Similarity Distribution"),
            ("doc2vec_similarity_distribution.png", "Doc2Vec Similarity Distribution"),
        ]
        sim_html = ["<div class='ev-section'>", "<h3 class='ev-h3'>Phân phối độ tương đồng theo mô hình</h3>", "<div class='ev-charts'>"]
        for filename, caption in sim_images:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                sim_html.append(_card(path, caption))
        sim_html.append("</div></div>")
        st.markdown("".join(sim_html), unsafe_allow_html=True)

        # --- Wordclouds for 3 models ---
        # Wordclouds
        wc_images = [
            ("tfidf_wordcloud.png", "TF-IDF Wordcloud"),
            ("gensim_wordcloud.png", "LSI (Gensim) Wordcloud"),
            ("doc2vec_wordcloud.png", "Doc2Vec Wordcloud"),
        ]
        wc_html = ["<div class='ev-section'>", "<h3 class='ev-h3'>Wordcloud theo mô hình</h3>", "<div class='ev-charts'>"]
        for filename, caption in wc_images:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                wc_html.append(_card(path, caption))
        wc_html.append("</div></div>")
        st.markdown("".join(wc_html), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Chưa tìm thấy thư mục images, hãy chạy lại bước save plots.")
