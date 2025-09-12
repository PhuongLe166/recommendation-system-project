# components/cards.py
# --------------------------------------------
# Fixed-size card layout
# --------------------------------------------

import streamlit as st
from typing import List
from streamlit.components.v1 import html as st_html


# --------- Helpers ---------
def _infer_tags(hotel: dict) -> List[str]:
    """Simple tag inference from description."""
    tags: List[str] = []
    desc = (hotel.get("desc", "") or "").lower()

    if any(w in desc for w in ["beach", "sea", "ocean", "coastal"]):
        tags.append("🏖️ Near beach")
    if any(w in desc for w in ["spa", "massage", "wellness", "relax"]):
        tags.append("💆 Spa")
    if any(w in desc for w in ["gym", "fitness", "workout"]):
        tags.append("💪 Gym")
    if any(w in desc for w in ["family", "kids", "children", "playground"]):
        tags.append("👨‍👩‍👧‍👦 Family")

    # Tối đa 4 tag
    return tags[:4]


def _pct10(score) -> float:
    """Convert 0-10 scale to percent (0-100)."""
    try:
        x = float(score)
    except Exception:
        x = 0.0
    return max(0.0, min(x * 10.0, 100.0))


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _line_clamp_html(text: str) -> str:
    return (text or "").strip() or "No description."


# --------- Card Renderer ---------
def _render_card(hotel: dict, index: int) -> None:
    """Render one hotel card with fixed layout."""

    # Safe extract
    name = hotel.get("name", "Unknown Hotel")
    stars = _safe_int(hotel.get("stars"), 0)
    similarity = _safe_float(hotel.get("similarity"), 0.0)
    match_score = round(similarity * 100.0, 1)

    total = _safe_float(hotel.get("total"), 0.0)
    loc = _safe_float(hotel.get("loc"), 0.0)
    clean = _safe_float(hotel.get("clean"), 0.0)
    serv = _safe_float(hotel.get("serv"), 0.0)

    address = hotel.get("addr", "N/A")
    comments = _safe_int(hotel.get("comments"), 0)
    desc_html = _line_clamp_html(hotel.get("desc", ""))

    tags = _infer_tags(hotel)
    tags_html = " • ".join(tags)

    # Class ẩn nhưng giữ chỗ (visibility:hidden)
    stars_class = "hc-stars" if stars > 0 else "hc-stars hc-hide"
    tags_class = "hc-tags" if tags else "hc-tags hc-hide"
    comments_html = f" • 💬 {comments} reviews" if comments > 0 else ""

    # HTML + CSS (không thụt đầu dòng để tránh Markdown coi là code)
    html = f"""
<style>
  .hotel-card {{
    border: 1px solid #e5e7eb;
    border-left: 4px solid #667eea;
    border-radius: 12px;
    background: #fff;
    padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(17,24,39,0.06);

    /* LƯỚI 7 HÀNG CỐ ĐỊNH: đảm bảo từng khối luôn ở đúng vị trí */
    display: grid;
    grid-template-rows:
      60px   /* 1. Header: tiêu đề + sao + badge */
      22px   /* 2. Meta: địa chỉ + comments */
      96px   /* 3. Mô tả (clamp 4 dòng) */
      38px   /* 4. Tags (chip 1 dòng) */
      22px   /* 5. Label 'Đánh giá chi tiết:' */
      74px   /* 6. Ba progress bars */
      56px;  /* 7. Tổng điểm */
    row-gap: 8px;      /* 6 khoảng = 48px */
    height: 440px;     /* >= tổng hàng + gap (416px) để không cắt */
    overflow: hidden;
  }}

  /* Header */
  .hc-header {{ display:flex; justify-content:space-between; align-items:flex-start; }}
  .hc-title  {{
    margin:0; font-size:16px; color:#1f2937; font-weight:700;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
    overflow:hidden; line-height:1.3;
  }}
  .hc-stars {{ color:#f59e0b; font-size:14px; height:18px; }}
  .hc-hide  {{ visibility:hidden; }}

  .hc-badge {{
    background: linear-gradient(135deg,#667eea,#764ba2);
    color:#fff; padding:6px 12px; border-radius:18px; font-size:13px; font-weight:700;
    white-space:nowrap;
  }}

  /* Meta */
  .hc-meta {{
    color:#6b7280; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }}

  /* Description */
  .hc-desc {{
    color:#374151; font-size:14px; line-height:1.6; overflow:hidden;
    display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical;
  }}

  /* Tags */
  .hc-tags {{
    background:#eff6ff; color:#1e40af; font-size:13px; padding:8px 10px; border-radius:8px;
    display:flex; align-items:center; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
  }}

  /* Scores */
  .hc-label {{ font-weight:600; color:#374151; font-size:14px; display:flex; align-items:flex-end; }}
  .hc-grid  {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
  .hc-rowlbl {{ display:flex; justify-content:space-between; font-size:12px; color:#6b7280; }}
  .hc-bar  {{ background:#e5e7eb; height:6px; border-radius:3px; overflow:hidden; }}
  .hc-fill {{ background:#3b82f6; height:100%; }}

  /* Total */
  .hc-total {{ border-top:1px solid #e5e7eb; display:flex; align-items:center; }}
  .hc-total-inner {{ width:100%; display:flex; justify-content:space-between; align-items:center; }}
  .hc-total .val {{ font-size:22px; font-weight:800; color:#1f2937; }}
</style>

<div class="hotel-card">
  <!-- 1) Header -->
  <div class="hc-header">
    <div style="max-width:72%;">
      <h3 class="hc-title">{index}. {name}</h3>
      <div class="{stars_class}">{'⭐' * stars} ({stars} stars)</div>
    </div>
    <div class="hc-badge">{match_score}% match</div>
  </div>

  <!-- 2) Meta -->
  <div class="hc-meta">📍 {address}{comments_html}</div>

  <!-- 3) Description -->
  <div class="hc-desc">{desc_html}</div>

  <!-- 4) Tags -->
  <div class="{tags_class}">{tags_html}</div>

  <!-- 5) Label -->
  <div class="hc-label">Detailed scores:</div>

  <!-- 6) Progress bars -->
  <div class="hc-grid">
    <div>
      <div class="hc-rowlbl"><span>Location</span><span>{loc:.1f}</span></div>
      <div class="hc-bar"><div class="hc-fill" style="width:{_pct10(loc)}%"></div></div>
    </div>
    <div>
      <div class="hc-rowlbl"><span>Cleanliness</span><span>{clean:.1f}</span></div>
      <div class="hc-bar"><div class="hc-fill" style="width:{_pct10(clean)}%"></div></div>
    </div>
    <div>
      <div class="hc-rowlbl"><span>Service</span><span>{serv:.1f}</span></div>
      <div class="hc-bar"><div class="hc-fill" style="width:{_pct10(serv)}%"></div></div>
    </div>
  </div>

  <!-- 7) Total -->
  <div class="hc-total">
    <div class="hc-total-inner">
      <span style="font-size:14px; color:#6b7280;">Total score</span>
      <span class="val">{total:.1f}/10</span>
    </div>
  </div>
</div>
"""
    # Chiều cao frame lớn hơn chút để không cắt bóng đổ/border
    st_html(html, height=470, scrolling=False)


# --------- Public API ---------
def render_cards_grid(hotels: List[dict]) -> None:
    """Hiển thị danh sách khách sạn theo grid 2 cột, card cố định size."""
    if not hotels:
        return

    st.markdown("### 📋 Recommended hotels")

    for i in range(0, len(hotels), 2):
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if i < len(hotels):
                _render_card(hotels[i], i + 1)
        with col2:
            if i + 1 < len(hotels):
                _render_card(hotels[i + 1], i + 2)
