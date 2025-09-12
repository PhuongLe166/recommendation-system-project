import streamlit as st
from textwrap import dedent
import re

def render_business_problem():
    html = dedent("""
<style>
/* Match Evaluation page palette and card layout */
.ev-title { margin-bottom: 25px; font-size: 26px; font-weight: 600; color: #222; }
.ev-section { margin-bottom: 35px; }
.ev-h3 { font-size: 20px; font-weight: 600; margin: 6px 0 12px 0; border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; color:#111827; }
.ev-card { background: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e9edf5; }
.ev-card p { margin: 6px 0; font-size: 15px; color:#1f2937; }
.ev-card ul { margin: 8px 0 0 20px; padding: 0; }
.ev-card ul li { margin-bottom: 6px; font-size: 15px; color:#374151; }
</style>

<h1 class="ev-title">Business Problem</h1>

<div class="ev-section">
  <h3 class="ev-h3">1. Bối cảnh</h3>
  <div class="ev-card">
    <p>
      Agoda là nền tảng đặt phòng trực tuyến toàn cầu, cung cấp dịch vụ khách sạn, resort, căn hộ, homestay với mức giá cạnh tranh.
      Người dùng có thể dễ dàng tìm kiếm, so sánh và đặt phòng. Tuy nhiên, việc lựa chọn giữa hàng nghìn khách sạn
      vẫn là một thách thức lớn, đặc biệt khi nhu cầu của mỗi khách hàng rất đa dạng.
    </p>
  </div>
  
</div>

<div class="ev-section">
  <h3 class="ev-h3">2. Thách thức</h3>
  <div class="ev-card">
    <ul>
      <li>Người dùng mất nhiều thời gian để lọc khách sạn phù hợp với nhu cầu cá nhân.</li>
      <li>Các bộ lọc truyền thống (giá, sao, địa điểm) chưa đủ để phản ánh sở thích thực sự.</li>
      <li>Chủ khách sạn thiếu dữ liệu phân tích hành vi, dẫn đến chiến lược dịch vụ và marketing chưa tối ưu.</li>
    </ul>
  </div>
</div>

<div class="ev-section">
  <h3 class="ev-h3">3. Mục tiêu kinh doanh</h3>
  <div class="ev-card">
    <p><strong>Đối với khách hàng:</strong></p>
    <ul>
      <li>Tiết kiệm thời gian tìm kiếm.</li>
      <li>Nhận gợi ý khách sạn cá nhân hóa dựa trên mô tả, sở thích và trải nghiệm trước đó.</li>
    </ul>
    <p><strong>Đối với chủ khách sạn/doanh nghiệp:</strong></p>
    <ul>
      <li>Hiểu rõ hành vi khách hàng để cải thiện dịch vụ.</li>
      <li>Tăng tỷ lệ chuyển đổi từ tìm kiếm → đặt phòng.</li>
      <li>Gia tăng sự hài lòng và lòng trung thành của khách hàng.</li>
    </ul>
  </div>
</div>

<div class="ev-section">
  <h3 class="ev-h3">4. Giải pháp đề xuất</h3>
  <div class="ev-card">
    <ul>
      <li>Xây dựng <strong>Recommender System</strong> dựa trên Content-Based Filtering (TF-IDF, Gensim, Doc2Vec).</li>
      <li>Kết hợp dữ liệu mô tả khách sạn, đánh giá, tiện ích và hành vi khách hàng.</li>
      <li>Cho phép tìm kiếm thông minh:
        <ul>
          <li>Tìm theo mô tả (VD: <em>"spa, luxury beach resort, family hotel"</em>).</li>
          <li>Tìm theo khách sạn tương tự.</li>
        </ul>
      </li>
      <li>Cung cấp giao diện trực quan với sidebar filter và bảng kết quả.</li>
    </ul>
  </div>
</div>

<div class="ev-section">
  <h3 class="ev-h3">5. Kỳ vọng kết quả</h3>
  <div class="ev-card">
    <ul>
      <li><strong>Khách hàng:</strong> nhanh chóng tìm được khách sạn phù hợp, trải nghiệm cá nhân hóa.</li>
      <li><strong>Doanh nghiệp:</strong> hiểu rõ khách hàng, tối ưu chiến lược kinh doanh, tăng tỷ lệ đặt phòng.</li>
      <li><strong>Hệ thống:</strong> trở thành công cụ phân tích dữ liệu và hỗ trợ ra quyết định.</li>
    </ul>
  </div>
</div>
""")
    # Remove any leading indentation on each line to prevent Markdown from
    # interpreting the block as code (4-space indented lines become code blocks)
    html_no_indent = re.sub(r'^[ \t]+', '', html, flags=re.MULTILINE)
    st.markdown(html_no_indent, unsafe_allow_html=True)
