"""
====================================================================
WEEK 6: STREAMLIT MULTI-MODAL CHAT INTERFACE
====================================================================
Kết nối tới FastAPI backend (Port 8000)
"""

import streamlit as st
import requests
import time
import os

# Địa chỉ URL của Backend FastAPI
API_URL = "http://127.0.0.1:8000"

# ============================================================
# CẤU HÌNH TRANG WEB
# ============================================================
st.set_page_config(
    page_title="IFAB Football Laws AI",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Trợ lý AI Luật Bóng Đá IFAB")
st.markdown("Hãy hỏi bất kỳ câu hỏi nào về luật bóng đá, các tình huống việt vị, thẻ phạt hoặc lỗi chạm tay!")

# ============================================================
# QUAN TRỌNG: KHỞI TẠO LƯU TRỮ LỊCH SỬ CHAT
# ============================================================
# Mỗi lần người dùng bấm nút, Streamlit sẽ chạy lại toàn bộ file từ trên xuống dưới.
# Do đó ta phải dùng session_state để lưu lại lịch sử chat mà không bị mất dữ liệu.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# HÀM BỔ TRỢ: KIỂM TRA BACKEND ĐÃ SẴN SÀNG CHƯA
# ============================================================
def is_backend_ready():
    """Gọi tới Endpoint Readiness Probe của Tuần 5 để check xem GPU đã load xong model chưa."""
    try:
        response = requests.get(f"{API_URL}/api/chat/ready", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# ============================================================
# HIỂN THỊ LỊCH SỬ CHAT TRÊN GIAO DIỆN
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Nếu tin nhắn có chứa ảnh minh họa từ RAG -> Hiển thị ra luôn
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "images" in message and message["images"]:
                        for img_data in message["images"]:
                            # Kiểm tra nếu img_data là Dict, ta lấy key chứa đường dẫn (thường là 'image_path' hoặc 'path')
                            actual_path = img_data.get("image_path") if isinstance(img_data, dict) else img_data
                            if actual_path:
                                st.image(actual_path, caption="Sơ đồ minh họa", use_container_width=True)

# ============================================================
# Ô NHẬP LIỆU VÀ XỬ LÝ SEARCH RAG
# ============================================================
if prompt := st.chat_input("Nhập câu hỏi của bạn về tình huống trận đấu..."):
    
    # 1. Hiển thị câu hỏi của người dùng lên màn hình và lưu vào bộ nhớ lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý phản hồi từ Assistant AI
    with st.chat_message("assistant"):
        
        # Kiểm tra Cổng bảo vệ (Readiness Gate) của Tuần 5
        if not is_backend_ready():
            st.error("⏳ Hệ thống AI đang khởi động hoặc chưa bật Server. Vui lòng đợi vài giây rồi thử lại!")
        else:
            with st.spinner("🤖 Đang lục tìm và phân tích sách luật IFAB..."):
                try:
                    # Gửi request POST tới FastAPI Server
                    response = requests.post(
                        f"{API_URL}/api/chat",
                        json={"query": prompt},
                        timeout=100  # Đồng bộ thời gian chờ với backend
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Lấy câu trả lời và mảng ảnh sơ đồ (Khớp với các key trả về ở tuần 4/5)
                        answer = data.get("chatbot_answer", "Không có câu trả lời.")
                        images = data.get("ui_render_images", [])

                        # In câu trả lời văn bản ra giao diện
                        st.markdown(answer)
                        
                        # In các hình ảnh sơ đồ luật đi kèm (nếu có)
                        if images:
                            for img_data in images:
                                actual_path = img_data.get("image_path") if isinstance(img_data, dict) else img_data
                                
                                if actual_path:
                                    # ── LOGIC SỬA LỖI ĐƯỜNG DẪN ──────────────────────────────────────
                                    # Nếu đường dẫn tương đối không tồn tại trực tiếp ở gốc
                                    if not os.path.exists(actual_path):
                                        # Thử tìm nó bên trong thư mục con "RAG football"
                                        alternative_path = os.path.join("RAG football", actual_path)
                                        if os.path.exists(alternative_path):
                                            actual_path = alternative_path
            # ─────────────────────────────────────────────────────────────────

                                    # Kiểm tra cuối cùng xem file có thực sự tồn tại hay không trước khi in
                                    if os.path.exists(actual_path):
                                        st.image(actual_path, caption="Sơ đồ luật liên quan", use_container_width=True)
                                    else:
                                        # Nếu không tìm thấy ở cả 2 nơi, hiện thông báo text nhẹ nhàng thay vì crash web
                                        st.warning(f"📷 Không tìm thấy file ảnh minh họa tại: {actual_path}")

                        # Lưu câu trả lời của AI + Ảnh vào lịch sử chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "images": images
                        })
                        
                    else:
                        st.error(f"Lỗi Server {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Không thể kết nối tới Backend. Bạn đã chạy file `week5_api_server.py` chưa?")
                except requests.exceptions.Timeout:
                    st.error("⏰ Đã quá thời gian phản hồi. Mô hình AI mất quá nhiều thời gian để suy luận.")