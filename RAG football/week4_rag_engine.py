"""
====================================================================
WEEK 4: MULTI-MODAL HYBRID RAG ENGINE (IFAB FOOTBALL LAWS)
====================================================================
Kiến trúc: Local Retrieval (BGE-M3) + Cloud Generation (Gemini 2.0)
Thư viện chuẩn mới: google-genai (Thay thế gói google-generativeai cũ)
"""

import os
import sys
import json

# Đảm bảo Terminal hiển thị đúng font tiếng Việt có dấu
sys.stdout.reconfigure(encoding="utf-8")

from google import genai
from langchain_community.vectorstores import FAISS
# Gọi lớp LocalHuggingFaceEmbeddings đã định nghĩa từ file Tuần 3 của bạn
from week3_chunking import LocalHuggingFaceEmbeddings

# ====================================================================
# CẤU HÌNH ĐƯỜNG DẪN & BIẾN MÔI TRƯỜNG
# ====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "faiss_ifab_local_db")

API_KEY = os.environ.get("GOOGLE_API_KEY", "GEMINI_API_KEY")
client  = genai.Client(api_key=API_KEY)
LLM_MODEL_NAME = "gemini-3.1-flash-lite"


class IFABHybridRAGEngine:
    """
    Hệ thống lõi RAG: Tiếp nhận câu hỏi, thực hiện truy xuất đa phương thức
    (Văn bản + Sơ đồ) từ DB local, tổng hợp prompt và sinh câu trả lời qua Gemini.
    """
    def __init__(self):
        print("🤖 [Hệ thống] Đang khởi tạo bộ suy luận RAG Đa phương thức...")
        
        # 1. Khởi tạo lại mô hình nhúng Local BGE-M3
        self.embeddings = LocalHuggingFaceEmbeddings()
        
        # 2. Nạp cơ sở dữ liệu Vector từ ổ đĩa cứng
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"❌ [Lỗi] Không tìm thấy thư mục Vector DB tại: {DB_PATH}.\n"
                f"Vui lòng chạy file tuần 3 (week3_chunking.py) trước để tạo DB!"
            )
            
        self.vector_db = FAISS.load_local(
            DB_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True  # Cho phép giải mã cấu trúc file .pkl an toàn
        )
        print("✅ [Hệ thống] Đã nạp thành công bộ nhớ Vector FAISS cục bộ.")

    def _hybrid_multimodal_retrieve(self, query: str, top_text: int = 3, top_image: int = 2) -> tuple[list[dict], list[dict]]:
        """
        Bước 1 + Bước 2: Thực hiện tìm kiếm ngữ nghĩa song song dựa trên vector
        và dùng bộ lọc Metadata để tách biệt luồng Văn bản luật và luồng Sơ đồ ảnh.
        """
        # Quét lấy dư số lượng ứng viên (k) đề phòng trường hợp các kết quả đầu tiên bị lệch về một loại
        candidates = self.vector_db.similarity_search(query, k=top_text + top_image + 5)
        
        retrieved_texts = []
        retrieved_images = []
        
        for doc in candidates:
            element_type = doc.metadata.get("element_type")
            
            # Phân loại và thu thập khối văn bản luật (Text)
            if element_type == "text" and len(retrieved_texts) < top_text:
                retrieved_texts.append({
                    "content": doc.page_content,
                    "printed_page": doc.metadata.get("printed_page"),
                    "absolute_page": doc.metadata.get("absolute_page")
                })
                
            # Phân loại và thu thập khối thông tin sơ đồ chiến thuật (Image Diagram)
            elif element_type == "tactical_diagram" and len(retrieved_images) < top_image:
                retrieved_images.append({
                    "description": doc.page_content, # Nội dung văn bản mô tả sơ đồ của Gemini
                    "image_file": doc.metadata.get("image_file"),
                    "printed_page": doc.metadata.get("printed_page"),
                    "absolute_page": doc.metadata.get("absolute_page"),
                    "region_index": doc.metadata.get("region_index")
                })
                
            # Nếu đã gom đủ số lượng cấu hình cho cả 2 loại thì dừng quét để tối ưu hiệu năng
            if len(retrieved_texts) == top_text and len(retrieved_images) == top_image:
                break
                
        return retrieved_texts, retrieved_images

    def retrieve_context(self, query: str) -> list:
        """
        Truy xuất các đoạn văn bản và mô tả sơ đồ liên quan đến câu hỏi.
        Trả về danh sách đối tượng có thuộc tính .text để tương thích với Ragas.
        """
        from types import SimpleNamespace
        texts, images = self._hybrid_multimodal_retrieve(query)
        retrieved = []
        for t in texts:
            retrieved.append(SimpleNamespace(text=t["content"]))
        for img in images:
            retrieved.append(SimpleNamespace(text=img["description"]))
        return retrieved

    def generate_answer(self, query: str) -> dict:
        """
        Bước 3 + Bước 4: Lắp ráp cấu trúc Prompt tổng hợp, gọi API LLM mới 
        và cấu trúc hóa dữ liệu đầu ra phục vụ cho việc Render giao diện (UI).
        """
        # 1. Truy xuất dữ liệu lai
        texts, images = self._hybrid_multimodal_retrieve(query, top_text=3, top_image=2)
        
        # 2. Xây dựng khối văn bản Ngữ cảnh (Context Assembly)
        context_segments = []
        
        context_segments.append("=== PHẦN 1: CÁC ĐIỀU LUẬT BẰNG VĂN BẢN TRÍCH XUẤT ===")
        for idx, t in enumerate(texts):
            context_segments.append(
                f"[Tài liệu tham khảo {idx+1}] (Nguồn: Trang luật {t['printed_page']}):\n{t['content']}"
            )
            
        if images:
            context_segments.append("\n=== PHẦN 2: MÔ TẢ CÁC SƠ ĐỒ HÌNH ẢNH MINH HỌA LIÊN QUAN ===")
            for idx, img in enumerate(images):
                context_segments.append(
                    f"[Sơ đồ minh họa {idx+1}] (Nguồn: Trang luật {img['printed_page']} - Phân vùng số {img['region_index']}):\n{img['description']}"
                )
                
        full_context = "\n\n".join(context_segments)
        
        # 3. Thiết lập Prompt chỉ thị hệ thống (System Prompt)
        system_instruction = (
            "Bạn là một Trợ lý AI cấp cao, đóng vai trò Chuyên gia Trọng tài tối cao của IFAB Việt Nam.\n"
            "Nhiệm vụ của bạn là giải thích luật bóng đá một cách chính xác, chặt chẽ, chuyên nghiệp và dễ hiểu.\n"
            "Yêu cầu nghiêm ngặt:\n"
            "1. Chỉ sử dụng thông tin được cung cấp trong mục 'NGỮ CẢNH THAM KHẢO CỤC BỘ' bên dưới để trả lời.\n"
            "2. Trình bày câu trả lời có cấu trúc rõ ràng, phân tích logic theo từng khía cạnh của tình huống hỏi.\n"
            "3. Nếu trong phần ngữ cảnh có nhắc đến Sơ đồ minh họa, hãy khéo léo gợi ý người dùng xem hình ảnh sơ đồ hiển thị kèm theo ở cuối câu trả lời."
        )
        
        # Lắp ráp cấu trúc Prompt hoàn chỉnh gửi lên Google
        final_prompt = f"""
{system_instruction}

----------------------------------------------------------------------
NGỮ CẢNH THAM KHẢO CỤC BỘ:
{full_context}
----------------------------------------------------------------------

CÂU HỎI CỦA NGƯỜI DÙNG: 
{query}

CÂU TRẢ LỜI CỦA BẠN (Viết hoàn toàn bằng tiếng Việt tự nhiên):
"""

        # 4. Gọi sinh câu trả lời bằng thư viện SDK mới (google-genai)
        try:
            llm_response = client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=final_prompt
            )
            answer_text = llm_response.text
        except Exception as e:
            print(f"❌ [Lỗi Gọi API] Có lỗi xảy ra khi kết nối với Gemini: {e}")
            answer_text = "Xin lỗi, hệ thống không thể kết nối tới bộ não LLM của Google lúc này. Vui lòng kiểm tra lại API Key hoặc mạng internet."

        # 5. Định hình dữ liệu cấu trúc trả về cho Frontend UI Render
        # Trích xuất đường dẫn tương đối để Frontend chỉ việc nạp ảnh từ thư mục assets/cropped_images
        ui_images = []
        for img in images:
            ui_images.append({
                "image_path": f"cropped_images/{img['image_file']}",
                "printed_page": img['printed_page'],
                "region_index": img['region_index']
            })
            
        return {
            "status": "success",
            "user_query": query,
            "chatbot_answer": answer_text,
            "ui_render_images": ui_images,
            "metadata_debug": {
                "text_chunks_found": len(texts),
                "image_chunks_found": len(images)
            }
        }


# ====================================================================
# ĐOẠN CODE KIỂM THỬ THỰC TẾ LUỒNG VẬN HÀNH
# ====================================================================
if __name__ == "__main__":
    # Khởi chạy bộ khung RAG
    rag_engine = IFABHybridRAGEngine()
    
    # Câu hỏi tình huống luật phức tạp bạn đưa ra làm mẫu
    sample_query = "Cầu thủ đứng dưới thủ môn nhưng đứng trên một hậu vệ đối phương thì có bị phạt việt vị không?"
    
    print("\n" + "="*70)
    print(f"❓ CÂU HỎI TRUY VẤN: {sample_query}")
    print("⏳ Hệ thống đang quét không gian vector và gửi truy vấn tổng hợp...")
    print("="*70 + "\n")
    
    # Thực thi quy trình sinh phản hồi
    response_data = rag_engine.generate_answer(sample_query)
    
    # 1. Mô phỏng in câu trả lời dạng Text trên Khung Chat
    print("🤖 [CHATBOT TRẢ LỜI]:")
    print(response_data["chatbot_answer"])
    print("\n" + "-"*50)
    
    # 2. Mô phỏng Render cấu trúc hình ảnh đi kèm lên UI màn hình
    print("🖼️  [GIAO DIỆN UI RENDER - KHỐI SƠ ĐỒ LUẬT MINH HỌA]:")
    if response_data["ui_render_images"]:
        for i, img in enumerate(response_data["ui_render_images"]):
            print(f"   👉 [Sơ đồ {i+1}]:")
            print(f"      - Sử dụng thẻ: <img src='{img['image_path']}' />")
            print(f"      - Ghi chú: Hình trích xuất từ Trang in số {img['printed_page']} (Khu vực ảnh số {img['region_index']})")
    else:
        print("   ℹ️ Hệ thống không tìm thấy sơ đồ khớp với ngữ cảnh câu hỏi này.")
    print("="*70)