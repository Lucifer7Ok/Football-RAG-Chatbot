## 🗺️ Lộ trình Phát triển Hệ thống (7-Week Pipeline)

Dưới đây là chi tiết kiến trúc và tiến độ thực hiện hệ thống Football RAG Chatbot qua 7 tuần nghiên cứu và triển khai:

| Tuần | Tên Giai Đoạn | Công Nghệ Cốt Lõi | Đầu Ra Chính |
| :--- | :--- | :--- | :--- |
| **Week 1** | Phân loại & Dịch thuật | PyMuPDF, Gemini API | JSON cấu trúc sách luật sạch |
| **Week 2** | Trích xuất & Làm giàu Ảnh | `page.get_image_rects()`, Gemini Vision | Kho sơ đồ sa bàn kèm Metadata |
| **Week 3** | Phân mảnh & Vector DB | LangChain, text-embedding-004, FAISS | Kiến trúc Hybrid Vector Index |
| **Week 4** | Xây dựng RAG Engine | Hybrid Multi-Modal Retriever | Prompt Assembly & Generation |
| **Week 5** | Production API Server | FastAPI, Lifespan, Semaphore, GPU Optimization | API Production v2.2.0 ổn định |
| **Week 6** | Giao diện Người dùng | Streamlit / Next.js, TailwindCSS | Multi-modal Chat UI Streaming |
| **Week 7** | Kiểm thử & Đánh giá | Ragas Framework, Gemini 3.1 Flash-Lite | Bảng điểm Excel kiểm định chất lượng |

---

### 📋 Chi Tiết Kỹ Thuật Qua Từng Tuần

### 🔹 TUẦN 1: Phân Loại Trang & Dịch Thuật Văn Bản Chuẩn Cấu Trúc
* **Mục tiêu:** Quét toàn bộ file PDF gốc để phân loại thành 4 loại trang cấu trúc: `content`, `diagram`, `photo`, `blank`.
* **Điểm nhấn kỹ thuật:** * Xử lý dịch thuật mượt mà, nối câu tự nhiên không bị ngắt quãng bởi các thẻ ảnh nằm xen kẽ.
    * Bổ sung cờ hiệu định danh `has_mixed_image` trên các trang `content` để làm điều kiện kích hoạt cho pipeline xử lý ảnh ở Tuần 2.

### 🔹 TUẦN 2: Trích Xuất Tọa Độ, Cắt Ảnh & Làm Giàu Ngữ Cảnh (Vision Grounding)
* **Mục tiêu:** Quét mục tiêu chuẩn xác, chỉ lao vào xử lý những trang được đánh nhãn là `diagram`, `photo` hoặc trang `content` có cờ `has_mixed_image == True`.
* **Các bước thực hiện:**
    1.  **Xác định Bounding Box vật lý:** Kết hợp hàm `page.get_image_rects()` và gom cụm tọa độ khối `page.get_drawings()` để định vị vùng chứa hình ảnh $(x_0, y_0, x_1, y_1)$.
    2.  **Cắt ảnh vật lý chất lượng cao:** Sử dụng thuộc tính Matrix phóng đại của PyMuPDF nhằm tăng độ phân giải lên gấp 3 lần chống mờ nét:
        ```python
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
        pix.save(f"cropped_images/trang_{printed_page}_anh_{counter}.jpg")
        ```
    3.  **Làm giàu ngữ cảnh (Contextual Grounding):** Gửi sơ đồ kết hợp đoạn văn bản Tiếng Việt (`content_vi`) lên mô hình Gemini Vision để sinh tóm tắt luật tự động.
* **Đầu ra:** Thư mục ảnh sơ đồ sa bàn sạch và file `dataset_images_metadata.json` liên kết tên file với mô tả ngữ cảnh tương ứng.

### 🔹 TUẦN 3: Phân Mảnh Chiến Thuật (Chunking) & Lưu Trữ Vector Database
* **Mục tiêu:** Biến đổi văn bản thô thành các đơn vị tri thức nhỏ gọn (Chunks) tối ưu, ngăn chặn tình trạng cắt đôi câu giữa chừng.
* **Các bước thực hiện:**
    * **Text Chunking:** Sử dụng `RecursiveCharacterTextSplitter` băm văn bản luật theo dấu chấm câu hoặc mục điều luật (`chunk_size=600`, `chunk_overlap=120`).
    * **Image Metadata Chunking:** Tạo các khối chunk đại diện cho hình ảnh dựa trên đoạn mô tả ngữ cảnh do Gemini Vision sinh ra ở Tuần 2 kèm nhãn chú thích (`captions_vi`).
    * **Gắn Siêu dữ liệu cố định (Hard Metadata):** Đính kèm thuộc tính cha bắt buộc:
        ```json
        {
          "source_file": "Laws_of_the_game_2025_26.pdf",
          "absolute_page": 45,
          "printed_page": "43",
          "element_type": "text" // hoặc "tactical_diagram"
        }
        ```
    * **Vector hóa:** Sử dụng mô hình `text-embedding-004` để nhúng toàn bộ kho chữ và kho ảnh vào Vector Database (**FAISS**).

### 🔹 TUẦN 4: Xây Dựng RAG Engine & Bộ Truy Xuất Đa Phương Thức (Hybrid Multi-Modal Retriever)
* **Mục tiêu:** Tiếp nhận câu hỏi của người dùng, tìm kiếm tri thức thông minh đồng thời trên cả kho chữ và kho sơ đồ, trả về câu trả lời tự nhiên kèm ảnh minh họa.
* **Luồng vận hành:**
    ```text
    [User Question] ──> [Embedding] ──> [FAISS Hybrid Retrieval] 
                                              │
                                              ├──> Top 3 Text Chunks (Văn bản luật)
                                              └──> Top 2 Image Chunks (Sơ đồ sa bàn)
                                              │
    [Final Answer + Images] <── [Gemini 3.1 Flash-Lite] <── [Context Assembly Prompt]
    ```

### 🔹 TUẦN 5: Thiết Lập API Server Chuyên Nghiệp Trên Production (FastAPI Multi-Modal Server)
* **Mục tiêu:** Đóng gói RAG Engine thành dịch vụ API chất lượng cao (v2.2.0). Xử lý triệt để lỗi nghẽn hệ thống (**Race Condition**), tối ưu hóa bộ nhớ phần cứng (RAM/VRAM).
* **Giải pháp mã nguồn tích hợp:**
    * **Sửa lỗi sập 503/Treo máy ở request đầu tiên:** Sử dụng sự kiện `lifespan` kết hợp cờ trạng thái toàn cục `asyncio.Event()` (`ready_event`). Đưa tiến trình khởi tạo RAG Engine vào luồng riêng biệt bằng `asyncio.to_thread` nhằm giữ Event Loop luôn tự do.
    * **Quản lý tài nguyên phần cứng (Concurrency):** Đặt giới hạn tác vụ đồng thời thông qua bộ đếm `asyncio.Semaphore(2)` để ngăn chặn tràn bộ nhớ GPU (Out of VRAM).
    * **Cơ chế ngắt tự động (Timeout):** Bọc hàm suy luận trong `asyncio.wait_for(..., timeout=90)` để tự động hủy tác vụ bị đóng băng quá 90 giây và trả về mã lỗi chuẩn `504 Gateway Timeout`.
    * **Giải phóng bộ nhớ chủ động:** Kích hoạt trình thu gom rác `gc.collect()` và dọn dẹp phân mảnh đồ họa `torch.cuda.empty_cache()` ngay sau khi giải phóng tác vụ hoặc khi tắt hệ thống (SHUTDOWN).
* **Hệ thống Endpoints:**
    * `GET /`: Health Check (Theo dõi thiết bị CPU/GPU, dung lượng VRAM đang bị chiếm dụng).
    * `GET /api/chat/ready`: Readiness Probe (Frontend dùng để Polling kiểm tra trạng thái tải mô hình, trả về `200` nếu sẵn sàng hoặc `503` kèm header `Retry-After: 5`).
    * `POST /api/chat`: Chat Endpoint tiếp nhận câu hỏi qua Schema `ChatRequest` (Giới hạn từ 1 đến 500 ký tự).

### 🔹 TUẦN 6: Xây Dựng Giao Diện Người Dùng (Frontend UI)
* **Mục tiêu:** Thiết kế giao diện Web trực quan, hỗ trợ hiển thị câu trả lời dạng streaming kết hợp kết xuất sa bàn bóng đá đi kèm.
* **Các bước thực hiện:**
    * **Công nghệ:** Triển khai nhanh ứng dụng Web Dashboard trực quan bằng thư viện **Streamlit**.
    * **Khung hiển thị Đa phương thức:** Thiết kế các bong bóng hội thoại (`Chat bubbles`) mượt mà. Khi API trả về mảng danh sách hình ảnh sơ đồ (`source_images`), hệ thống tự động render các thẻ ảnh (`Image Cards`) ở vùng mở rộng "Sơ đồ minh họa trực quan".
    * **Hứng luồng dữ liệu:** Tích hợp bộ đọc luồng text đổ về từ FastAPI, hiển thị hiệu ứng `skeleton loading` cho hình ảnh trong lúc chờ API hoàn tất truy xuất.

### 🔹 TUẦN 7: Kiểm Thử Tự Động & Đánh Giá (Ragas Evaluation)
* **Mục tiêu:** Đo lường chính xác chất lượng câu trả lời của Chatbot dựa trên tiêu chuẩn khoa học thay vì thử nghiệm thủ công bằng mắt.
* **Các bước thực hiện:**
    1.  **Xây dựng Tập Kiểm Thử (Ground Truth Dataset):** Thiết lập file `test_dataset.json` chứa từ 30 - 50 câu hỏi tình huống luật thực tế kèm câu trả lời chuẩn trích xuất từ sách luật IFAB.
    2.  **Chạy Pipeline lấy dữ liệu đánh giá:** Đẩy tự động tập test qua API Server để thu thập đủ 4 yếu tố: `question`, `contexts`, `answer`, và `ground_truth`.
    3.  **Cấu hình chỉ số đo lường Ragas (Target > 0.85):**
        * *Faithfulness (Độ trung thực):* Kiểm tra lỗi ảo giác (Hallucination), đảm bảo LLM chỉ nói dựa trên luật được cung cấp.
        * *Answer Relevance (Độ liên quan):* Đánh giá câu trả lời có đi đúng vào trọng tâm câu hỏi.
        * *Context Precision & Recall:* Đo lường thuật toán tìm kiếm vector đã bốc đúng và đủ các điều luật, sơ đồ cần thiết chưa.
* **Đầu ra:** Bảng điểm tổng hợp tự động xuất ra file Excel (`ragas_report.xlsx`) để định hướng tối ưu lại tham số hệ thống.
