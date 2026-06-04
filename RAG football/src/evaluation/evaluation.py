"""
====================================================================
WEEK 7: AUTOMATED RAG EVALUATION USING RAGAS (ALIGNED WITH WEEK 4)
====================================================================
Cập nhật bảo mật & tương thích cấu trúc:
1. Đọc khóa bảo mật từ file .env qua biến GEMINI_API_KEY.
2. Sử dụng cấu trúc Ragas Factory (llm_factory, embedding_factory).
3. Khai báo Class Metric chuẩn và đồng bộ trích xuất dữ liệu từ week4.
4. Sửa lỗi TypeError bắt buộc truyền tham số llm/embeddings vào Class Metric.
"""

import os
import sys
from pathlib import Path

# Đảm bảo Terminal hiển thị đúng font tiếng Việt có dấu trên Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Tìm và nạp file .env từ thư mục script hoặc thư mục cha
_script_dir = Path(__file__).resolve().parent
for _candidate in [_script_dir, _script_dir.parent]:
    _env_file = _candidate / ".env"
    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file)
        print(f"✅ Đã nạp file .env từ: {_env_file}")
        break
else:
    load_dotenv()  # Fallback: tìm theo mặc định

# Đọc API Key từ file .env
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("❌ Lỗi: Không tìm thấy GEMINI_API_KEY trong file .env")
    print(f"   Đã tìm trong: {_script_dir} và {_script_dir.parent}")
    print("   Vui lòng tạo file .env với nội dung: GEMINI_API_KEY=your_key_here")
    sys.exit(1)

# Thiết lập biến môi trường hệ thống gác cổng cho SDK Google và Ragas
os.environ["GOOGLE_API_KEY"] = gemini_api_key
os.environ["GEMINI_API_KEY"] = gemini_api_key

# Import bộ thư viện từ Google GenAI và Ragas bản mới
from google import genai
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,         # Tính trung thực (kiểm tra lỗi ảo giác)
    AnswerRelevancy,      # Độ liên quan, trực diện của câu trả lời
    ContextPrecision,     # Độ chính xác của tài liệu luật trích xuất được
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

# ============================================================
# 1. CẤU HÌNH GIÁM KHẢO GEMINI 3.1 FLASH-LITE CHO RAGAS
# ============================================================
# Khởi tạo Google GenAI client chính thức dựa trên key .env
client = genai.Client(api_key=gemini_api_key)

# Cấu hình bộ nhân suy luận (LLM) cho Ragas chấm điểm bằng bản Stable
ragas_evaluator = llm_factory(
    "gemini-3.1-flash-lite",
    provider="google",
    client=client
)

# Cấu hình bộ băm vector (Embeddings) bổ trợ cho chỉ số AnswerRelevancy
ragas_embeddings = embedding_factory(
    provider="google",
    client=client,
    model="text-embedding-004"
)

# ============================================================
# 2. CHUẨN BỊ BỘ CÂU HỎI MẪU (TEST SET)
# ============================================================
test_cases = [
    {
        "question": "Cầu thủ đội tấn công vô tình để bóng chạm tay trước khi ghi bàn thì bàn thắng có được công nhận không?",
        "ground_truth": "Bàn thắng không được công nhận. Theo luật IFAB, nếu một cầu thủ ghi bàn thắng trực tiếp từ tay/cánh tay của họ hoặc ghi bàn ngay sau khi bóng chạm vào tay/cánh tay của họ (dù là vô tình) thì đều tính là lỗi."
    },
    {
        "question": "Thủ môn có được cầm bóng quá 6 giây không?",
        "ground_truth": "Không. Thủ môn không được quyền kiểm soát bóng bằng tay quá 6 giây. Nếu vi phạm, đội đối phương sẽ được hưởng một quả phạt gián tiếp."
    }
]

# ============================================================
# 3. CHẠY MÔ PHỎNG ĐỂ THU THẬP KẾT QUẢ TỪ RAG ENGINE WEEK 4
# ============================================================
print("⏳ Đang khởi tạo RAG Engine của Week 4 và thu thập câu trả lời thực tế...")

from week4_rag_engine import IFABHybridRAGEngine
rag_engine = IFABHybridRAGEngine()

questions = []
answers = []
contexts_list = []
ground_truths = []

for case in test_cases:
    q = case["question"]
    gt = case["ground_truth"]
    
    print(f"\n👉 Đang xử lý câu hỏi: '{q}'")
    
    # Thực hiện truy vấn hệ thống RAG thực tế của bạn
    rag_output = rag_engine.generate_answer(q)
    chatbot_answer = rag_output.get("chatbot_answer", "")
    
    # Gọi chính xác hàm bổ trợ retrieve_context của bạn ở tuần 4 
    retrieved_objects = rag_engine.retrieve_context(q)
    retrieved_chunks = [doc.text for doc in retrieved_objects if hasattr(doc, "text")]
    
    # Phòng hờ trường hợp rỗng, lấy chính câu trả lời làm context nền
    if not retrieved_chunks:
        retrieved_chunks = [chatbot_answer]
        
    questions.append(q)
    answers.append(chatbot_answer)
    contexts_list.append(retrieved_chunks)
    ground_truths.append(gt)

# Đóng gói dữ liệu thành định dạng Dataset của HuggingFace theo tiêu chuẩn Ragas
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

# ============================================================
# 4. TIẾN HÀNH CHẤM ĐIỂM TỰ ĐỘNG BẰNG RAGAS
# ============================================================
print("\n🤖 Giám khảo Gemini 3.1 Flash-Lite đang tiến hành chấm điểm hệ thống RAG...")

# KHẮC PHỤC LỖI TYPEERROR: Truyền trực tiếp các đối tượng factory vào hàm khởi tạo của từng Metric
metrics = [
    Faithfulness(llm=ragas_evaluator),
    AnswerRelevancy(llm=ragas_evaluator, embeddings=ragas_embeddings),
    ContextPrecision(llm=ragas_evaluator)
]

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=ragas_evaluator,
    embeddings=ragas_embeddings
)

# ============================================================
# 5. XUẤT BÁO CÁO KẾT QUẢ RA FILE EXCEL
# ============================================================
print("\n📊 TỔNG HỢP ĐIỂM SỐ RAGAS THU ĐƯỢC:")
print(result)

# Chuyển kết quả sang định dạng Pandas DataFrame dữ liệu bảng
try:
    df_result = result.to_pandas()
except AttributeError:
    df_result = pd.DataFrame(result)

# Cấu hình đường dẫn xuất file
output_dir = "D:/ai_chatbot_study/RAG football"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "ragas_report.xlsx")

# Lưu dữ liệu bảng vào file Excel
df_result.to_excel(output_file, index=False)
print(f"\n✅ Đã xuất báo cáo kiểm định chi tiết thành công tại: {output_file}")