"""
Week 3 Chunking + Local Embedding + FAISS
=========================================
Sử dụng mô hình BAAI/bge-m3 chạy hoàn toàn LOCAL thông qua sentence-transformers.
Không cần API Key, không lo lỗi kết nối mạng hay lỗi endpoint.
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# ====================================================================
# CẤU HÌNH HỆ THỐNG LOCAL
# ====================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 120
SOURCE_FILE   = "Laws_of_the_game_2025_26.pdf"

# Model tối ưu chạy local (Tự động tải về máy trong lần chạy đầu tiên)
LOCAL_MODEL_NAME = "BAAI/bge-m3" 


# ====================================================================
# LOCAL EMBEDDINGS WRAPPER
# ====================================================================
class LocalHuggingFaceEmbeddings(Embeddings):
    """
    Wrapper chạy mô hình Embedding Local tương thích với Interface của LangChain.
    Sử dụng GPU (CUDA) nếu máy bạn có, ngược lại tự động dùng CPU.
    """
    def __init__(self, model_name: str = LOCAL_MODEL_NAME):
        print(f"📥 Đang tải/Khởi tạo mô hình Local: {model_name}...")
        # Tự động kiểm tra phần cứng (Ưu tiên card đồ họa Nvidia nếu có)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Mô hình sẽ chạy trên thiết bị: {device.upper()}")
        
        self.model = SentenceTransformer(model_name, device=device,trust_remote_code=True,          # bge-m3 cần cái này
            cache_folder="./model_cache")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Nhúng danh sách văn bản (Mô hình BGE-M3 sinh ra vector 1024 chiều)
        embeddings = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        # Nhúng câu hỏi truy vấn
        embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()


# ====================================================================
# PROCESSOR (Giữ nguyên logic Hybrid Chunking của bạn)
# ====================================================================
class IFABProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )

    def _meta(self, item: dict, element_type: str) -> dict:
        return {
            "source_file":   SOURCE_FILE,
            "absolute_page": item.get("absolute_page"),
            "printed_page":  str(item.get("printed_page", "")),
            "element_type":  element_type,
        }

    def process_text_chunks(self, json_path: str) -> list[Document]:
        print("📄 Đang băm Text (Hybrid Strategy)...")
        if not os.path.exists(json_path):
            print(f"❌ Không tìm thấy: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        SKIP_TYPES = {"blank", "photo", "cover"}
        docs = []

        for item in data:
            if item.get("page_type") in SKIP_TYPES:
                continue
            content = item.get("content_vi", "").strip()
            if not content:
                continue

            meta = self._meta(item, "text")
            if len(content) <= CHUNK_SIZE:
                docs.append(Document(page_content=content, metadata=meta))
            else:
                for chunk in self.splitter.split_text(content):
                    docs.append(Document(page_content=chunk, metadata=meta))

        print(f"   → Tạo thành công {len(docs)} text chunks.")
        return docs

    def process_image_chunks(self, json_path: str) -> list[Document]:
        print("🖼️  Đang tạo Image Chunks...")
        if not os.path.exists(json_path):
            print(f"❌ Không tìm thấy: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for item in data:
            desc = item.get("ai_contextual_description", "").strip()
            captions = " ".join(item.get("captions_vi", [])) if isinstance(item.get("captions_vi"), list) else str(item.get("captions_vi", ""))
            
            combined_desc = f"{desc} {captions}".strip()
            if not combined_desc:
                continue
                
            docs.append(Document(
                page_content=f"[SƠ ĐỒ/HÌNH ẢNH LUẬT]: {combined_desc}",
                metadata={
                    **self._meta(item, "tactical_diagram"),
                    "image_file":   item.get("image_file", ""),
                    "region_index": item.get("region_index", 1),
                },
            ))

        print(f"   → Tạo thành công {len(docs)} image chunks.")
        return docs


# ====================================================================
# PIPELINE KHỞI CHẠY
# ====================================================================
def run_pipeline(text_file: str, img_file: str, db_name: str):
    text_path = os.path.join(BASE_DIR, text_file)
    img_path  = os.path.join(BASE_DIR, img_file)
    db_path   = os.path.join(BASE_DIR, db_name)

    processor = IFABProcessor()
    all_docs  = (processor.process_text_chunks(text_path) +
                 processor.process_image_chunks(img_path))

    if not all_docs:
        print("⚠️ Không có dữ liệu để xử lý.")
        return

    # Khởi tạo mô hình nhúng Local
    embeddings = LocalHuggingFaceEmbeddings(model_name=LOCAL_MODEL_NAME)

    print(f"\n🚀 Đang tiến hành nhúng {len(all_docs)} chunks hoàn toàn offline...")
    vector_db = FAISS.from_documents(all_docs, embeddings)
    
    # Lưu cơ sở dữ liệu FAISS xuống ổ đĩa local
    vector_db.save_local(db_path)

    print(f"\n✅ HOÀN TẤT HỆ THỐNG LOCAL RAG STORAGE")
    print(f"   Tổng số chunks đã nạp : {len(all_docs)}")
    print(f"   Đường dẫn thư mục DB  : {db_path}/")


# ====================================================================
if __name__ == "__main__":
    run_pipeline(
        text_file="data/dataset_tieng_viet_step1_clean.json",
        img_file ="data/dataset_images_metadata.json",
        db_name  ="faiss_ifab_local_db", # Đổi tên thư mục DB local để phân biệt
    )