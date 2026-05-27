"""
Week 2 Vision Pipeline — RAG Luật Bóng Đá IFAB 2025/26
=======================================================
"""

# TỰ ĐỘNG NẠP THÔNG TIN CẤU HÌNH TỪ FILE .ENV
from dotenv import load_dotenv
import fitz
import json
import os
import time
import sys
import re
import warnings

load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ====================================================================
# CẤU HÌNH (ĐÃ TỐI ƯU CHO 1 KEY)
# ====================================================================
API_KEYS: list[str] = []
_env = os.environ.get("GEMINI_API_KEY")

# SỬA LỖI: Điền chính xác key từ file .env vào danh sách hệ thống, loại bỏ key cứng
if _env:
    if "," in _env:
        API_KEYS.extend([k.strip() for k in _env.split(",") if k.strip()])
    else:
        API_KEYS.append(_env.strip())

API_KEYS = list(dict.fromkeys(API_KEYS))  # dedup, giữ thứ tự

if not API_KEYS:
    raise ValueError("❌ LỖI: Không tìm thấy API Key nào! Hãy chắc chắn bạn đã tạo file `.env` "
                     "chứa biến `GEMINI_API_KEY=your_key` nằm cùng thư mục với file code.")

MODEL_NAME   = "gemini-3.1-flash-lite"    # dùng flash để tận dụng vision
MAX_RETRIES  = 4
MIN_INTERVAL = 7.0  # giây tối thiểu giữa các request (free tier: 10 req/min)


# ====================================================================
# MODULE 1: GEMINI CLIENT — Hỗ trợ Local Retry thông minh cho 1 Key
# ====================================================================

class GeminiClient:
    _system_instruction = """
    Bạn là chuyên gia Luật Bóng đá IFAB 2025/26.
    1. TUYỆT ĐỐI KHÔNG nhắc đến Futsal, bóng đá phong trào trừ khi hình ảnh minh họa rõ ràng về chủ đề đó. 
    2. Luôn bám sát Luật Bóng đá sân 11 người. Nếu tài liệu so sánh luật cũ/mới, BẮT BUỘC ưu tiên giải thích theo luật MỚI nhất (cập nhật từ 1/7/2025).
    3. Tập trung vào: vị trí cầu thủ, ký hiệu hình học, tín hiệu trọng tài và kết luận luật.
    4. Trả lời ngắn gọn bằng tiếng Việt (2-4 câu). KHÔNG dùng Markdown, không xuống dòng thừa.
    """

    def __init__(self, keys: list, model: str):
        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
        except ImportError:
            raise ImportError("pip install google-genai")

        self._keys       = keys
        self._model      = model
        self._exhausted  = set()
        self._cur_idx    = 0
        self._client     = self._new_client()
        self._last_ts    = 0.0

    def _new_client(self):
        return self._genai.Client(api_key=self._keys[self._cur_idx])

    def _rotate(self) -> bool:
        """Rotate sang key chưa exhausted. Trả False nếu hết sạch."""
        total_keys = len(self._keys)
        if total_keys <= 1:
            return False # Không có key khác để xoay tua
            
        for _ in range(total_keys):
            self._cur_idx = (self._cur_idx + 1) % total_keys
            key = self._keys[self._cur_idx]
            if key not in self._exhausted:
                self._client = self._new_client()
                print(f"\n  🔑 Rotate → key ...{key[-6:]}")
                return True
        return False

    def _throttle(self):
        elapsed = time.time() - self._last_ts
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

    def describe(self, image_path: str, prompt: str) -> str:
        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        total_keys = len(self._keys)
        # Định nghĩa số lần thử lại cục bộ cho mỗi Key trước khi chịu thua hoặc đổi key
        LOCAL_RETRIES = 3 
        
        for attempt in range(MAX_RETRIES):
            # Kiểm tra nếu chỉ có 1 key và key đó đã bị đánh dấu cạn kiệt, vẫn cho phép chạy tiếp để thử lại
            if total_keys > 1 and not [k for k in self._keys if k not in self._exhausted]:
                print("\n  🚫 Tất cả key hết daily quota.")
                return ""

            self._throttle()
            
            # Thực hiện vòng lặp thử lại cục bộ tránh crash do lag mạng
            for local_attempt in range(1, LOCAL_RETRIES + 1):
                try:
                    resp = self._client.models.generate_content(
                        model=self._model,
                        contents=[
                            self._types.Part.from_bytes(data=img_bytes, mime_type=mime),
                            self._types.Part.from_text(text=prompt),
                        ],
                        config=self._types.GenerateContentConfig(
                            system_instruction=self._system_instruction
                        )
                    )
                    self._last_ts = time.time()
                    return resp.text.strip() if resp.text else ""

                except Exception as e:
                    err = str(e)
                    self._last_ts = time.time()
                    cur_key = self._keys[self._cur_idx]
                    
                    print(f"\n  ⚠️ Lỗi API (Key ...{cur_key[-6:]}) - Lần thử {local_attempt}/{LOCAL_RETRIES}: {err}")

                    # Kiểm tra lỗi cạn kiệt hạn mức hàng ngày (Daily Quota)
                    if ("RESOURCE_EXHAUSTED" in err or "quota_id" in err 
                            or "GenerateRequestsPerDay" in err):
                        
                        if total_keys > 1:
                            print(f"\n  🚫 Key ...{cur_key[-6:]} cạn hạn mức. Đổi key...")
                            self._exhausted.add(cur_key)
                            if self._rotate():
                                break # Thoát vòng lặp local để dùng cấu hình key mới
                            return ""
                        else:
                            # Nếu chỉ có 1 key, chờ giãn cách tăng cường (Exponential Backoff) rồi thử lại
                            wait_time = 30 * local_attempt
                            print(f"  ⏳ Hệ thống chỉ có 1 Key. Chờ {wait_time}s trước khi thử lại...")
                            time.sleep(wait_time)
                            continue

                    # Kiểm tra lỗi giới hạn băng thông tạm thời (RPM / Rate limit)
                    if "429" in err or "RATE_LIMIT" in err:
                        wait = min(20 * (2 ** local_attempt), 90)
                        print(f"  ⏳ Chạm giới hạn Rate Limit. Nghỉ {wait}s...")
                        time.sleep(wait)
                        continue

                    # Các lỗi kết nối hoặc lỗi HTTP khác
                    if local_attempt < LOCAL_RETRIES:
                        time.sleep(5)
                        
        return ""


# ====================================================================
# MODULE 2: FILTER — xác định trang có giá trị RAG thực sự
# ====================================================================

def has_visual_rule_content(entry: dict, page: "fitz.Page") -> tuple[bool, str]:
    """
    Xác định trang có visual content đáng gọi Vision API không.

    Logic:
    - diagram            → True (luôn xử lý)
    - content+mixed_img  → True (sơ đồ nhúng trong trang luật)
    - photo              → chỉ True khi có diagram vector chồng lên ảnh nền
                           (colored_drawings >= 60) hoặc là trang rule illustration
                           rõ ràng (raster full-page + nhiều captions mô tả luật)
    - blank / photo-cover → False
    """
    pt      = entry.get("page_type", "")
    mixed   = entry.get("has_mixed_image", False)
    caps    = entry.get("captions_raw", [])
    cvi     = entry.get("content_vi", "").strip()

    junk_keywords = ["thay đồ", "trình bày", "chứng nhận", "ifa quality"]
    text_content = entry.get("content_vi", "").lower()
    if any(k in text_content for k in junk_keywords) and not mixed:
        return False, "junk_image"

    if pt == "blank":
        return False, "blank"

    if pt == "diagram":
        return True, "diagram"

    if pt == "content":
        return (True, "content_with_diagram") if mixed else (False, "content_no_image")

    if pt == "photo":
        # Tiêu chí ảnh photo có giá trị RAG thực sự:
        #   (captions >= 3 VÀ content_vi >= 60 chars) HOẶC content_vi >= 80 chars
        #
        # Lý do:
        # - p47 (goalpost diagram): caps=8, cvi=147 → KEEP
        # - p85 (AAR goal signal): caps=3, cvi=105 → KEEP
        # - p98 (goal line tech):  caps=5, cvi=105 → KEEP
        # - p1,p5 (front cover):   caps=1-2, cvi=18-20 → SKIP
        # - p40,p54 (Law covers):  caps=1, cvi=7 → SKIP
        # - p186 (section cover):  caps=3, cvi=33 → SKIP (cvi<60)
        n_caps  = len(caps)
        cvi_len = len(cvi)
        if (n_caps >= 3 and cvi_len >= 60) or cvi_len >= 80:
            return True, "photo_with_rule_content"
        return False, "photo_no_rule_content"

    return False, "unknown"


# ====================================================================
# MODULE 3: CROP ẢNH — pixel analysis (detect cả raster lẫn vector)
# ====================================================================

def find_visual_regions(page: "fitz.Page", dpi: int = 150) -> list:
    """
    Tìm vùng diagram/ảnh bằng pixel analysis.
    Hoạt động với mọi loại page (raster, vector, mixed).

    Thuật toán:
    1. Rasterize trang ở DPI thấp
    2. Với mỗi row: bỏ qua nếu là paragraph text (>= 40 chars) hoặc header/footer
    3. Kiểm tra row có pixel màu (không phải trắng) → đánh dấu là content
    4. Merge rows liền kề thành regions, filter quá nhỏ

    Lưu ý: KHÔNG exclude text ngắn (<40 chars) vì đó thường là
    diagram labels nằm bên trong ảnh (GK, Offside, Advantage...)
    """
    ph, pw = page.rect.height, page.rect.width

    # Chỉ exclude đoạn văn dài (>= 40 chars) — không exclude label ngắn trong diagram
    text_y: set[int] = set()
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        block_text = "".join(
            s["text"] for ln in block.get("lines", [])
            for s in ln.get("spans", [])
        )
        if len(block_text.strip()) < 40:
            continue  # label ngắn → bỏ qua, không exclude
        for ln in block.get("lines", []):
            y0, y1 = int(ln["bbox"][1]), int(ln["bbox"][3])
            text_y.update(range(y0, y1 + 1))

    pix   = page.get_pixmap(dpi=dpi)
    scale = dpi / 72.0
    samp  = pix.samples
    nc    = pix.n
    MGN   = int(25 * scale)
    WHT   = 240

    rows: list[int] = []
    for row in range(pix.height):
        py = int(row / scale)
        if py < 55 or py > ph - 55:        # header/footer zone
            continue
        if py in text_y:                    # paragraph text zone
            continue
        for col in range(MGN, pix.width - MGN, 2):
            idx = (row * pix.width + col) * nc
            if idx + 2 >= len(samp):
                continue
            r, g, b = samp[idx], samp[idx+1], samp[idx+2]
            if r < WHT or g < WHT or b < WHT:
                rows.append(py)
                break

    if not rows:
        return []

    GAP, MIN_H = 12, 50
    regs: list[tuple] = []
    start = rows[0]
    prev  = rows[0]
    for y in rows[1:]:
        if y - prev > GAP:
            if prev - start >= MIN_H:
                regs.append((start, prev))
            start = y
        prev = y
    if prev - start >= MIN_H:
        regs.append((start, prev))

    PAD = 5
    return [
        fitz.Rect(PAD, max(0, y0 - PAD), pw - PAD, min(ph, y1 + PAD))
        for y0, y1 in regs
    ]


def crop_region(page: "fitz.Page", rect: "fitz.Rect",
                out_path: str, dpi: int = 200):
    page.get_pixmap(dpi=dpi, clip=rect).save(out_path)


# ====================================================================
# MODULE 4: CHECKPOINT
# ====================================================================

def load_db(path: str) -> tuple[list, set]:
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            db = [x for x in json.load(f) if x.get("ai_contextual_description")]
        processed = {x["image_file"] for x in db}
        print(f"  → Checkpoint: {len(db)} ảnh đã xử lý")
        return db, processed
    except Exception:
        return [], set()


def save_db(path: str, db: list):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ====================================================================
# PIPELINE CHÍNH
# ====================================================================

def run_vision_pipeline(pdf_path, input_json_path, output_json_path, img_dir):
    os.makedirs(img_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    with open(input_json_path, "r", encoding="utf-8") as f:
        step1 = json.load(f)

    db, processed_imgs = load_db(output_json_path)
    client = GeminiClient(API_KEYS, MODEL_NAME)

    # Pre-scan: đếm trang thực sự cần xử lý
    targets = []
    skip_stats: dict[str, int] = {}
    for entry in step1:
        page = doc[entry["absolute_page"] - 1]
        ok, reason = has_visual_rule_content(entry, page)
        if ok:
            targets.append((entry, reason))
        else:
            skip_stats[reason] = skip_stats.get(reason, 0) + 1

    print(f"\n{'='*60}")
    print(f"🚀  Week 2 Vision Pipeline")
    print(f"    Trang cần Vision API: {len(targets)}")
    print(f"    Bỏ qua: {dict(skip_stats)}")
    print(f"    Ước tính thời gian: ~{len(targets) * MIN_INTERVAL // 60:.0f} phút")
    print(f"{'='*60}\n")

    for entry, reason in targets:
        abs_page     = entry["absolute_page"]
        printed_page = entry["printed_page"]
        page_type    = entry["page_type"]

        print(f"[p{abs_page:3d}] {reason:28s} | ", end="", flush=True)

        page   = doc[abs_page - 1]
        rects  = find_visual_regions(page, dpi=150)

        if not rects:
            print("0 regions → skip")
            continue

        print(f"{len(rects)} region(s) → ", end="", flush=True)

        # Context text cho prompt
        context = entry.get("content_vi", "").strip()
        if not context:
            context = " ".join(entry.get("captions_vi", []))
        context = context[:800]

        for idx, rect in enumerate(rects, 1):
            img_name = f"trang_{printed_page}_anh_{idx}.jpg"
            img_path = os.path.join(img_dir, img_name)

            if img_name in processed_imgs:
                print(f"[r{idx}:✓cached] ", end="", flush=True)
                continue

            crop_region(page, rect, img_path, dpi=200)

            prompt = f"""Đây là hình ảnh/sơ đồ từ Trang {printed_page} sách Luật Bóng đá IFAB.

Ngữ cảnh văn bản cùng trang:
{context if context else '(Không có text đi kèm)'}

Nhiệm vụ: Mô tả ngắn gọn bằng tiếng Việt (2-4 câu) hình ảnh này minh hoạ quy tắc/tình huống nào trong luật bóng đá. Tập trung: vị trí cầu thủ, loại tình huống (việt vị/phạm lỗi/hợp lệ/tín hiệu trọng tài), kết luận luật. Không dùng Markdown."""

            desc = client.describe(img_path, prompt)

            if desc:
                db.append({
                    "source_file":              "Laws of the Game 2025_26_single pages.pdf",
                    "image_file":               img_name,
                    "absolute_page":            abs_page,
                    "printed_page":             printed_page,
                    "region_index":             idx,
                    "element_type":             reason,
                    "ai_contextual_description": desc,
                })
                processed_imgs.add(img_name)
                save_db(output_json_path, db)
                print(f"[r{idx}:✓] ", end="", flush=True)
            else:
                print(f"[r{idx}:✗] ", end="", flush=True)

        print()

    print(f"\n{'='*60}")
    print(f"✅  HOÀN THÀNH | {len(db)} ảnh đã xử lý")
    from collections import Counter
    type_stats = Counter(x["element_type"] for x in db)
    for t, c in type_stats.items():
        print(f"   {t}: {c}")
    print(f"   Output: {output_json_path}")
    print(f"   Ảnh:    {img_dir}/")
    print(f"{'='*60}")


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    base = (os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir() else os.getcwd())
    run_vision_pipeline(
        pdf_path         = os.path.join(base, "Laws of the Game 2025_26_single pages.pdf"),
        input_json_path  = os.path.join(base, "dataset_tieng_viet_step1.json"),
        output_json_path = os.path.join(base, "dataset_images_metadata.json"),
        img_dir          = os.path.join(base, "cropped_images"),
    )