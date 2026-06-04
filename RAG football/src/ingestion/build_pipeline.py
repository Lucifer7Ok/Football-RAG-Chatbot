import sys
import warnings
import fitz
import json
import re
import time
import os
import google.generativeai as genai
# THÊM THƯ VIỆN ĐỂ ĐỌC FILE .ENV
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ====================================================================
# CẤU HÌNH HỆ THỐNG & API
# ====================================================================
load_dotenv()

API_KEYS = []
env_key = os.environ.get("GEMINI_API_KEY")

if env_key:
    if "," in env_key:
        API_KEYS.extend([k.strip() for k in env_key.split(",") if k.strip()])
    else:
        API_KEYS.append(env_key.strip())

API_KEYS = list(dict.fromkeys(API_KEYS))

if not API_KEYS:
    raise ValueError("❌ LỖI: Không tìm thấy API Key nào trong file `.env`!")

MODELS = ["gemini-3.1-flash-lite", "gemini-1.5-flash", "gemini-2.5-flash"]
current_key_idx = 0
current_model_idx = 0

sys_instruct = """Bạn là cỗ máy phiên dịch Anh-Việt chuyên ngành Luật Bóng Đá. 
Nhiệm vụ ĐỘC NHẤT của bạn là dịch MỌI THỨ nhận được sang Tiếng Việt. 
TUYỆT ĐỐI KHÔNG ĐƯỢC trả về nguyên văn tiếng Anh. Phải dịch tất cả: Mục lục, Lời tựa, Tên sách, Tuyên bố bản quyền."""

def generate_content_with_fallback(prompt, generation_config=None):
    global current_key_idx, current_model_idx
    total_keys = len(API_KEYS)
    total_models = len(MODELS)
    
    # Định nghĩa số lần thử lại tối đa cho mỗi cặp (Key + Model) trước khi đổi cấu hình
    LOCAL_RETRIES = 3 
    attempts_limit = total_keys * total_models * LOCAL_RETRIES
    
    last_exc = None
    model_try_count = 0  # Biến đếm số lần thử thất bại của model hiện tại
    
    for _ in range(attempts_limit):
        api_key = API_KEYS[current_key_idx]
        model_name = MODELS[current_model_idx]
        try:
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model_name, system_instruction=sys_instruct)
            response = model_instance.generate_content(prompt, generation_config=generation_config)
            return response
        except Exception as e:
            last_exc = e
            model_try_count += 1
            print(f"\n⚠️ Lỗi API (Key Index {current_key_idx+1}, Model {model_name}) - Lần thử {model_try_count}/{LOCAL_RETRIES}: {e}")
            
            # Nếu đã thử quá số lần quy định cho model này hoặc hệ thống có nhiều key để xoay tua
            if model_try_count >= LOCAL_RETRIES or total_keys > 1:
                model_try_count = 0 # Reset bộ đếm
                
                # Tiến hành xoay vòng Key trước
                current_key_idx = (current_key_idx + 1) % total_keys
                
                # Nếu đã lướt qua hết sạch các Key (hoặc chỉ có 1 Key duy nhất), tiến hành đổi Model
                if current_key_idx == 0:
                    current_model_idx = (current_model_idx + 1) % total_models
                    print(f"👉 Tự động chuyển cấu hình sang model: {MODELS[current_model_idx]}")
            
            time.sleep(2) # Nghỉ 2 giây tạo khoảng giãn cách an toàn tránh dồn dập request
            
    raise last_exc

MAX_RETRIES = 5

# Ngưỡng phân loại hình học vật lý (IFAB 2025/26)
DIAGRAM_MIN_COLORED  = 80   
DIAGRAM_MAX_BODY     = 600  
PHOTO_RASTER_RATIO   = 0.80 
PHOTO_MAX_BODY       = 300  # Tăng ngưỡng để hốt thêm các chữ lề ở trang bìa phụ

GLOSSARY = """THUẬT NGỮ BẮT BUỘC:
- Offside / offside offence      → Việt vị / Lỗi việt vị
- Indirect free kick             → Quả phạt gián tiếp
- Direct free kick               → Quả phạt trực tiếp
- Penalty kick                   → Quả phạt đền (phạt 11m)
- Drop ball                      → Bóng thả
- Caution                        → Cảnh cáo (Thẻ vàng)
- Sending-off                    → Truất quyền thi đấu (Thẻ đỏ)
- Interfering with play          → Can thiệp vào tình huống
- Interfering with an opponent   → Cản trở đối thủ
- Goalkeeper                     → Thủ môn
- Assistant referee (AR)         → Trọng tài biên (TTB)
- Throw-in                       → Ném biên
- Kick-off                       → Quả giao bóng
- Foul                           → Phạm lỗi (Lỗi)
- Misconduct                     → Hành vi sai trái (Lỗi hành vi)
- Added time / Allowance for time lost → Thời gian bù giờ
- Extra time                     → Hiệp phụ
- Advantage                      → Phép lợi thế (Luật lợi thế)
- Reckless challenge             → Vào bóng liều lĩnh
- Serious foul play              → Phạm lỗi nghiêm trọng
- Violent conduct                → Hành vi bạo lực
- Handling / Handling the ball   → Lỗi chạm tay (Lỗi dùng tay chơi bóng)"""

# ====================================================================
# MODULE 1: TRÍCH XUẤT HÌNH HỌC VÀ PHÂN LOẠI TRANG
# ====================================================================

def _count_colored_drawings(page):
    return sum(
        1 for d in page.get_drawings()
        if d.get("fill") and d["fill"] not in [(1,1,1), (0,0,0), None]
    )

def _has_fullpage_raster(page):
    pw, ph = page.rect.width, page.rect.height
    for img_info in page.get_images(full=True):
        rects = page.get_image_rects(img_info[0])
        if rects:
            r = rects[0]
            if (r.x1 - r.x0) > pw * PHOTO_RASTER_RATIO and (r.y1 - r.y0) > ph * PHOTO_RASTER_RATIO:
                return True
    return False

def _extract_body_blocks(page, padding=55):  # SỬA LỖI: Thêm tham số lề động linh hoạt
    ph = page.rect.height
    y_min, y_max = padding, ph - padding
    blocks = []
    for b in page.get_text("blocks"):
        text = b[4].strip()
        if not text or not (y_min <= b[1] <= y_max):
            continue
        if re.match(r"^\d{1,3}$", text) or "laws of the game" in text.lower():
            continue
        blocks.append(b)
    return blocks

def classify_page(page):
    # Bước 1: Quét lề rộng (padding=20) để đảm bảo không bỏ sót chữ nhận diện thể loại trang
    broad_blocks = _extract_body_blocks(page, padding=20)
    broad_len = sum(len(b[4].strip()) for b in broad_blocks)
    
    colored_count = _count_colored_drawings(page)
    has_raster    = _has_fullpage_raster(page)

    if has_raster and broad_len < PHOTO_MAX_BODY:
        return "photo", broad_blocks, colored_count, False

    if colored_count >= DIAGRAM_MIN_COLORED and broad_len < DIAGRAM_MAX_BODY:
        return "diagram", broad_blocks, colored_count, False

    if broad_len <= 5: # Hạ ngưỡng trang trống tuyệt đối
        return "blank", [], colored_count, False

    # Bước 2: Nếu là trang văn bản luật thường, quét lại với lề nghiêm ngặt (padding=55) 
    # để dọn sạch hoàn toàn Header/Footer lặp lại
    strict_blocks = _extract_body_blocks(page, padding=55)
    has_mixed_image = (colored_count >= 50) or (len(page.get_images()) > 0)
    
    return "content", strict_blocks, colored_count, has_mixed_image

# ====================================================================
# MODULE 2: XỬ LÝ SỐ TRANG IN VẬT LÝ
# ====================================================================

def extract_printed_page_number(page):
    ph, pw = page.rect.height, page.rect.width
    candidates = []
    try:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0: continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    x0, y0 = span["bbox"][0], span["bbox"][1]
                    text = span["text"].strip()
                    if re.match(r"^\d{1,3}$", text):
                        if (y0 < 50 or y0 > ph - 50) and (x0 < 60 or x0 > pw - 60):
                            candidates.append(text)
    except Exception: pass
    return candidates[0] if candidates else None

# ====================================================================
# MODULE 3: DỊCH THUẬT NĂNG LỰC CAO (CÓ DỌN RÁC PDF)
# ====================================================================

def translate_content_page(raw_text, abs_page, printed_page):
    clean_raw_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_text)

    prompt = f"""Dịch văn bản pháp lý sau từ Trang {abs_page} (Số trang in: {printed_page}) sang tiếng Việt.

{GLOSSARY}

YÊU CẦU ĐỊNH DẠNG VÀ DỊCH THUẬT BẮT BUỘC:
1. KHÔNG GIỮ LẠI TIẾNG ANH: Bạn phải dịch toàn bộ văn bản sang tiếng Việt. Nếu là mục lục (Contents) hoặc bản quyền cũng phải dịch hết. "Laws of the Game" dịch là "Luật Bóng đá".
2. NỐI CÂU GÃY: Tự động ghép các từ bị băm nhỏ do xuống dòng (vd: "off-\\nside" → "offside").
3. Chỉ xuống dòng khi kết thúc đoạn văn hoặc sang mục điều luật mới.
4. Đầu ra: Chỉ trả về đoạn văn dịch Tiếng Việt hoàn chỉnh, không markdown, không giải thích thêm.

Văn bản gốc:
{clean_raw_text}"""
    
    gen_cfg = genai.types.GenerationConfig(temperature=0.1, top_p=0.95)
    for attempt in range(MAX_RETRIES):
        try:
            resp = generate_content_with_fallback(prompt, generation_config=gen_cfg)
            if resp.text:
                return resp.text.strip()
        except Exception as e:
            print(f"\n⚠️ Lỗi dịch nội dung trang {abs_page} (lần thử {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt * 2)
    return clean_raw_text

def translate_captions(captions, abs_page):
    if not captions: return []
    clean_captions = [re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', c) for c in captions]

    prompt = f"""Dịch danh sách nhãn chú thích hoặc tiêu đề bìa sách sau sang tiếng Việt (Trang {abs_page}).
{GLOSSARY}

BẮT BUỘC: Dịch 100% sang tiếng Việt. Nếu gặp "Laws of the Game" hãy dịch là "Luật Bóng đá".
Trả về một JSON Array chứa các chuỗi đã dịch theo đúng thứ tự 1-1, không giải thích.

Input Array: {json.dumps(clean_captions, ensure_ascii=False)}"""

    gen_cfg = genai.types.GenerationConfig(temperature=0.1, response_mime_type="application/json")
    for attempt in range(MAX_RETRIES):
        try:
            resp = generate_content_with_fallback(prompt, generation_config=gen_cfg)
            resp_text = resp.text.strip()
            if resp_text.startswith("```"):
                resp_text = re.sub(r"^```(?:json)?\s*", "", resp_text)
                resp_text = re.sub(r"\s*```$", "", resp_text)
            resp_text = resp_text.strip()
            result = json.loads(resp_text)
            return result if isinstance(result, list) else clean_captions
        except Exception as e:
            print(f"\n⚠️ Lỗi dịch chú thích trang {abs_page} (lần thử {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt * 2)
    return clean_captions

# ====================================================================
# MODULE 4: QUẢN LÝ TIẾN TRÌNH (ATOMIC CHECKPOINT)
# ====================================================================

def load_checkpoint(path):
    tmp = path + ".tmp"
    for p in [path, tmp]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    return d, {x["absolute_page"] for x in d}
            except Exception: pass
    return [], set()

def save_checkpoint(path, dataset):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"Lỗi checkpoint: {e}")

# ====================================================================
# TRỤC CHẠY CHÍNH PIPELINE
# ====================================================================

def run_pipeline(pdf_path, output_json_path):
    doc = fitz.open(pdf_path)
    total = len(doc)
    dataset, processed = load_checkpoint(output_json_path)

    print(f"🚀 BẮT ĐẦU CHẠY PIPELINE V3 TỐI ƯU | Đã xong: {len(processed)}/{total}")

    for idx in range(total):
        abs_page = idx + 1
        if abs_page in processed: continue

        page = doc[idx]
        page_type, body_blocks, colored_count, has_mixed_image = classify_page(page)
        printed_raw = extract_printed_page_number(page)
        printed_page = printed_raw if printed_raw else str(abs_page)

        print(f"[{abs_page:3d}/{total}] Loại: {page_type:8s} | Số trang in: {printed_page:4s}", end="", flush=True)

        page_data = {
            "absolute_page": abs_page,
            "printed_page": printed_page,
            "page_type": page_type,
            "has_mixed_image": has_mixed_image
        }

        if page_type == "blank":
            page_data["content_vi"] = ""
            print(" | (Trang trống)")

        elif page_type in ["photo", "diagram"]:
            raw_blocks = [b[4].strip() for b in body_blocks if b[4].strip()]
            captions_raw = [text for text in raw_blocks if "practical guidelines" not in text.lower()]
            captions_vi = translate_captions(captions_raw, abs_page)
            
            page_data["captions_raw"] = captions_raw
            page_data["captions_vi"] = captions_vi
            page_data["content_vi"] = "\n".join(captions_vi)
            print(f" | Dịch xong {len(captions_raw)} captions.")
            time.sleep(1.0)

        else: # SỬA LỖI: Loại 'content' sẽ lấy văn bản an toàn không lo rỗng biến đổi
            raw_text = "\n".join(b[4] for b in sorted(body_blocks, key=lambda x: (x[1], x[0])) if b[4].strip())
            if not raw_text.strip():
                # Nếu quét lề hẹp bị rỗng, cứu vớt bằng cách quét lại lề rộng (padding=20)
                backup_blocks = _extract_body_blocks(page, padding=20)
                raw_text = "\n".join(b[4] for b in sorted(backup_blocks, key=lambda x: (x[1], x[0])) if b[4].strip())
            
            if not raw_text.strip():
                page_data["page_type"] = "blank"
                page_data["content_vi"] = ""
                print(" | (Rỗng hoàn toàn)")
            else:
                translated = translate_content_page(raw_text, abs_page, printed_page)
                translated = translated.replace("\\n", "\n").replace("\\\\n", "\n")
                page_data["content_vi"] = re.sub(r"\n{3,}", "\n\n", translated)
                print(f" | Dịch xong văn bản chính. Cờ ảnh hỗn hợp: {has_mixed_image}")
                time.sleep(5)

        dataset.append(page_data)
        save_checkpoint(output_json_path, dataset)

    print("\n🎉 HOÀN THÀNH XUẤT SẮC PIPELINE V3!")

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    run_pipeline(
        pdf_path=os.path.join(base, "Laws of the Game 2025_26_single pages.pdf"),
        output_json_path=os.path.join(base, "dataset_tieng_viet_step1.json")
    )