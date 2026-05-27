import json
import re
import os

def normalize_law_newlines(text):
    if not text or not isinstance(text, str):
        return text
        
    # Bước 1: Chuẩn hóa khoảng trắng ngang (tab, khoảng trắng thừa) về khoảng trắng đơn
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Bước 2: Đánh dấu bảo vệ các phân đoạn lớn (\n\n)
    text = re.sub(r'\n\s*\n', '[[DOUBLE_NEWLINE]]', text)
    
    # Bước 3: Đánh dấu bảo vệ các cấu trúc danh sách điều luật của IFAB
    # Giữ các dấu \n đứng trước mục số (1., 2a.), dấu gạch đầu dòng (•, –,  )
    text = re.sub(r'\n\s*(•|–|\d+[a-z]?\.)', r'[[STRUCTURE_NEWLINE]]\1', text)
    
    # Bước 4: Xóa bỏ toàn bộ các dấu \n lỗi đơn lẻ còn lại ở giữa câu và thay bằng khoảng trắng
    text = text.replace('\n', ' ')
    
    # Bước 5: Khôi phục lại các định dạng cấu trúc chuẩn ban đầu
    text = text.replace('[[DOUBLE_NEWLINE]]', '\n\n')
    text = text.replace('[[STRUCTURE_NEWLINE]]', '\n')
    
    # Bước 6: Dọn dẹp các khoảng trắng kép sinh ra vô ý trong quá trình gộp dòng
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def process_dataset_cleanup(input_json_path, output_json_path):
    if not os.path.exists(input_json_path):
        print(f"❌ Không tìm thấy file dữ liệu đầu vào: {input_json_path}")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"🔄 Bắt đầu tối ưu dòng chảy văn bản cho {len(dataset)} trang...")
    
    cleaned_count = 0
    for item in dataset:
        # Xử lý chuẩn hóa chuỗi văn bản chính trong trường content_vi
        if "content_vi" in item and item["content_vi"]:
            old_text = item["content_vi"]
            new_text = normalize_law_newlines(old_text)
            item["content_vi"] = new_text
            if old_text != new_text:
                cleaned_count += 1
                
        # Xử lý chuẩn hóa tương tự cho danh sách captions_vi (nếu có)
        if "captions_vi" in item and isinstance(item["captions_vi"], list):
            item["captions_vi"] = [normalize_law_newlines(cap) for cap in item["captions_vi"] if cap]

    # Ghi đè hoặc lưu thành file sạch mới để chuẩn bị bàn giao cho Tuần 2
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 Hoàn thành! Đã sửa lỗi ngắt câu trên {cleaned_count} trang.")
    print(f"💾 File dữ liệu chuẩn cấu trúc được lưu tại: '{output_json_path}'")

if __name__ == "__main__":
    # Lấy đường dẫn thư mục hiện tại của bạn
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    
    input_file = os.path.join(base_dir, "dataset_tieng_viet_step1.json")
    # Khuyến khích xuất ra một file trung gian sạch (clean) để làm dữ liệu nền chuẩn cho Tuần 2
    output_file = os.path.join(base_dir, "dataset_tieng_viet_step1_clean.json")
    
    process_dataset_cleanup(input_file, output_file)