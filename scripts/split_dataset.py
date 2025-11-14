import os
import shutil
import glob
from sklearn.model_selection import train_test_split

# 1. CẤU HÌNH CÁC ĐƯỜNG DẪN CỦA BẠN (SỬA 3 DÒNG NÀY)

# ĐƯỜNG DẪN 1: Dán đường dẫn đến thư mục chứa ẢNH "raw" của bạn
RAW_IMAGES_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/data/raw/images" 

# ĐƯỜNG DẪN 2: Dán đường dẫn đến thư mục chứa NHÃN "raw" (.txt) của bạn
RAW_LABELS_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/data/raw/labels"

# ĐƯỜNG DẪN 3: Chọn nơi bạn muốn lưu DATASET MỚI (đã chia)
OUTPUT_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/data/AseanFlags_Dataset_Split" # (Tôi đổi tên output để tránh trùng)

# Tỷ lệ phần trăm cho bộ validation (ví dụ: 0.2 = 20%)
VAL_SIZE = 0.2

# 2. TẠO CẤU TRÚC THƯ MỤC CHO YOLO
train_img_dir = os.path.join(OUTPUT_PATH, "images", "train")
train_label_dir = os.path.join(OUTPUT_PATH, "labels", "train")
val_img_dir = os.path.join(OUTPUT_PATH, "images", "val")
val_label_dir = os.path.join(OUTPUT_PATH, "labels", "val")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

print(f"Đã tạo cấu trúc thư mục tại: {OUTPUT_PATH}")

# 3. THỰC HIỆN CHIA DỮ LIỆU
print(f"Đang quét ảnh trong: {RAW_IMAGES_PATH}")
image_files = []

# Danh sách các đuôi tệp phổ biến và các biến thể
extensions_to_check = [
    "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
    "*.png", "*.PNG", "*.bmp", "*.BMP", "*.webp", "*.WEBP"
]

for ext in extensions_to_check:
    image_files.extend(glob.glob(os.path.join(RAW_IMAGES_PATH, ext)))

image_files = sorted(list(set(image_files))) 
print(f"Tìm thấy tổng cộng {len(image_files)} ảnh.")

if not image_files:
    print("[LỖI] Không tìm thấy ảnh nào. Hãy kiểm tra lại RAW_IMAGES_PATH.")
else:
    # Dùng sklearn để chia danh sách
    train_files, val_files = train_test_split(
        image_files, 
        test_size=VAL_SIZE, 
        random_state=42 # Đảm bảo kết quả chia luôn giống nhau
    )

    print(f"Chia {len(train_files)} tệp cho train.")
    print(f"Chia {len(val_files)} tệp cho val.")

    # 4. HÀM HELPER ĐỂ SAO CHÉP TỆP (Cả Ảnh và Nhãn)
    def copy_files(file_list, img_dest_dir, label_dest_dir, label_source_dir):
        count = 0
        for img_path in file_list:
            try:
                base_filename = os.path.basename(img_path)
                name_part = os.path.splitext(base_filename)[0]
                
                # 1. Sao chép tệp ảnh
                shutil.copy(img_path, img_dest_dir)
                
                # 2. Xây dựng đường dẫn tệp nhãn .txt
                label_filename = name_part + ".txt"
                label_src_path = os.path.join(label_source_dir, label_filename)
                
                # 3. Sao chép tệp nhãn .txt nếu tồn tại
                if os.path.exists(label_src_path):
                    shutil.copy(label_src_path, label_dest_dir)
                    count += 1
                else:
                    print(f"[Cảnh báo] Không tìm thấy tệp nhãn: {label_src_path}")
                    
            except Exception as e:
                print(f"[LỖI] Không thể sao chép {img_path}: {e}")
        return count

    # 5. THỰC HIỆN SAO CHÉP
    print("\nĐang sao chép tệp huấn luyện (train)...")
    count_train = copy_files(train_files, train_img_dir, train_label_dir, RAW_LABELS_PATH)
    print(f"Hoàn tất sao chép {count_train} cặp ảnh/nhãn train.")

    print("\nĐang sao chép tệp kiểm tra (val)...")
    count_val = copy_files(val_files, val_img_dir, val_label_dir, RAW_LABELS_PATH)
    print(f"Hoàn tất sao chép {count_val} cặp ảnh/nhãn val.")

    print("\n--- QUÁ TRÌNH PHÂN CHIA HOÀN TẤT ---")

    