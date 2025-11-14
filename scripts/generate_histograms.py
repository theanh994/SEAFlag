import cv2
import numpy as np
import os
import pickle

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa 11 ảnh mẫu (sạch, phẳng)
SAMPLE_DIR = "D:/theanh/Documents/AseanFlags_Roboflow/data/flag_samples"

# Tên tệp CSDL histogram sẽ được tạo ra
OUTPUT_FILE = "histograms.pkl"

# Danh sách 11 tên class (Y HỆT như trong file .yaml)
# (Chúng ta dùng tên này để tìm tệp ảnh, ví dụ 'Brunei' -> 'Brunei.png')
CLASS_NAMES = [
    'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 
    'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam'
]

# --- HÀM TÍNH HISTOGRAM ---
def calculate_hist(image):
    """
    Chuyển ảnh sang HSV và tính toán 2D histogram cho Hue và Saturation.
    """
    # 1. Chuyển sang HSV (Tốt hơn cho việc phân biệt màu sắc)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 2. Tính 2D histogram cho kênh Hue (Tông màu) và Saturation (Độ bão hòa)
    # Đây là "cốt lõi": chúng ta quan tâm đến MÀU GÌ (Hue) và ĐẬM HAY NHẠT (Saturation)
    hist = cv2.calcHist(
        [hsv],          # Ảnh đầu vào
        [0, 1],         # Kênh 0 (Hue) và 1 (Saturation)
        None,           # Không dùng Mask
        [180, 256],     # Số bins: 180 cho Hue (0-179), 256 cho Sat (0-255)
        [0, 180, 0, 256] # Dải giá trị cho H và S
    )
    
    # 3. Chuẩn hóa histogram về 0-1 (rất quan trọng để so sánh)
    cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    
    return hist

# --- HÀM CHÍNH ---
def create_hist_database():
    print(f"Bắt đầu tạo CSDL Histogram từ: {SAMPLE_DIR}")
    
    # Dictionary để lưu trữ histogram
    histogram_database = {}
    
    for class_name in CLASS_NAMES:
        # Tự động tìm tệp ảnh (.png, .jpg, v.v.)
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            path = os.path.join(SAMPLE_DIR, class_name + ext)
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            print(f"[CẢNH BÁO] Không tìm thấy ảnh cho: {class_name}")
            continue
            
        # Đọc ảnh
        image = cv2.imread(img_path)
        if image is None:
            print(f"[LỖI] Không thể đọc ảnh: {img_path}")
            continue
            
        # Tính histogram
        hist = calculate_hist(image)
        
        # Lưu vào CSDL
        histogram_database[class_name] = hist
        print(f"Đã xử lý và lưu histogram cho: {class_name}")
        
    # Sau khi xong, lưu CSDL vào file .pkl
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(histogram_database, f)
        
    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã lưu CSDL histogram vào: {OUTPUT_FILE}")

# --- BẮT ĐẦU CHẠY ---
if __name__ == '__main__':
    create_hist_database()