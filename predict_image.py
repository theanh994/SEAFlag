import cv2
import pickle
import os
from ultralytics import YOLO

# --- CẤU HÌNH ---
# 1. Đường dẫn đến mô hình 'best.pt'
MODEL_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/runs/detect/AseanFlags_YOLOv8s_Run/weights/best.pt"

# 2. Đường dẫn đến CSDL histogram chúng ta vừa tạo
HIST_DB_PATH = "histograms.pkl"

# 3. Đường dẫn đến ảnh bạn muốn KIỂM TRA
TEST_IMAGE_PATH = r"D:\theanh\Documents\AseanFlags_Roboflow\country_flags_asean\Philippines\9201.png"

# 4. Ngưỡng (Thresholds)
YOLO_CONF_THRESHOLD = 0.5
HIST_SIMILARITY_THRESHOLD = 0.05

CLASS_NAMES = [
    'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 
    'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam'
]

# --- HÀM TÍNH HISTOGRAM ---
def calculate_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return hist

def load_histograms(path):
    """Tải CSDL histogram"""
    if not os.path.exists(path):
        print(f"[LỖI] Không tìm thấy tệp {path}!")
        print("Vui lòng chạy 'generate_histograms.py' trước tiên.")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Lỗi khi tải histogram: {e}")
        return None

# --- HÀM CHÍNH ĐỂ DỰ ĐOÁN ---
def predict_with_validation():
    print("Đang tải mô hình YOLO...")
    model = YOLO(MODEL_PATH)
    
    print(f"Đang tải CSDL Histogram từ {HIST_DB_PATH}...")
    ref_histograms = load_histograms(HIST_DB_PATH)
    if ref_histograms is None:
        return
        
    print(f"Đang xử lý ảnh: {TEST_IMAGE_PATH}")
    image_bgr = cv2.imread(TEST_IMAGE_PATH)
    if image_bgr is None:
        print("[LỖI] Không thể đọc ảnh test!")
        return

    # 1. BƯỚC 1: CHẠY YOLO
    print("Đang chạy YOLO (Bước 1)...")
    results = model.predict(
        image_bgr, 
        conf=YOLO_CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )
    result = results[0]

    # 2. BƯỚC 2: XÁC THỰC BẰNG HISTOGRAM
    print(f"YOLO tìm thấy {len(result.boxes)} đối tượng. Đang xác thực...")
    
    for box in result.boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        class_id = int(box.cls[0])
        
        if class_id < len(CLASS_NAMES):
             class_name = CLASS_NAMES[class_id]
        else:
             class_name = "Unknown" # Đề phòng lỗi

        conf = float(box.conf[0])

        ref_hist = ref_histograms.get(class_name)
        if ref_hist is None:
            print(f"[Cảnh báo] Không có histogram chuẩn cho {class_name}, bỏ qua.")
            continue
            
        try:
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0: continue
        except Exception as e:
            print(f"Lỗi khi cắt ROI: {e}")
            continue
            
        roi_hist = calculate_hist(roi)
        if roi_hist is None: continue

        score = cv2.compareHist(ref_hist, roi_hist, cv2.HISTCMP_CORREL)

        # --- LOGIC ĐÁNH DẤU MÀU (XANH/ĐỎ) ---

        hist_text = ""
        box_color = (0, 0, 0) # Màu mặc định

        if score >= HIST_SIMILARITY_THRESHOLD:
            print(f"-> XÁC THỰC: {class_name} (YOLO Conf: {conf:.2f}, Hist Score: {score:.2f}) -> Đạt")
            box_color = (0, 255, 0)  # MÀU XANH (Đạt)
            hist_text = f"Hist: {score:.2f} (Dat)"
        else:
            print(f"-> XÁC THỰC: {class_name} (YOLO Conf: {conf:.2f}, Hist Score: {score:.2f}) -> LOẠI")
            box_color = (0, 0, 255)  # MÀU ĐỎ (Loại)
            hist_text = f"Hist: {score:.2f} (Loai)"

        # --- LOGIC VẼ NÂNG CAO (LUÔN VẼ) ---
        # Vẽ box màu xanh
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, 2)
        
        # Tính kích thước text để làm nền vừa đủ
        text_lines = [
            class_name,
            f"YOLO: {conf:.2f}",
            # f"Hist: {score:.2f}"
            hist_text
        ]
        max_width = 0
        for txt in text_lines:
            (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            if w > max_width:
                max_width = w
        
        bg_width = max_width + 10
        bg_height = 50
        
        # Kiểm tra nếu box gần mép trên thì vẽ text BÊN TRONG box
        if y1 < 60:  # Nếu quá gần mép trên
            # Vẽ nền BÊN TRONG box (phía dưới cạnh trên)
            cv2.rectangle(image_bgr, (x1, y1 + 2), (x1 + bg_width, y1 + bg_height), box_color, -1)
            
            # Viết text BÊN TRONG
            cv2.putText(image_bgr, text_lines[0], (x1 + 5, y1 + 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image_bgr, text_lines[1], (x1 + 5, y1 + 33), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image_bgr, text_lines[2], (x1 + 5, y1 + 48), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:  # Box ở xa mép trên, vẽ text phía TRÊN box như bình thường
            # Vẽ nền màu xanh cho text
            cv2.rectangle(image_bgr, (x1, y1 - bg_height), (x1 + bg_width, y1), box_color, -1)
            
            # Viết text màu đen trên nền xanh
            cv2.putText(image_bgr, text_lines[0], (x1 + 5, y1 - 33), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image_bgr, text_lines[1], (x1 + 5, y1 - 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image_bgr, text_lines[2], (x1 + 5, y1 - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Resize ảnh để vừa màn hình (không quá lớn)
    h, w = image_bgr.shape[:2]
    max_width = 1280
    max_height = 720
    
    # Tính tỉ lệ scale
    scale = min(max_width / w, max_height / h, 1.0)  # Không phóng to nếu ảnh nhỏ
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h))
    
    # Hiển thị ảnh với cửa sổ tự động điều chỉnh
    cv2.imshow("Ket Qua Da Xac Thuc (An ESC de thoat)", image_bgr)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'): # phím ESC hoặc 'q'
            break

    cv2.destroyAllWindows()


# --- BẮT ĐẦU CHẠY ---
if __name__ == "__main__":
    predict_with_validation()