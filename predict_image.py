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
TEST_IMAGE_PATH = r"D:\theanh\Documents\AseanFlags_Roboflow\raw\images\9187_png.rf.4b17e1c8fee445b83daa1edb44460b59.jpg" # (Thay bằng ảnh của bạn)

# 4. Ngưỡng (Thresholds)
YOLO_CONF_THRESHOLD = 0.5   # Độ tự tin tối thiểu của YOLO
HIST_SIMILARITY_THRESHOLD = 0.05 # Độ tương đồng histogram (0.0 -> 1.0)
                                # (Chỉnh số này nếu cần, 0.6 là mức tốt)

# --- HÀM TÍNH HISTOGRAM (Y HỆT NHƯ TRONG SCRIPT 4.1) ---
def calculate_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return hist

# --- HÀM CHÍNH ĐỂ DỰ ĐOÁN ---
def predict_with_validation():
    print("Đang tải mô hình YOLO...")
    model = YOLO(MODEL_PATH)
    
    print(f"Đang tải CSDL Histogram từ {HIST_DB_PATH}...")
    with open(HIST_DB_PATH, 'rb') as f:
        ref_histograms = pickle.load(f)
        
    print(f"Đang xử lý ảnh: {TEST_IMAGE_PATH}")
    image_bgr = cv2.imread(TEST_IMAGE_PATH)
    if image_bgr is None:
        print("[LỖI] Không thể đọc ảnh test!")
        return

    # 1. BƯỚC 1: CHẠY YOLO (Đề xuất)
    # Lấy kết quả từ YOLO
    results = model.predict(image_bgr, conf=YOLO_CONF_THRESHOLD)
    
    result = results[0] # Lấy kết quả đầu tiên
    class_names = result.names # Lấy danh sách tên class (ví dụ: {10: 'Vietnam'})
    
    validated_boxes = [] # Danh sách box ĐÃ ĐƯỢC XÁC THỰC

    # 2. BƯỚC 2: XÁC THỰC BẰNG HISTOGRAM (Cốt lõi CV)
    print(f"YOLO tìm thấy {len(result.boxes)} đối tượng. Đang xác thực...")
    
    for box in result.boxes:
        # Lấy tọa độ và class
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        
        # Lấy histogram tham chiếu (chuẩn)
        ref_hist = ref_histograms.get(class_name)
        if ref_hist is None:
            print(f"[Cảnh báo] Không có histogram chuẩn cho {class_name}, bỏ qua.")
            continue
            
        # --- CỐT LÕI CV ---
        # 1. Cắt vùng ảnh (ROI) mà YOLO đề xuất
        try:
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0: continue # Bỏ qua nếu box bị lỗi
        except Exception as e:
            print(f"Lỗi khi cắt ROI: {e}")
            continue
            
        # 2. Tính histogram của vùng ROI
        roi_hist = calculate_hist(roi)
        
        # 3. So sánh 2 histogram (dùng phương pháp Correlation)
        score = cv2.compareHist(ref_hist, roi_hist, cv2.HISTCMP_CORREL)
        
        # 4. Ra quyết định
        if score >= HIST_SIMILARITY_THRESHOLD:
            print(f"-> XÁC THỰC: {class_name} (Score: {score:.2f}) -> Đạt")
            # Nếu đạt, lưu lại để vẽ
            validated_boxes.append((x1, y1, x2, y2, class_name, score))
        else:
            print(f"-> XÁC THỰC: {class_name} (Score: {score:.2f}) -> LOẠI BỎ (Không giống)")

    # 3. BƯỚC 3: VẼ KẾT QUẢ ĐÃ XÁC THỰC
    print(f"Vẽ {len(validated_boxes)} đối tượng đã được xác thực.")
    for (x1, y1, x2, y2, class_name, score) in validated_boxes:
        # Vẽ box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Viết nhãn
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image_bgr, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị ảnh
    cv2.imshow("Ket Qua Da Xac Thuc (An ESC de thoat)", image_bgr)
    cv2.waitKey(0) # Đợi nhấn phím
    cv2.destroyAllWindows()


# --- BẮT ĐẦU CHẠY ---
if __name__ == "__main__":
    predict_with_validation()