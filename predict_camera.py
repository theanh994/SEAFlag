import cv2
from ultralytics import YOLO

# --- CẤU HÌNH ---
# 1. Đường dẫn đến mô hình 'best.pt' (MÔ HÌNH MỚI NHẤT CỦA BẠN)
MODEL_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/runs/detect/AseanFlags_YOLOv8s_Run/weights/best.pt"

# 2. Nguồn video
# (Sửa 'VIDEO_PATH' thành '0' để gọi camera laptop mặc định)
VIDEO_SOURCE = 0 

# 3. Ngưỡng (Threshold)
YOLO_CONF_THRESHOLD = 0.4   

# --- HÀM CHÍNH ĐỂ DỰ ĐOÁN CAMERA ---
def predict_camera():
    print(f"Đang tải mô hình từ: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f"Đang mở camera (Nguồn: {VIDEO_SOURCE})...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở camera. (Nguồn: {VIDEO_SOURCE})")
        return

    while True:
        # Đọc từng frame
        success, frame = cap.read()
        
        if not success:
            print("Lỗi đọc camera.")
            break

        # --- CỐT LĨ: DÙNG model.track() ---
        results = model.track(
            frame, 
            persist=True, 
            conf=YOLO_CONF_THRESHOLD,
            verbose=False # Thêm dòng này để tắt bớt log thừa
        )
        
        # Lấy frame đã được vẽ box
        annotated_frame = results[0].plot()

        # Hiển thị kết quả
        cv2.imshow("Nhan dien Co Camera (An 'q' hoac 'ESC' de thoat)", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Đã nhấn phím thoát...")
            break
    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng camera.")

# --- BẮT ĐẦU CHẠY ---
if __name__ == "__main__":
    predict_camera()