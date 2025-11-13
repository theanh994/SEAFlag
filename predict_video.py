import cv2
from ultralytics import YOLO

# --- CẤU HÌNH ---
# 1. Đường dẫn đến mô hình 'best.pt' (MÔ HÌNH MỚI NHẤT CỦA BẠN)
# (Giả sử bạn đã huấn luyện lại với yolov8s)
MODEL_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/runs/detect/AseanFlags_YOLOv8s_Run/weights/best.pt"

# 2. Đường dẫn đến video bạn muốn KIỂM TRA
VIDEO_PATH = r"D:\Downloads\videoplayback.mp4" # (Thay bằng video của bạn)

# 3. Ngưỡng (Threshold)
# (Chúng ta có thể dùng ngưỡng thấp hơn cho video/ảnh khó)
YOLO_CONF_THRESHOLD = 0.4   

# --- HÀM CHÍNH ĐỂ DỰ ĐOÁN VIDEO ---
def predict_video():
    print(f"Đang tải mô hình từ: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f"Đang mở video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở file video: {VIDEO_PATH}")
        return

    while True:
        # Đọc từng frame
        success, frame = cap.read()
        
        if not success:
            print("Kết thúc video.")
            break # Hết video thì dừng

        # --- CỐT LĨ: DÙNG model.track() ---
        # Đây là chiến lược tối ưu tốc độ (FPS)
        # 'persist=True' báo cho tracker biết các frame này là liên tục
        results = model.track(
            frame, 
            persist=True, 
            conf=YOLO_CONF_THRESHOLD
        )
        
        # Lấy frame đã được vẽ box (rất tiện lợi)
        # Nó sẽ tự động lấy tên đúng 'Timor-Leste' từ mô hình mới
        annotated_frame = results[0].plot()

        # Hiển thị kết quả
        cv2.imshow("Nhan dien Co (An 'q' de thoat)", annotated_frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng video.")

# --- BẮT ĐẦU CHẠY ---
if __name__ == "__main__":
    predict_video()