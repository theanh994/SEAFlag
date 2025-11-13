import os
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO

def start_training():
    # -------------------------------------------------------------------
    # 1. CẤU HÌNH (SỬA 1 DÒNG NÀY)
    # -------------------------------------------------------------------

    # Dán đường dẫn TUYỆT ĐỐI đến tệp .yaml bạn vừa tạo (dùng dấu /)
    YAML_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/AseanFlags_Dataset_Split/asean_flags.yaml"

    # -------------------------------------------------------------------
    # 2. CHỌN MÔ HÌNH
    # -------------------------------------------------------------------

    # Tải mô hình YOLOv8 'nano' (.pt). 
    # Đây là mô hình nhỏ nhất, phù hợp nhất cho GTX 1650 (4GB VRAM).
    model = YOLO('yolov8s.pt')

    print("Đã tải mô hình yolov8s.pt.")
    print(f"Sử dụng tệp cấu hình: {YAML_PATH}")

    # -------------------------------------------------------------------
    # 3. THỰC HIỆN HUẤN LUYỆN
    # -------------------------------------------------------------------
    print("Bắt đầu huấn luyện...")

    try:
        results = model.train(
            # --- Cấu hình chính ---
            data=YAML_PATH,         # Đường dẫn đến tệp .yaml
            epochs=100,             # Số lần học (100 là mức tốt để bắt đầu)
            imgsz=640,              # Resize tất cả ảnh về 640x640
            device=0,               # Chỉ định sử dụng GPU (0 = GPU đầu tiên)
            
            # --- Tối ưu cho GTX 1650 (4GB VRAM) ---
            batch=4,                # Số ảnh xử lý 1 lần. 
                                    # GTX 1650 4GB chỉ nên dùng batch=4 hoặc thấp hơn
            # --- THÊM DÒNG NÀY ĐỂ ĐẶT TÊN CỤ THỂ ---
            name="AseanFlags_YOLOv8s_Run",  # Bạn có thể đặt tên bất kỳ

            # --- CỐT LÕI: TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION) ---
            # Đây là các tham số bạn muốn kiểm soát trong code
            degrees=15.0,           # Xoay ngẫu nhiên +/- 15 độ
            translate=0.1,          # Dịch chuyển ảnh +/- 10%
            scale=0.1,              # Phóng to/thu nhỏ +/- 10%
            shear=5.0,              # Làm nghiêng ảnh +/- 5 độ
            perspective=0.001,      # Thay đổi góc nhìn (giả lập 3D)
            flipud=0.0,             # Lật trên/dưới (KHÔNG DÙNG cho cờ)
            fliplr=0.5,             # Lật trái/phải (50% cơ hội)
            mosaic=1.0,             # Cốt lõi YOLO: Ghép 4 ảnh (100% cơ hội)
            mixup=0.1,              # Trộn 2 ảnh (10% cơ hội)
            
            # Tăng cường màu sắc (Rất quan trọng cho cờ)
            hsv_h=0.015,            # Thay đổi TÔNG MÀU +/- 1.5%
            hsv_s=0.7,              # Thay đổi ĐỘ BÃO HÒA (màu nhạt/đậm) +/- 70%
            hsv_v=0.4               # Thay đổi ĐỘ SÁNG +/- 40%
        )
        
        print("--- HUẤN LUYỆN HOÀN TẤT ---")
        print("Mô hình và kết quả được lưu trong thư mục 'runs/detect/train'")

    except torch.cuda.OutOfMemoryError:
        print("\n[LỖI] Lỗi CUDA Out of Memory!")
        print("VRAM của GTX 1650 (4GB) đã bị đầy.")
        print("Hãy thử giảm tham số 'batch' xuống 4 và chạy lại.")

    except Exception as e:
        print(f"\n[LỖI] Một lỗi không mong muốn đã xảy ra: {e}")

# Khối này đảm bảo mã huấn luyện CHỈ chạy khi là tiến trình chính
if __name__ == '__main__':
    freeze_support()  # Cần thiết cho Windows
    start_training()    # Gọi hàm huấn luyện    