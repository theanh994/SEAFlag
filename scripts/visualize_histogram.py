import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CẤU HÌNH (Sửa 1 dòng này) ---

# 1. Đường dẫn đến MỘT ảnh mẫu "sạch" mà bạn muốn minh họa
# (Hãy đảm bảo tên tệp khớp, ví dụ 'Vietnam.jpg' hoặc 'Timor-Leste.jpg')
SAMPLE_IMAGE_PATH = r"D:\theanh\Documents\AseanFlags_Roboflow\data\country_flags_asean\Vietnam\21194.png"

# 2. Tên tệp ảnh sẽ được tạo ra
OUTPUT_FIGURE_NAME = "Hinh_2_3.png"

# --- HÀM CỐT LÕI (Copy từ generate_histograms.py) ---
def calculate_hist(image):
    """
    Hàm "cốt lõi" CV: Tính 2D Histogram (Hue, Saturation)
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Chúng ta dùng 2 kênh Hue (màu sắc) và Saturation (độ bão hòa)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]
        )
        # KHÔNG chuẩn hóa (normalize) ở đây, để Matplotlib tự xử lý
        # cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
        return hist
    except cv2.error:
        return None

# --- HÀM CHÍNH ĐỂ VẼ BIỂU ĐỒ ---
def generate_plot():
    print(f"Đang đọc ảnh mẫu từ: {SAMPLE_IMAGE_PATH}")
    image = cv2.imread(SAMPLE_IMAGE_PATH)
    
    if image is None:
        print(f"[LỖI] Không thể đọc ảnh. Vui lòng kiểm tra lại đường dẫn SAMPLE_IMAGE_PATH.")
        return

    # 1. Tính toán histogram "cốt lõi"
    hist = calculate_hist(image)
    
    if hist is None:
        print("[LỖI] Không thể tính toán histogram.")
        return

    # 2. Dùng Matplotlib để vẽ (Đây là phần tạo ra hình)
    print("Đang vẽ biểu đồ 2D Histogram...")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Dùng log scale (np.log(hist+1)) để các vùng màu nhỏ hiển thị rõ hơn
    # Dùng 'nearest' để thấy rõ các pixel bin
    cax = ax.imshow(np.log(hist + 1), interpolation='nearest', aspect='auto', cmap='viridis')

    # 3. Đặt tên cho các trục và tiêu đề
    ax.set_title("Biểu đồ 2D Histogram (H-S)", fontsize=16)
    ax.set_ylabel("Hue (Tông màu)", fontsize=12)
    ax.set_xlabel("Saturation (Độ bão hòa)", fontsize=12)
    
    # Thêm thanh màu (color bar)
    fig.colorbar(cax, label='Log(Số lượng Pixel)')

    # 4. Lưu tệp ảnh
    plt.savefig(OUTPUT_FIGURE_NAME)
    print(f"--- THÀNH CÔNG ---")
    print(f"Đã lưu hình ảnh vào: {OUTPUT_FIGURE_NAME}")
    
    # 5. (Tùy chọn) Hiển thị biểu đồ
    # plt.show()

# --- BẮT ĐẦU CHẠY ---
if __name__ == "__main__":
    # Đảm bảo bạn đã chạy: pip install matplotlib
    generate_plot()