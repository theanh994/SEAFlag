import customtkinter as ctk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import os
import pickle
from tkinter import filedialog

# --- CẤU HÌNH CỐT LÕI (SỬA LẠI CHO ĐÚNG VỚI MÁY CỦA BẠN) ---

# 1. Đường dẫn đến mô hình 'best.pt' MỚI NHẤT
# (Nên là mô hình yolov8s bạn vừa train lại với dataset mới)
MODEL_PATH = "D:/theanh/Documents/AseanFlags_Roboflow/runs/detect/AseanFlags_YOLOv8s_Run/weights/best.pt"

# 2. Đường dẫn đến CSDL histogram (từ generate_histograms.py)
HIST_DB_PATH = "histograms.pkl" 

# 3. Ngưỡng (Thresholds) - Đã tinh chỉnh
YOLO_CONF_THRESHOLD = 0.25       # Ngưỡng tự tin của YOLO (để bắt cờ mờ/nhỏ)
HIST_SIMILARITY_THRESHOLD = 0.4  # Ngưỡng giống nhau của Histogram (nới lỏng cho ảnh thực tế)

# 4. Danh sách 11 class (PHẢI KHỚP VỚI data.yaml MỚI)
CLASS_NAMES = [
    'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 
    'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam'
]

# --- KẾT THÚC CẤU HÌNH ---


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class SEAFlagApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NHẬN DIỆN CỜ ĐÔNG NAM Á (Phiên bản Cốt Lõi 2.1)")
        self.geometry("1100x750")
        
        self.running_stream = False  # Cờ kiểm soát vòng lặp camera/video
        self.cap = None              # Biến giữ đối tượng VideoCapture
        self.after_id = None         # Thêm biến để lưu ID vòng lặp

        # --- Tải các "cốt lõi" MỘT LẦN lúc khởi động ---
        try:
            print("Đang tải mô hình YOLO...")
            self.model = YOLO(MODEL_PATH)
            print("Đang tải CSDL Histogram...")
            self.ref_histograms = self.load_histograms(HIST_DB_PATH)
            if self.ref_histograms is None:
                raise FileNotFoundError(f"Không tìm thấy hoặc không thể đọc {HIST_DB_PATH}")
            print("Ứng dụng sẵn sàng!")
        except Exception as e:
            print(f"LỖI KHỞI ĐỘNG NGHIÊM TRỌNG: {e}")
            print("Vui lòng kiểm tra lại đường dẫn MODEL_PATH và HIST_DB_PATH")
            # Hiển thị lỗi trên GUI
            self.video_label = ctk.CTkLabel(self, text=f"LỖI: {e}\nKiểm tra lại đường dẫn trong code.", font=("Roboto", 16), text_color="red")
            self.video_label.pack(pady=200)
            return

        self.create_widgets()

    def create_widgets(self):
        """Tạo giao diện người dùng (GUI)"""
        ctk.CTkLabel(self, text="NHẬN DIỆN CỜ ĐÔNG NAM Á", font=("Roboto", 28, "bold")).pack(pady=15)
        
        self.video_frame = ctk.CTkFrame(self, width=840, height=560, fg_color="black")
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False) # Ngăn frame co lại
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Chọn Ảnh, Video hoặc Mở Camera", font=("Roboto", 20))
        self.video_label.pack(expand=True)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(btn_frame, text="Chọn Ảnh", command=self.select_photo, width=160, height=45).grid(row=0, column=0, padx=10)
        ctk.CTkButton(btn_frame, text="Chọn Video", command=self.select_video, width=160, height=45).grid(row=0, column=1, padx=10)
        ctk.CTkButton(btn_frame, text="Mở Camera", command=self.start_camera, width=160, height=45).grid(row=0, column=2, padx=10)
        ctk.CTkButton(btn_frame, text="Dừng", command=self.stop_stream, fg_color="red", hover_color="#CC0000", width=160, height=45).grid(row=0, column=3, padx=10)

    # --- CÁC HÀM TIỆN ÍCH (TỪ CÁC TỆP .PY CŨ) ---

    def load_histograms(self, path):
        if not os.path.exists(path):
            print(f"[LỖI] Không tìm thấy tệp {path}! Hãy chạy generate_histograms.py trước.")
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Lỗi khi tải histogram: {e}")
            return None

    def calculate_hist(self, image):
        """Hàm "cốt lõi" CV: Tính 2D Histogram (Hue, Saturation)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]
            )
            cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
            return hist
        except cv2.error:
            return None

    # --- LOGIC XỬ LÝ ẢNH (BƯỚC 4 - HYBRID) ---
    
    def select_photo(self):
        self.stop_stream() # Dừng camera/video nếu đang chạy
        path = filedialog.askopenfilename(title="Chọn một ảnh", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.video_label.configure(text="Đang xử lý ảnh...", font=("Roboto", 20))
            self.update_idletasks() # Cập nhật GUI ngay
            self.process_image_hybrid(path)

    def process_image_hybrid(self, path):
        """
        Logic "cốt lõi" (YOLO + Histogram) VỚI LOGIC VẼ ĐƯỢC NÂNG CẤP.
        Giờ đây sẽ vẽ cả box "Đạt" (Xanh) và "Loại" (Đỏ).
        """
        img_bgr = cv2.imread(path)
        if img_bgr is None: 
            self.video_label.configure(text="Không thể đọc tệp ảnh")
            return

        print("Đang chạy YOLO (Bước 1)...")
        # KHÔNG resize thủ công, để YOLO tự xử lý (tự letterbox)
        results = self.model.predict(
            img_bgr, 
            conf=YOLO_CONF_THRESHOLD, 
            imgsz=640, # Yêu cầu YOLO xử lý ở 640
            verbose=False
        )
        result = results[0]
        
        print(f"YOLO tìm thấy {len(result.boxes)} đối tượng. Đang xác thực (Bước 2)...")
        # box_count = 0

        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id] # Lấy tên từ danh sách chuẩn
            
            conf = float(box.conf[0])
             
            ref_hist = self.ref_histograms.get(class_name)
            if ref_hist is None:
                print(f"[Cảnh báo] Không có histogram mẫu cho {class_name}")
                continue
                
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size == 0: continue
            
            roi_hist = self.calculate_hist(roi)
            if roi_hist is None: continue

            score = cv2.compareHist(ref_hist, roi_hist, cv2.HISTCMP_CORREL)
            
            hist_text = ""
            box_color = (0, 0, 0) # Màu mặc định

            if score >= HIST_SIMILARITY_THRESHOLD:
                print(f"-> XÁC THỰC: {class_name} (YOLO: {conf:.2f}, Hist: {score:.2f}) -> Đạt")
                box_color = (0, 255, 0)  # MÀU XANH (Đạt)
                hist_text = f"Hist: {score:.2f} (Dat)"
            else:
                print(f"-> XÁC THỰC: {class_name} (YOLO: {conf:.2f}, Hist: {score:.2f}) -> Loai")
                box_color = (0, 0, 255)  # MÀU ĐỎ (Loại)
                hist_text = f"Hist: {score:.2f} (Loai)"

                # Vẽ box và nhãn nếu Đạt
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color, 2)
                # Tính kích thước text để làm nền
            text_lines = [
                class_name,
                f"YOLO: {conf:.2f}",
                # f"Hist: {score:.2f}"
                hist_text
            ]
            max_width = 0
            for txt in text_lines:
                # (Lưu ý: Kích thước font và độ dày nên giống nhau)
                (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                if w > max_width:
                    max_width = w
                
            bg_width = max_width + 10
            bg_height = 50 # (15px * 3 dòng + lề)

            # Kiểm tra nếu box gần mép trên
            if y1 < 60: 
            # Vẽ nền BÊN TRONG (với màu đã chọn)
                cv2.rectangle(img_bgr, (x1, y1 + 2), (x1 + bg_width, y1 + bg_height), box_color, -1)
                    
                # Viết text (luôn là màu đen)
                cv2.putText(img_bgr, text_lines[0], (x1 + 5, y1 + 18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(img_bgr, text_lines[1], (x1 + 5, y1 + 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(img_bgr, text_lines[2], (x1 + 5, y1 + 48), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            else: 
                # Vẽ nền BÊN TRÊN (với màu đã chọn)
                cv2.rectangle(img_bgr, (x1, y1 - bg_height), (x1 + bg_width, y1), box_color, -1)
                
                # Viết text (luôn là màu đen)
                cv2.putText(img_bgr, text_lines[0], (x1 + 5, y1 - 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(img_bgr, text_lines[1], (x1 + 5, y1 - 18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(img_bgr, text_lines[2], (x1 + 5, y1 - 3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # --- KẾT THÚC LOGIC VẼ NÂNG CAO ---
                
        print("Đã hoàn tất xử lý ảnh.")
        self.display_image_in_frame(img_bgr)


    # --- LOGIC XỬ LÝ VIDEO/CAMERA (BƯỚC 5 - TỐI ƯU) ---

    def select_video(self):
        self.stop_stream() # Dừng stream cũ
        path = filedialog.askopenfilename(title="Chọn một video", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if path:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self.video_label.configure(text="Không thể mở tệp video")
                return
            self.running_stream = True
            self.update_stream_frame() # Gọi vòng lặp chung

    def start_camera(self):
        self.stop_stream() # Dừng stream cũ
        self.cap = cv2.VideoCapture(0) # 0 = Camera mặc định
        if not self.cap.isOpened():
            self.video_label.configure(text="Không thể mở camera")
            return
        self.running_stream = True
        self.update_stream_frame() # Gọi vòng lặp chung

    def stop_stream(self):
        """Hàm dừng đã được nâng cấp."""
        ### SỬA : Thêm logic hủy vòng lặp 'after' ###
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        
        self.running_stream = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image=None, text="Đã dừng", font=("Roboto", 20))

    def update_stream_frame(self):
        """
        Hàm "cốt lõi" của Bước 5: Dùng model.track() cho cả Video và Camera.
        Đây là vòng lặp không-chặn (non-blocking) của CustomTkinter.
        """
        if not self.running_stream or not self.cap:
            self.stop_stream()
            return

        ret, frame = self.cap.read()
        
        if not ret:
            print("Kết thúc video hoặc lỗi camera.")
            self.stop_stream()
            return
        
        # --- CỐT LÕI TỐI ƯU: model.track() ---
        results = self.model.track(
            frame, 
            persist=True,               # Giữ bộ nhớ theo dõi giữa các frame
            conf=YOLO_CONF_THRESHOLD,
            verbose=False               # Tắt log thừa trong terminal
        )
        
        # results[0].plot() sẽ tự động vẽ box với ID theo dõi
        annotated_frame = results[0].plot()
        
        # Hiển thị
        self.display_image_in_frame(annotated_frame)
        
        # Lên lịch cho frame tiếp theo
        self.after(10, self.update_stream_frame) # (10ms ~ 100 FPS, đủ nhanh)


    # --- HÀM HIỂN THỊ (QUAN TRỌNG) ---

    def display_image_in_frame(self, cv_img):
        """
        Hàm này nhận ảnh BGR, resize nó cho vừa khung 840x560
        (giữ đúng tỉ lệ) và hiển thị lên GUI.
        """
        try:
            # 1. Chuyển sang RGB
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # 2. Lấy kích thước ảnh và kích thước khung
            img_h, img_w = cv_img_rgb.shape[:2]
            frame_w, frame_h = 840, 560
            
            # 3. Tính toán tỉ lệ để giữ nguyên aspect ratio (thêm viền)
            scale = min(frame_w / img_w, frame_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            # 4. Resize ảnh
            resized_img = cv2.resize(cv_img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 5. Tạo ảnh PIL và ảnh CTkImage
            pil_img = Image.fromarray(resized_img)
            ctk_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
            
            # 6. Hiển thị
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image
        
        except Exception as e:
            print(f"Lỗi hiển thị: {e}")
            
    def on_closing(self):
        """Dọn dẹp khi tắt app"""
        print("Đang đóng ứng dụng...")
        self.stop_stream()
        self.destroy()

# --- CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    app = SEAFlagApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()