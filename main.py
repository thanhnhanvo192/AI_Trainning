# FILE: main.py
import cv2
import logging
from ultralytics import YOLO

# --- IMPORT MODULE BỔ TRỢ ---
# Yêu cầu: File helper.py và utils_rotate.py nằm CÙNG THƯ MỤC với main.py
try:
    import util.helper as helper
    import util.utils_rotate as utils_rotate
    print(">> Đã nạp thành công các module bổ trợ.")
except ImportError as e:
    print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file bổ trợ ({e})")
    print("Vui lòng kiểm tra lại file helper.py và utils_rotate.py")
    exit()

# --- CẤU HÌNH ---
PATH_DETECT = 'model/detect_best.pt'  # Model phát hiện biển
PATH_OCR    = 'model/ocr_best.pt'     # Model đọc chữ

# --- 1. LOAD MODELS ---
# Tắt bớt log của YOLO cho đỡ rối mắt
logging.getLogger("ultralytics").setLevel(logging.ERROR)

print(f">> Đang tải model Detect: {PATH_DETECT} ...")
model_plate = YOLO(PATH_DETECT)
model_char = YOLO(PATH_OCR, task='detect')

print(f">> Đang tải model OCR: {PATH_OCR} ...")
model_char = YOLO(PATH_OCR) 

# --- 2. KHỞI TẠO CAMERA ---
# Đổi thành 0 nếu dùng Webcam laptop, 1 nếu dùng Cam rời
vid = cv2.VideoCapture(0)
# Cài đặt độ phân giải (giúp FPS ổn định hơn)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n--- HỆ THỐNG ĐANG CHẠY (Nhấn 'q' để thoát) ---\n")

while True:
    ret, frame = vid.read()
    if not ret:
        print("Mất kết nối Camera!")
        break

    # ==========================================================
    # BƯỚC 1: PHÁT HIỆN BIỂN SỐ (DETECT)
    # ==========================================================
    results_plate = model_plate.predict(frame, conf=0.5, verbose=False)

    for res in results_plate:
        for box in res.boxes:
            # Lấy tọa độ khung biển số
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            
            # Vẽ khung biển số lên màn hình chính
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

            # Cắt ảnh biển số (Crop)
            crop_img = frame[py1:py2, px1:px2]
            
            # Bỏ qua nếu ảnh cắt lỗi hoặc quá nhỏ
            if crop_img.size == 0 or crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
                continue

            # ==========================================================
            # BƯỚC 2: XỬ LÝ ẢNH (DESKEW / XOAY THẲNG)
            # ==========================================================
            try:
                # Tăng tương phản + Xoay thẳng
                crop_img = utils_rotate.deskew(crop_img, change_cons=1, center_thres=1)
            except Exception as e:
                # Nếu xoay lỗi thì dùng ảnh gốc, không sao cả
                pass

            # ==========================================================
            # BƯỚC 3: NHẬN DIỆN KÝ TỰ (OCR)
            # ==========================================================
            # conf=0.5: Chỉ lấy ký tự rõ nét
            # iou=0.45: Loại bỏ các khung trùng nhau
            ocr_results = model_char.predict(crop_img, conf=0.5, iou=0.45, verbose=False)
            
            # Chuẩn bị danh sách dữ liệu để gửi sang helper
            list_chars = []
            
            for r in ocr_results:
                for char_box in r.boxes:
                    # Lấy tọa độ ký tự
                    bx1, by1, bx2, by2 = map(float, char_box.xyxy[0])
                    
                    # Lấy nhãn (Label) - Vì dùng .pt nên model tự biết tên
                    cls_id = int(char_box.cls[0])
                    label = model_char.names[cls_id] 
                    
                    # Tính toán các thông số hình học
                    cx = (bx1 + bx2) / 2  # Tâm X
                    cy = (by1 + by2) / 2  # Tâm Y
                    h  = by2 - by1        # Chiều cao ký tự
                    
                    # Thêm vào danh sách
                    list_chars.append({'cx': cx, 'cy': cy, 'label': label, 'h': h})

            # ==========================================================
            # BƯỚC 4: SẮP XẾP & HIỂN THỊ (HELPER)
            # ==========================================================
            final_plate = helper.read_plate(list_chars)

            if final_plate:
                # In biển số lên màn hình (phía trên khung detect)
                cv2.putText(frame, final_plate, (px1, py1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                print(f"{final_plate}")
                
                # (Tùy chọn) Hiện ảnh crop để debug xem nó đã xoay thẳng chưa
                # cv2.imshow('Debug Crop', crop_img)

    # Hiển thị Camera chính
    cv2.imshow('PBL4 - License Plate Recognition', frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()