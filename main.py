import cv2
import time
import math
from ultralytics import YOLO

# --- CẤU HÌNH ĐƯỜNG DẪN MODEL ---
PATH_DETECT = 'model/best.pt'  # Model phát hiện biển số
PATH_OCR = 'model/ocr.pt'      # Model phát hiện ký tự (custom train)

# --- 1. KHỞI TẠO MODEL ---
print("1. Đang tải model Detect biển số...")
model_plate = YOLO(PATH_DETECT)

print("2. Đang tải model Detect ký tự...")
model_char = YOLO(PATH_OCR)

# --- HÀM SẮP XẾP KÝ TỰ (QUAN TRỌNG NHẤT) ---
def sort_characters(boxes):
    """
    Input: List các box ký tự [(x1, y1, x2, y2, class_name, conf), ...]
    Output: String biển số đã sắp xếp đúng thứ tự
    """
    if not boxes:
        return ""

    # Tính toán tọa độ trung tâm (cx, cy) cho mỗi ký tự
    # Cấu trúc item: [cx, cy, x1, y1, x2, y2, char_text]
    chars = []
    for box in boxes:
        x1, y1, x2, y2, cls, conf = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # Lấy tên class từ model (ví dụ: class 0 -> '0', class 10 -> 'A')
        char_text = model_char.names[int(cls)]
        chars.append([cx, cy, x1, y1, x2, y2, char_text])

    # --- XỬ LÝ BIỂN 2 DÒNG (BIỂN VUÔNG) ---
    # Logic: Nếu độ lệch chiều cao giữa các ký tự quá lớn -> Có 2 dòng
    # Sắp xếp tạm thời theo y (chiều cao) để tìm min/max
    chars.sort(key=lambda k: k[1]) 
    
    # Lấy trung bình chiều cao của tất cả ký tự (h = y2 - y1)
    avg_height = sum([(c[5] - c[3]) for c in chars]) / len(chars)
    
    # Nếu khoảng cách giữa ký tự cao nhất và thấp nhất > chiều cao trung bình * 0.7
    # -> Khả năng cao là biển 2 dòng
    if (chars[-1][1] - chars[0][1]) > (avg_height * 0.7):
        # Tách thành dòng 1 và dòng 2
        # Ngưỡng phân chia là trung bình cộng của y_min và y_max
        split_y = (chars[0][1] + chars[-1][1]) / 2
        
        row1 = [c for c in chars if c[1] < split_y]  # Dòng trên
        row2 = [c for c in chars if c[1] >= split_y] # Dòng dưới
        
        # Sắp xếp từng dòng từ trái sang phải (theo x)
        row1.sort(key=lambda k: k[0])
        row2.sort(key=lambda k: k[0])
        
        final_list = row1 + row2
    else:
        # --- XỬ LÝ BIỂN 1 DÒNG ---
        # Chỉ cần sắp xếp theo x (trái sang phải)
        chars.sort(key=lambda k: k[0])
        final_list = chars

    # Ghép thành chuỗi
    plate_str = "".join([c[6] for c in final_list])
    return plate_str

# --- MAIN LOOP ---
vid = cv2.VideoCapture(0) # Đổi thành 1 nếu dùng cam rời
# vid.set(3, 640)
# vid.set(4, 480)

print("--- HỆ THỐNG SẴN SÀNG (Nhấn 'q' để thoát) ---")

while True:
    ret, frame = vid.read()
    if not ret: break

    # 1. Detect Biển số
    results_plate = model_plate.predict(frame, conf=0.5, verbose=False)

    for r in results_plate:
        boxes = r.boxes
        for box in boxes:
            # Lấy tọa độ biển số
            px1, py1, px2, py2 = box.xyxy[0]
            px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

            # Cắt ảnh biển số (Crop)
            crop_img = frame[py1:py2, px1:px2]
            
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0: continue

            # 2. Detect Ký tự (trên ảnh Crop)
            # conf=0.25: Để thấp chút để bắt được hết các số mờ
            results_char = model_char.predict(crop_img, conf=0.35, verbose=False)
            
            char_boxes = []
            # Lấy thông tin các ký tự tìm được
            for rc in results_char:
                for cb in rc.boxes:
                    cx1, cy1, cx2, cy2 = cb.xyxy[0]
                    cls = cb.cls[0]
                    conf = cb.conf[0]
                    char_boxes.append([float(cx1), float(cy1), float(cx2), float(cy2), int(cls), float(conf)])

            # 3. Sắp xếp và ghép chuỗi
            final_text = sort_characters(char_boxes)

            # 4. Hiển thị kết quả
            # Vẽ khung biển số
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            
            # Vẽ chữ lên trên biển số
            cv2.putText(frame, final_text, (px1, py1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            print(f"Biển số: {final_text}")

            # (Debug) Hiện ảnh crop để xem model OCR nhìn thấy gì
            cv2.imshow('Crop Plate', crop_img)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()