import cv2
import pathlib
from ultralytics import YOLO

try:
    import util.utils_rotate as utils_rotate
    import util.helper as helper
except:
    pass

# Cấu hình đường dẫn
PATH_DETECT = 'model/best.pt'
PATH_OCR = 'model/ocr.pt' # Hoặc ocr_best.onnx tùy bạn

# Load Model
print("Loading models...")
model_plate = YOLO(PATH_DETECT)
model_char = YOLO(PATH_OCR, task='detect')

# Mở Camera
vid = cv2.VideoCapture(0) # Đổi thành 1 nếu dùng cam rời

while True:
    ret, frame = vid.read()
    if not ret: break

    # 1. Detect Biển số
    results = model_plate.predict(frame, conf=0.5, verbose=False)

    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Vẽ khung
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Cắt ảnh
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size == 0: continue

            # 2. Xoay ảnh (Deskew) - RẤT QUAN TRỌNG
            try:
                # Xoay thẳng ảnh giúp tách dòng chuẩn hơn
                crop_img = utils_rotate.deskew(crop_img, change_cons=1, center_thres=1)
            except: 
                pass
            
            # 3. Đọc ký tự bằng Helper (Đã sửa logic)
            # Truyền thẳng model_char và ảnh crop vào
            final_plate = helper.read_plate(model_char, crop_img)

            if final_plate != "":
                # Hiển thị kết quả
                cv2.putText(frame, final_plate, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                print(f"Biển số: {final_plate}")
                
                cv2.imshow('Deskewed', crop_img)

    cv2.imshow('Nhan Dien Bien So', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vid.release()
cv2.destroyAllWindows()