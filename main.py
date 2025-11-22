import cv2
import pathlib
from ultralytics import YOLO

# Import helper và utils_rotate
try:
    import util.helper as helper
    import util.utils_rotate as utils_rotate
except ImportError:
    print("Lỗi import file bổ trợ!")
    exit()

# Cấu hình
PATH_DETECT = 'model/detect_best.pt'
PATH_OCR = 'model/ocr_best.onnx'

# Load Model
print("Loading Models...")
model_plate = YOLO(PATH_DETECT)
model_char = YOLO(PATH_OCR, task='detect') # Đây chính là biến sẽ truyền vào helper

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if not ret: break

    # Detect Biển
    results = model_plate.predict(frame, conf=0.5, verbose=False)

    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size == 0: continue

            # 1. Xoay ảnh (Bắt buộc để logic đường thẳng của helper hoạt động tốt)
            try:
                crop_img = utils_rotate.deskew(crop_img, change_cons=1, center_thres=1)
            except: pass

            # 2. Gọi Helper theo đúng chuẩn cũ (Model, Ảnh)
            # helper sẽ tự detect bên trong nó bằng model_char
            final_plate = helper.read_plate(model_char, crop_img)

            if final_plate != "unknown":
                cv2.putText(frame, final_plate, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                print(f"Biển số: {final_plate}")
                
                cv2.imshow('Deskewed', crop_img)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vid.release()
cv2.destroyAllWindows()