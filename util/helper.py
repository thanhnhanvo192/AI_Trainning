from sklearn.cluster import KMeans
import numpy as np

def read_plate(yolo_license_plate, im):
    # ==== 1. YOLO OCR ====
    results = yolo_license_plate.predict(im, conf=0.4, verbose=False)

    bb_list = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_license_plate.names[cls] if hasattr(yolo_license_plate, 'names') else str(cls)
            bb_list.append([x1, y1, x2, y2, conf, cls, label])

    if len(bb_list) == 0:
        return "unknown"

    # ==== 2. Tạo center_list ====
    center_list = []
    for bb in bb_list:
        x_center = (bb[0] + bb[2]) / 2
        y_center = (bb[1] + bb[3]) / 2
        center_list.append([x_center, y_center, bb[-1]])

    # ==== 3. Nếu chỉ 1 dòng ====
    ys = [c[1] for c in center_list]
    if max(ys) - min(ys) < 20:  # biển 1 dòng
        line = sorted(center_list, key=lambda x: x[0])
        return "".join([c[2] for c in line])

    # ==== 4. Biển 2 dòng → phân cụm K-means theo trục Y ====
    ys_np = np.array([[c[1]] for c in center_list])
    kmeans = KMeans(n_clusters=2, n_init=10).fit(ys_np)

<<<<<<< HEAD
    labels = kmeans.labels_

    line1 = []
    line2 = []

    for i, c in enumerate(center_list):
        if labels[i] == 0:
            line1.append(c)
        else:
            line2.append(c)

    # ==== 5. sắp theo X trong từng dòng ====
    line1 = sorted(line1, key=lambda x: x[0])
    line2 = sorted(line2, key=lambda x: x[0])

    # ==== 6. xác định dòng nào là dòng trên ====
    if np.mean([c[1] for c in line1]) > np.mean([c[1] for c in line2]):
        line1, line2 = line2, line1

    # ==== 7. ghép kết quả dòng trên + dòng dưới ====
    plate = "".join([c[2] for c in line1]) + "".join([c[2] for c in line2])
    return plate
=======
    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += str(l[2])
    return license_plate
>>>>>>> 225b004ca5c5a0e829bbf27d4187b67e8ab43829
