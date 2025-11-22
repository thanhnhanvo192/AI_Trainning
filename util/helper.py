import math

# --- LOGIC CŨ (Giữ lại để tham khảo, nhưng không dùng để quyết định dòng nữa) ---
def linear_equation(x1, y1, x2, y2):
    if x2 - x1 == 0: return 0, 0
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

# --- HÀM CHÍNH ĐÃ ĐƯỢC TỐI ƯU ---
def read_plate(yolo_license_plate, im):
    # 1. DỰ ĐOÁN BẰNG YOLOv8 (Thay cho .pandas() của YOLOv5)
    try:
        # verbose=False để tắt log, conf=0.4 để lọc nhiễu
        results = yolo_license_plate.predict(im, conf=0.4, verbose=False)
    except:
        return ""

    # 2. CHUYỂN ĐỔI KẾT QUẢ SANG LIST
    # Cấu trúc center_list: [center_x, center_y, label_name, height]
    center_list = []
    y_sum = 0
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls = int(box.cls[0])
            
            # Lấy tên label
            if hasattr(yolo_license_plate, 'names'):
                label = yolo_license_plate.names[cls]
            else:
                label = str(cls)
            
            x_c = (x1 + x2) / 2
            y_c = (y1 + y2) / 2
            h = y2 - y1
            
            y_sum += y_c
            center_list.append([x_c, y_c, label, h])

    # Kiểm tra số lượng ký tự
    if len(center_list) == 0: return ""

    # 3. THUẬT TOÁN XÁC ĐỊNH 1 DÒNG HAY 2 DÒNG (MỚI)
    # Sắp xếp theo Y để tìm chữ cao nhất và thấp nhất
    center_list.sort(key=lambda x: x[1])
    min_y = center_list[0][1]
    max_y = center_list[-1][1]
    
    # Tính chiều cao trung bình của ký tự
    avg_height = sum([c[3] for c in center_list]) / len(center_list)
    
    # LOGIC QUAN TRỌNG: Nếu khoảng cách Y lớn hơn 60% chiều cao chữ -> 2 DÒNG
    if (max_y - min_y) > (avg_height * 0.6):
        LP_type = "2"
    else:
        LP_type = "1"

    # 4. SẮP XẾP VÀ GHÉP CHUỖI
    license_plate = ""
    
    if LP_type == "2":
        # Tính đường trung bình chia đôi dòng
        y_mean = (min_y + max_y) / 2
        
        line_1 = []
        line_2 = []
        for c in center_list:
            if c[1] > y_mean:
                line_2.append(c) # Dòng dưới
            else:
                line_1.append(c) # Dòng trên
        
        # Sắp xếp từng dòng từ Trái qua Phải (theo x_c)
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += str(l1[2])
        
        # license_plate += "-" # Bỏ comment nếu muốn dấu gạch
        
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += str(l2[2])
            
    else:
        # 1 dòng: Sắp xếp từ Trái qua Phải
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += str(l[2])

    return license_plate