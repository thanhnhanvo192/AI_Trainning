# FILE: helper.py
def read_plate(chars):
    """
    Hàm sắp xếp ký tự biển số xe (Hỗ trợ 1 dòng và 2 dòng).
    
    Input: 
        chars: List các dictionary [{'cx': float, 'cy': float, 'label': str, 'h': float}, ...]
    
    Output: 
        String biển số đã sắp xếp hoàn chỉnh.
    """
    # Nếu ít hơn 3 ký tự thì coi như chưa đọc được hoặc nhiễu
    if len(chars) < 3: return ""

    # --- BƯỚC 1: SẮP XẾP THEO CHIỀU DỌC (Y) ---
    # Để tìm xem ký tự nào nằm trên, ký tự nào nằm dưới
    chars.sort(key=lambda k: k['cy'])
    
    # --- BƯỚC 2: THUẬT TOÁN TÁCH DÒNG (MAX GAP) ---
    # Tìm khoảng cách lớn nhất giữa tâm của 2 ký tự liền kề theo chiều dọc
    max_gap = 0
    split_index = 0
    
    for i in range(1, len(chars)):
        # Khoảng cách từ ký tự hiện tại (i) so với ký tự ngay trên nó (i-1)
        gap = chars[i]['cy'] - chars[i-1]['cy']
        
        if gap > max_gap:
            max_gap = gap
            split_index = i

    # --- BƯỚC 3: QUYẾT ĐỊNH 1 HAY 2 DÒNG ---
    # Tính chiều cao trung bình của các ký tự để làm thước đo
    avg_height = sum([c['h'] for c in chars]) / len(chars)
    
    # LOGIC QUAN TRỌNG:
    # Nếu khoảng trống lớn nhất (max_gap) lớn hơn 60% chiều cao trung bình của một con chữ
    # -> Kết luận: ĐÂY LÀ BIỂN 2 DÒNG.
    is_two_lines = max_gap > (avg_height * 0.6)

    final_string = ""
    
    if is_two_lines:
        # --- XỬ LÝ BIỂN 2 DÒNG ---
        # Cắt danh sách tại vị trí khoảng trống lớn nhất
        line_1 = chars[:split_index] # Nhóm trên
        line_2 = chars[split_index:] # Nhóm dưới
        
        # Trong mỗi dòng, sắp xếp từ TRÁI sang PHẢI (theo cx)
        line_1.sort(key=lambda k: k['cx'])
        line_2.sort(key=lambda k: k['cx'])
        
        # Ghép chuỗi
        str1 = "".join([str(c['label']) for c in line_1])
        str2 = "".join([str(c['label']) for c in line_2])
        
        # Kết quả: Dòng trên + Dòng dưới (VD: 59F1 + 12345)
        final_string = str1 + str2 
        
    else:
        # --- XỬ LÝ BIỂN 1 DÒNG ---
        # Chỉ cần sắp xếp từ Trái sang Phải
        chars.sort(key=lambda k: k['cx'])
        final_string = "".join([str(c['label']) for c in chars])

    return final_string
