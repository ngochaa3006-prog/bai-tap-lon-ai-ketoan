import pandas as pd

def tinh_trung_binh_nganh(df: pd.DataFrame, ten_cot: str) -> float:
    """
    Tính toán mốc trung bình của ngành cho một chỉ số tài chính.
    Tự động bẫy lỗi dữ liệu rỗng hoặc sai định dạng chuỗi.
    """
    # 1. Bẫy lỗi: Kiểm tra xem cột được gọi có tồn tại trong bảng không
    if ten_cot not in df.columns:
        return 0.0
        
    # 2. Ép kiểu dữ liệu: Chuyển toàn bộ dữ liệu trong cột về dạng số thực (float)
    # Nếu có ô nào nhập sai (ví dụ chứa chữ cái), 'coerce' sẽ biến ô đó thành NaN
    du_lieu_so = pd.to_numeric(df[ten_cot], errors='coerce')
    
    # 3. Làm sạch: Lọc bỏ toàn bộ các ô trống (NaN) để không làm sai lệch kết quả
    cot_hop_le = du_lieu_so.dropna()
    
    # 4. Trả kết quả: Bẫy lỗi chia cho 0 nếu cột hoàn toàn trống
    if cot_hop_le.empty:
        return 0.0
        
    # Dùng hàm .mean() của Pandas để tính trung bình với độ chính xác cao nhất
    trung_binh = float(cot_hop_le.mean())
    return trung_binh