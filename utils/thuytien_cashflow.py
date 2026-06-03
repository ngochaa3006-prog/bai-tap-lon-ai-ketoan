# utils/cashflow.py
import pandas as pd

def tinh_loi_nhuan_12_thang(von_nhap, lai_suat_nhap):
    """
    Hàm tính toán và dự phóng dòng tiền ngân sách phục hồi trong 12 tháng
    """
    # Chuyển đổi lãi suất năm sang lãi suất tháng (dạng thập phân)
    lai_suat_thang = (lai_suat_nhap / 100) / 12
    
    danh_sach_thang = []
    danh_sach_ngan_sach = []
    
    ngan_sach_tich_luy = von_nhap
    
    for i in range(1, 13):
        # Tính toán ngân sách tăng trưởng cộng dồn theo từng tháng
        ngan_sach_tich_luy = ngan_sach_tich_luy * (1 + lai_suat_thang)
        
        danh_sach_thang.append(f"Tháng {i}")
        danh_sach_ngan_sach.append(round(ngan_sach_tich_luy, 2))
        
    # Trả về định dạng DataFrame để Streamlit vẽ đồ thị
    df_ket_qua = pd.DataFrame({
        "Tháng": danh_sach_thang,
        "Ngân sách phục hồi (VND)": danh_sach_ngan_sach
    })
    
    return df_ket_qua