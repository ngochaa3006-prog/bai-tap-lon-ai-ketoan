import pandas as pd
import os

def main():
    print("==========================================")
    print("   TIẾN HÀNH LÀM SẠCH DỮ LIỆU KẾ TOÁN     ")
    print("==========================================")
    
    raw_path = os.path.join('data', 'dulieuketoan_raw.csv')
    clean_path = os.path.join('data', 'dulieuketoan_clean.csv')
    
    if not os.path.exists(raw_path):
        print(f"❌ LỖI: Không tìm thấy file dữ liệu thô tại: {raw_path}")
        return

    # Đọc file thô
    df_raw = pd.read_csv(raw_path, encoding='utf-8-sig')
    
    # Tiền xử lý cơ bản: Xử lý các ô trống (Missing values) bằng cách điền số 0
    df_raw.fillna(0, inplace=True)
    
    # Lưu ra file sạch
    df_raw.to_csv(clean_path, index=False, encoding='utf-8-sig')
    print(f"✅ THÀNH CÔNG: Đã xuất file dữ liệu sạch tại: {clean_path}")
    print("==========================================")

if __name__ == "__main__":
    main()