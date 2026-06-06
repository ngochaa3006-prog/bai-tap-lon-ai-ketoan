import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_and_clean(file_path):
    """
    Đọc dữ liệu sạch, lọc các cột số và chuẩn hóa dữ liệu trước khi đưa vào PCA.
    """
    # Đọc file dữ liệu sạch
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 1. Tự động lọc lấy các cột dữ liệu dạng số (PCA chỉ tính toán được trên số)
    # Loại bỏ cột mục tiêu 'Bankrupt?' ra khỏi danh sách tính toán PCA để khách quan
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in all_numeric if col != 'Bankrupt?']
    
    df_numeric = df[numeric_cols]
    
    # 2. Chuẩn hóa dữ liệu (Z-score Scaling) - Bước BẮT BUỘC cực kỳ quan trọng trước khi làm PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    return df, df_numeric, scaled_data, numeric_cols

def apply_pca_algorithm(scaled_data, n_components=2):
    """
    Hàm thực hiện thuật toán PCA để giảm chiều dữ liệu kế toán phức tạp
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Tỷ lệ phương sai được giải thích bởi các thành phần chính
    explained_variance = pca.explained_variance_ratio_
    
    return pca_result, explained_variance, pca
