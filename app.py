import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.utils import preprocess_and_clean, apply_pca_algorithm

# Cấu hình trang web rộng rãi, đẹp mắt
st.set_page_config(page_title="Dự án AI Kế Toán - Nhóm 1", layout="wide")

st.title("📊 HỆ THỐNG TRỰC QUAN HÓA & PHÂN TÍCH AI KẾ TOÁN")
st.write("---")

# Phân chia Tab theo yêu cầu Bài Tập Lớn
tab1, tab2 = st.tabs(["Tab 1: Khám phá Dữ liệu (EDA) & PCA", "Tab 2: Tính năng thành viên khác (Trống)"])

with tab1:
    st.header("🎯 Nhiệm vụ Thành viên 1: Khám phá dữ liệu & Thuật toán PCA")
    
    # Đường dẫn file dữ liệu sạch đã tạo từ bước trước
    clean_path = os.path.join('data', 'dulieuketoan_clean.csv')
    
    if not os.path.exists(clean_path):
        st.error("❌ Không tìm thấy file dữ liệu sạch! Vui lòng chạy lệnh `python clean_data.py` trước.")
    else:
        # Gọi hàm xử lý từ utils.py
        df, df_numeric, scaled_data, numeric_cols = preprocess_and_clean(clean_path)
        
        # ----------------------------------------------------
        # 1. Hiển thị bảng dữ liệu kế toán ban đầu
        # ----------------------------------------------------
        st.subheader("1. Hiển thị bảng dữ liệu kế toán sạch")
        st.dataframe(df, use_container_width=True)
        st.info(f"💡 Hệ thống đã tự động nhận diện được {df.shape[0]} biểu mẫu với {len(numeric_cols)} chỉ số kế toán dạng số.")
        
        # ----------------------------------------------------
        # 2. Vẽ biểu đồ trực quan hóa dữ liệu (EDA)
        # ----------------------------------------------------
        st.write("---")
        st.subheader("2. Biểu đồ trực quan hóa khám phá dữ liệu (EDA)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Phân phối tần suất của chỉ số kế toán:**")
            selected_col = st.selectbox("Chọn một chỉ số để xem đồ thị phân phối:", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[selected_col], kde=True, ax=ax, color='skyblue')
            ax.set_title(f"Phân phối của {selected_col}")
            st.pyplot(fig)
            
        with col2:
            st.write("**Ma trận tương quan (Trực quan hóa 8 chỉ số đầu tiên để tránh rối mắt):**")
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            # Vì bộ dữ liệu có rất nhiều cột, vẽ hết sẽ bị khít chữ nên ta lấy 8 cột đầu làm mẫu đại diện
            sample_corr = df_numeric.iloc[:, :8]
            sns.heatmap(sample_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2, annot_kws={"size": 8})
            ax2.set_title("Correlation Matrix (Top 8 Indicators)")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            st.pyplot(fig2)
            
        # ----------------------------------------------------
        # 3. Áp dụng thuật toán giảm chiều dữ liệu PCA
        # ----------------------------------------------------
        st.write("---")
        st.subheader("3. Ứng dụng thuật toán giảm chiều dữ liệu PCA")
        
        # Thanh kéo chọn số lượng chiều (Thành phần chính)
        n_comp = st.slider("Chọn số lượng thành phần chính (Components):", 2, min(5, len(numeric_cols)), 2)
        
        # Chạy thuật toán PCA
        pca_result, explained_variance, pca = apply_pca_algorithm(scaled_data, n_components=n_comp)
        
        # Hiển thị độ hiệu quả giữ thông tin của mô hình PCA
        total_variance = sum(explained_variance) * 100
        st.success(f"⚡ Thuật toán hoàn thành! Lựa chọn {n_comp} thành phần chính giúp giữ lại được {total_variance:.2f}% lượng thông tin gốc của toàn bộ các chỉ số kế toán.")
        
        # Trực quan hóa không gian dữ liệu sau giảm chiều lên đồ thị 2D
        if n_comp >= 2:
            st.write("**Đồ thị không gian giảm chiều dữ liệu kế toán (PC1 vs PC2):**")
            fig3, ax3 = plt.subplots(figsize=(7, 4.5))
            
            # Tô màu các chấm dữ liệu theo trạng thái Phá sản (Bankrupt?) để đồ thị cực kỳ chuyên nghiệp
            if 'Bankrupt?' in df.columns:
                scatter = ax3.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Bankrupt?'], cmap='bwr', alpha=0.6, edgecolors='k', s=30)
                legend = ax3.legend(*scatter.legend_elements(), title="Trạng thái Phá sản (0: An toàn, 1: Nguy cơ)")
                ax3.add_artist(legend)
            else:
                ax3.scatter(pca_result[:, 0], pca_result[:, 1], c='teal', alpha=0.6, edgecolors='k', s=30)
                
            ax3.set_title('Biểu đồ phân tích không gian PCA 2D')
            ax3.set_xlabel(f'Thành phần chính 1 (PC1) - Giải thích {explained_variance[0]*100:.1f}%')
            ax3.set_ylabel(f'Thành phần chính 2 (PC2) - Giải thích {explained_variance[1]*100:.1f}%')
            ax3.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig3)