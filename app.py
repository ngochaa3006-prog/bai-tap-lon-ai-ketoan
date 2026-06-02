import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.utils import preprocess_and_clean, apply_pca_algorithm
from utils.thuytrang_benchmark import tinh_trung_binh_nganh

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
            # =======================================================
# NHIỆM VỤ THÀNH VIÊN 2 (THÙY TRANG) - ĐỐI CHIẾU TRUNG BÌNH NGÀNH
# =======================================================
with tab2:
    st.header("🎯 Nhiệm vụ Thành viên 2: Phân tích & Đối chiếu Trung bình Ngành")
    st.caption("Giải pháp quản trị: Xác định vị thế tài chính và cảnh báo sớm rủi ro cấu trúc vốn")
    st.write("---")

    try:
        df_tab2 = pd.read_csv("data/dulieuketoan_clean.csv")
        # Lọc lấy danh sách các cột dạng số thực sự để tính toán
        numeric_cols_tab2 = df_tab2.select_dtypes(include=['number']).columns.tolist()
        if 'Bankrupt?' in numeric_cols_tab2:
            numeric_cols_tab2.remove('Bankrupt?') # Loại bỏ biến mục tiêu ra khỏi chỉ số đối chiếu
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file dữ liệu 'dulieuketoan_clean.csv' trong thư mục data.")
        df_tab2 = pd.DataFrame()
        numeric_cols_tab2 = []

    if not df_tab2.empty and len(numeric_cols_tab2) > 0:
        
        # ----------------------------------------------------
        # KHU VỰC ĐIỀU KHIỂN TƯƠNG TÁC (CONTROL PANEL)
        # ----------------------------------------------------
        st.subheader("🛠️ Bộ điều khiển phân tích chuyên sâu")
        ctrl_col1, ctrl_col2 = st.columns(2)
        
        with ctrl_col1:
            # Cho phép chọn bất kỳ doanh nghiệp nào theo ID dòng
            danh_sach_dn = list(range(len(df_tab2)))
            idx_doanh_nghiep = st.selectbox(
                "1. Chọn doanh nghiệp mẫu cần kiểm tra:", 
                options=danh_sach_dn,
                format_func=lambda x: f"Doanh nghiệp mã số ứng viên #{x}"
            )
            
        with ctrl_col2:
            # Tự động tìm kiếm các cột liên quan đến Nợ/Đòn bẩy để ưu tiên hiển thị trước
            default_index = 0
            for i, col in enumerate(numeric_cols_tab2):
                if 'debt' in col.lower() or 'no' in col.lower():
                    default_index = i
                    break
            
            # Cho phép chọn bất kỳ chỉ số nào trong data để đối chiếu với toàn ngành
            target_metric = st.selectbox(
                "2. Chọn chỉ số tài chính cần đối chiếu Benchmark:", 
                options=numeric_cols_tab2,
                index=default_index
            )

        st.write("---")
        
        # ----------------------------------------------------
        # THUẬT TOÁN XỬ LÝ & TÍNH TOÁN (BACKEND TRUY XUẤT)
        # ----------------------------------------------------
        moc_trung_binh = tinh_trung_binh_nganh(df_tab2, target_metric)
        gia_tri_dn = float(df_tab2[target_metric].iloc[idx_doanh_nghiep])
        chenh_lech = gia_tri_dn - moc_trung_binh
        
        # Xác định logic tài chính: Hệ số nợ cao = nguy cơ; ROA/ROE cao = tốt
        is_leverage_metric = any(keyword in target_metric.lower() for keyword in ['debt', 'no', 'liability', 'dependency'])
        
        if is_leverage_metric:
            color_logic = "inverse"  # Chỉ số nợ: Cao hơn ngành = Đỏ, Thấp hơn = Xanh
            is_risky = gia_tri_dn > moc_trung_binh
        else:
            color_logic = "normal"   # Chỉ số hiệu quả: Cao hơn = Xanh, Thấp hơn = Đỏ
            is_risky = gia_tri_dn < moc_trung_binh

        # ----------------------------------------------------
        # HIỂN THỊ THẺ KPI VÀ TRỰC QUAN HÓA
        # ----------------------------------------------------
        st.subheader(f"📊 Kết quả đối chiếu chỉ số: {target_metric}")
        
        view_col1, view_col2 = st.columns([1, 1.2])
        
        with view_col1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(
                label=f"🏢 Giá trị tại Doanh nghiệp #{idx_doanh_nghiep}", 
                value=f"{gia_tri_dn:.4f}", 
                delta=f"{chenh_lech:.4f} so với mức chung",
                delta_color=color_logic
            )
            st.metric(
                label="🏭 Mốc Trung bình Toàn Ngành (Benchmark)", 
                value=f"{moc_trung_binh:.4f}"
            )
            
        with view_col2:
            # Tạo biểu đồ trực quan hóa Benchmark riêng cho Tab 2
            fig_bench, ax_bench = plt.subplots(figsize=(6, 3.2))
            categories = ['Doanh nghiệp đang xét', 'Trung bình Toàn Ngành']
            values = [gia_tri_dn, moc_trung_binh]
            bar_colors = ['#FF4B4B' if is_risky else '#29B5E8', '#4A5568']
            
            bars = ax_bench.bar(categories, values, color=bar_colors, width=0.4, edgecolor='black', alpha=0.85)
            ax_bench.set_title(f"So sánh trực quan chỉ số {target_metric}", fontsize=10, fontweight='bold')
            ax_bench.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Thêm nhãn giá trị lên đầu cột biểu đồ
            for bar in bars:
                height = bar.get_height()
                ax_bench.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            st.pyplot(fig_bench)

        # ----------------------------------------------------
        # HỆ THỐNG CỐ VẤN TỰ ĐỘNG
        # ----------------------------------------------------
        st.write("---")
        st.markdown("### 💡 Đánh giá & Khuyến nghị Quản trị từ Chuyên gia:")
        
        with st.container():
            if is_leverage_metric:
                if gia_tri_dn > moc_trung_binh:
                    st.error(f"**🔴 CẢNH BÁO CẤU TRÚC VỐN:** Doanh nghiệp #{idx_doanh_nghiep} có chỉ số đòn bẩy tài chính ({gia_tri_dn:.4f}) vượt mức trung bình ngành ({moc_trung_binh:.4f}). Doanh nghiệp đang phụ thuộc lớn vào vốn vay ngoài, dẫn tới áp lực chi phí lãi vay tăng cao, làm thu hẹp biên lợi nhuận ròng và gia tăng rủi ro thanh khoản.")
                else:
                    st.success(f"**🟢 CẤU TRÚC VỐN AN TOÀN:** Chỉ số đòn bẩy tài chính ({gia_tri_dn:.4f}) của doanh nghiệp được kiểm soát an toàn dưới ngưỡng trung bình của ngành ({moc_trung_binh:.4f}). Doanh nghiệp có năng lực tự chủ tài chính tốt và rủi ro vỡ nợ thấp.")
            else:
                if gia_tri_dn >= moc_trung_binh:
                    st.success(f"**🟢 HIỆU QUẢ HOẠT ĐỘNG VƯỢT TRỘI:** Chỉ số hiệu năng kinh doanh ({gia_tri_dn:.4f}) cao hơn mặt bằng chung toàn ngành ({moc_trung_binh:.4f}). Chứng tỏ doanh nghiệp có lợi thế cạnh tranh cao và quản lý tài sản hiệu quả.")
                else:
                    st.error(f"**🔴 CẢNH BÁO HIỆU QUẢ HOẠT ĐỘNG:** Chỉ số năng lực ({gia_tri_dn:.4f}) đang chạy tụt hậu so với mốc trung bình toàn ngành ({moc_trung_binh:.4f}). Cần rà soát ngay các điểm nghẽn trong chuỗi vận hành và tái cấu trúc tài sản.")
                    
        with st.expander("📘 Đọc hiểu tài liệu: Tầm quan trọng của Mốc Trung bình Ngành (Benchmark)"):
            st.markdown("""
            Trong phân tích tài chính kế toán hiện đại, một con số đơn lẻ trên Báo cáo tài chính không mang nhiều ý nghĩa nếu không được đặt vào bối cảnh toàn ngành.
            * **Xác định vị thế:** Mốc trung bình ngành hoạt động như một hệ quy chiếu giúp nhà quản trị biết doanh nghiệp mình đang đứng ở đâu.
            * **Phát hiện bất thường:** Sự lệch pha quá lớn của một chỉ số so với Benchmark là tín hiệu cảnh báo sớm các rủi ro tiềm ẩn.
            """)