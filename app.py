import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
def tinh_giai_thua(n):
    return math.factorial(n)
from utils.thuytien_cashflow import tinh_loi_nhuan_12_thang
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
# NHIỆM VỤ THÀNH VIÊN 2 (THÙY TRANG) - NHẬP DỮ LIỆU & THẨM ĐỊNH CHUYÊN SÂU
# =======================================================
with tab2:
    st.header("🎯 Phân tích Đối chiếu Benchmark & Thẩm định Chuyên sâu")
    st.caption("Giải pháp quản trị: Cho phép nhập chỉ số tài chính động để tự động xuất báo cáo cảnh báo sớm rủi ro")
    st.write("---")

    try:
        df_tab2 = pd.read_csv("data/dulieuketoan_clean.csv")
        # Lọc lấy danh sách các cột dạng số thực sự để tính toán trung bình ngành
        numeric_cols_tab2 = df_tab2.select_dtypes(include=['number']).columns.tolist()
        if 'Bankrupt?' in numeric_cols_tab2:
            numeric_cols_tab2.remove('Bankrupt?') 
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file dữ liệu 'dulieuketoan_clean.csv' trong thư mục data.")
        df_tab2 = pd.DataFrame()
        numeric_cols_tab2 = []

    if not df_tab2.empty and len(numeric_cols_tab2) > 0:
        
        # ----------------------------------------------------
        # KHU VỰC NHẬP DỮ LIỆU ĐỘNG (INTERACTIVE INPUT PANEL)
        # ----------------------------------------------------
        st.subheader("✍️ Nhập số liệu Doanh nghiệp cần Thẩm định")
        
        ctrl_col1, ctrl_col2 = st.columns(2)
        
        with ctrl_col1:
            # Chọn chỉ số cần phân tích trước
            target_metric = st.selectbox(
                "1. Chọn chỉ số tài chính cần kiểm tra Benchmark:", 
                options=numeric_cols_tab2
            )
            # Tính toán ngay mốc trung bình ngành từ file CSV để làm hệ quy chiếu
            moc_trung_binh = tinh_trung_binh_nganh(df_tab2, target_metric)
            st.info(f"🏭 Mức trung bình toàn ngành hiện tại: **{moc_trung_binh:.4f}**")
            
        with ctrl_col2:
            # KHU VỰC CHO PHÉP TỰ GÕ SỐ: Dùng st.number_input thay vì selectbox
            # Để mặc định ban đầu bằng chính mức trung bình ngành cho đẹp giao diện
            gia_tri_dn = st.number_input(
                "2. Nhập giá trị thực tế của Doanh nghiệp bạn (Gõ số vào đây):", 
                value=round(moc_trung_binh, 4),
                format="%.4f",
                step=0.0001
            )

        st.write("---")
        
        # ----------------------------------------------------
        # THUẬT TOÁN XỬ LÝ ĐỘ LỆCH VÀ PHÂN LOẠI BIẾN TÀI CHÍNH
        # ----------------------------------------------------
        chenh_lech = gia_tri_dn - moc_trung_binh
        phan_tram_lech = (chenh_lech / moc_trung_binh) * 100 if moc_trung_binh != 0 else 0
        
        # Xác định logic tài chính: Nhóm Đòn bẩy/Nợ vay (Càng cao càng rủi ro) và Nhóm Hiệu năng/Sinh lời (Càng cao càng tốt)
        is_leverage_metric = any(keyword in target_metric.lower() for keyword in ['debt', 'no', 'liability', 'dependency', 'borrow'])
        
        if is_leverage_metric:
            color_logic = "inverse"  # Chỉ số nợ: Cao hơn ngành = Đỏ (Nguy hiểm), Thấp hơn = Xanh (An toàn)
            is_risky = gia_tri_dn > moc_trung_binh
        else:
            color_logic = "normal"   # Chỉ số hiệu quả: Cao hơn = Xanh (Tốt), Thấp hơn = Đỏ (Kém)
            is_risky = gia_tri_dn < moc_trung_binh

        # ----------------------------------------------------
        # HIỂN THỊ TRỰC QUAN HÓA THEO DỮ LIỆU NHẬP
        # ----------------------------------------------------
        st.subheader("📊 Trực quan hóa vị thế Doanh nghiệp so với Ngành")
        
        view_col1, view_col2 = st.columns([1, 1.3])
        
        with view_col1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(
                label="🏢 Chỉ số Doanh nghiệp bạn gõ", 
                value=f"{gia_tri_dn:.4f}", 
                delta=f"{phan_tram_lech:.2f}% so với ngành",
                delta_color=color_logic
            )
            st.metric(
                label="🏭 Benchmark (Trung bình Ngành)", 
                value=f"{moc_trung_binh:.4f}"
            )
            
        with view_col2:
            # Vẽ biểu đồ thanh ngang so sánh trực quan dữ liệu người dùng nhập vào
            fig_bench, ax_bench = plt.subplots(figsize=(6, 3))
            categories = ['Doanh nghiệp của bạn', 'Trung bình Ngành']
            values = [gia_tri_dn, moc_trung_binh]
            bar_colors = ['#FF4B4B' if is_risky else '#29B5E8', '#4A5568']
            
            bars = ax_bench.barh(categories, values, color=bar_colors, height=0.4, edgecolor='black', alpha=0.9)
            ax_bench.set_title(f"Đối chiếu dữ liệu nhập vào của chỉ số {target_metric}", fontsize=9, fontweight='bold')
            ax_bench.grid(True, axis='x', linestyle='--', alpha=0.5)
            
            for bar in bars:
                width = bar.get_width()
                ax_bench.annotate(f' {width:.4f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9, fontweight='bold')
            
            st.pyplot(fig_bench)

        # ----------------------------------------------------
        # HỆ THỐNG XUẤT BÁO CÁO THẨM ĐỊNH CHUYÊN SÂU ĐỘNG
        # ----------------------------------------------------
        st.write("---")
        st.subheader("📝 BÁO CÁO THẨM ĐỊNH TÀI CHÍNH & KHUYẾN NGHỊ QUẢN TRỊ CHUYÊN SÂU")
        
        with st.container():
            if is_leverage_metric:
                if gia_tri_dn > moc_trung_binh:
                    st.error(f"""
                    **🔴 CẢNH BÁO RỦI RO CẤU TRÚC VỐN (Đòn bẩy tài chính vượt ngưỡng an toàn):**
                    * **Thực trạng định lượng:** Chỉ số đòn bẩy tài chính bạn nhập vào đang cao hơn mức trung bình ngành **{abs(phan_tram_lech):.2f}%**.
                    * **Hệ quả Kế toán Quản trị:** Doanh nghiệp đang phụ thuộc quá mức vào nguồn vốn vay từ bên ngoài. Việc này đẩy **Chi phí sử dụng vốn bình quân (WACC)** lên cao, tạo áp lực trả gốc và lãi vay đè nặng lên dòng tiền thuần từ hoạt động kinh doanh, làm suy giảm nghiêm trọng biên lợi nhuận ròng và gia tăng rủi ro kiệt quệ tài chính (Financial Distress).
                    * **Giải pháp khuyến nghị:** Hội đồng quản trị cần hạn chế ký kết các hợp đồng vay ngắn hạn mới; xem xét phương án tái cấu trúc nguồn vốn bằng cách phát hành thêm cổ phiếu phổ thông để tăng tỷ trọng Vốn chủ sở hữu, hoặc thương lượng chuyển đổi nợ thành cổ phần để đưa hệ số an toàn về mức Benchmark ngành.
                    """)
                else:
                    st.success(f"""
                    **🟢 VỊ THẾ AN TOÀN VỐN (Cấu trúc vốn tự chủ vững chắc):**
                    * **Thực trạng định lượng:** Chỉ số rủi ro nợ vay thấp hơn mức trung bình ngành **{abs(phan_tram_lech):.2f}%**.
                    * **Hệ quả Kế toán Quản trị:** Doanh nghiệp kiểm soát rất tốt rủi ro vỡ nợ, có mức độ tự chủ tài chính cao, tạo lòng tin lớn cho các tổ chức tín dụng và nhà đầu tư dài hạn. Biên độ an toàn tài chính rộng giúp doanh nghiệp chống chịu tốt trước các cú sốc thắt chặt tiền tệ của thị trường.
                    * **Lưu ý tối ưu:** Tuy nhiên, đứng dưới góc độ Kế toán Quản trị chuyên sâu, việc duy trì tỷ lệ nợ quá thấp có thể khiến doanh nghiệp bỏ lỡ lợi ích từ **Lá chắn thuế từ lãi vay (Tax Shield)** và không tối ưu hóa được hiệu ứng đòn bẩy để gia tăng tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE). Doanh nghiệp có thể cân nhắc sử dụng vốn vay một cách có kiểm soát cho các dự án mở rộng có NPV dương.
                    """)
            else:
                if gia_tri_dn >= moc_trung_binh:
                    st.success(f"""
                    **🟢 LỢI THẾ CẠNH TRANH VÀ HIỆU QUẢ HOẠT ĐỘNG XUẤT SẮC:**
                    * **Thực trạng định lượng:** Chỉ số năng lực hoạt động/hiệu quả sinh lời vượt mức trung bình ngành **{abs(phan_tram_lech):.2f}%**.
                    * **Hệ quả Kế toán Quản trị:** Chứng tỏ doanh nghiệp sở hữu quy trình vận hành tối ưu, quản lý chi phí chặt chẽ (giảm thiểu giá vốn hàng bán COGS và chi phí quản lý doanh nghiệp OPEX), hoặc tốc độ vòng quay tài sản rất nhanh. Khả năng tạo ra thặng dư kinh tế trên mỗi đồng vốn đầu tư tốt hơn mặt bằng chung của thị trường.
                    * **Giải pháp định hướng:** Cần tiếp tục duy trì chiến lược tối ưu hóa này, đầu tư sâu vào hệ thống chuyển đổi số để khóa chặt lợi thế cạnh tranh về chi phí, đồng thời có thể áp dụng chính sách chi trả cổ tức cởi mở hơn để thu hút dòng vốn.
                    """)
                else:
                    st.error(f"""
                    **🔴 CẢNH BÁO SUY GIẢM HIỆU NĂNG HOẠT ĐỘNG VÀ NĂNG LỰC SINH LỜI:**
                    * **Thực trạng định lượng:** Chỉ số hiệu năng hoạt động hiện tại đang chạy tụt hậu so với mặt bằng chung toàn ngành **{abs(phan_tram_lech):.2f}%**.
                    * **Hệ quả Kế toán Quản trị:** Tín hiệu này cảnh báo việc sử dụng tài sản đang bị lãng phí, ứ đọng hàng tồn kho hoặc phát sinh nợ xấu trong các khoản phải thu. Biên lợi nhuận bị thu hẹp do không kiểm soát tốt chi phí đầu vào hoặc do năng lực cạnh tranh sản phẩm trên thị trường bị suy giảm.
                    * **Giải pháp khuyến nghị:** Ban điều hành cần bóc tách chỉ số này bằng mô hình DuPont để định vị chính xác điểm nghẽn nằm ở khâu Quản lý tài sản hay khâu Biên lợi nhuận thuần. Tiến hành rà soát cắt giảm triệt để các chi phí bất hợp lý và khẩn trương thu hồi công nợ quá hạn.
                    """)
                # ==========================================================
    # KHU VỰC CỦA THÀNH VIÊN 3 (ĐÃ ĐƯỢC TÁCH HÀM SANG UTILS/CASHFLOW.PY)
    # ==========================================================

    st.markdown("---") # Tạo một đường kẻ ngang phân cách
    st.header("📈 Nhiệm vụ 3: Tái cấu trúc Dòng tiền & Giả lập Phục hồi")
    st.info("Nhập các thông số vốn cứu trợ và lãi suất để theo dõi biểu đồ dự phóng 12 tháng.")

    # Thiết kế form nhập liệu chia làm 2 cột
    col_member3_1, col_member3_2 = st.columns(2)

    with col_member3_1:
        von_nhap = st.number_input(
            "Số vốn cứu trợ ban đầu (VND):", 
            min_value=0.0, 
            value=500000000.0, # Giá trị mặc định 500 triệu
            step=50000000.0,
            format="%f"
        )
        
    with col_member3_2:
        lai_suat_nhap = st.number_input(
            "Lãi suất sinh lời kỳ vọng (%/năm):", 
            min_value=0.0, 
            max_value=100.0, 
            value=10.0, # Giá trị mặc định 10%
            step=0.5
        )
        
    # Nút bấm kích hoạt tính toán và vẽ đồ thị
    if st.button("Chạy Giả Lập Phục Hồi Dòng Tiền", type="primary"):
        # Gọi hàm xử lý tính toán từ file utils/thuytien_cashflow.py
        df_ket_qua = tinh_loi_nhuan_12_thang(von_nhap, lai_suat_nhap)
        
        st.subheader("Bảng dự phóng xu hướng ngân sách phục hồi")
        st.line_chart(data=df_ket_qua, x="Tháng", y="Ngân sách phục hồi (VND)")
        
        with st.expander("🔍 Chi tiết số liệu kế toán từng tháng"):
            st.dataframe(df_ket_qua, use_container_width=True)
            st.write("---")
st.header("🔥 Stress Testing")

so_rui_ro = st.slider(
    "Chọn số lượng rủi ro",
    min_value=1,
    max_value=10,
    value=3
)

so_kich_ban = tinh_giai_thua(so_rui_ro)

st.success(
    f"📊 Số lượng kịch bản khủng hoảng có thể xảy ra: {so_kich_ban}"
)

if so_kich_ban > 1000:
    st.error(
        "⚠️ CẢNH BÁO: Số lượng kịch bản khủng hoảng rất lớn!"
    )
      
