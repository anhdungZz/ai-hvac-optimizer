import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="HVAC AI Optimizer - Dự Báo Phụ Tải & Tối Ưu Năng Lượng",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #2563EB;
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #6B7280;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #4B5563;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(rel_path):
    candidates = [
        rel_path,
        os.path.join("..", rel_path),
        os.path.join(os.path.dirname(__file__), rel_path),
        os.path.join(os.path.dirname(__file__), "..", rel_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return rel_path

@st.cache_resource
def load_models():
    hl_path = get_asset_path("models/best_model_hl.joblib")
    cl_path = get_asset_path("models/best_model_cl.joblib")
    p90_hl_path = get_asset_path("models/p90_model_hl.joblib")
    p90_cl_path = get_asset_path("models/p90_model_cl.joblib")
    
    m_hl = joblib.load(hl_path) if os.path.exists(hl_path) else None
    m_cl = joblib.load(cl_path) if os.path.exists(cl_path) else None
    m_p90_hl = joblib.load(p90_hl_path) if os.path.exists(p90_hl_path) else None
    m_p90_cl = joblib.load(p90_cl_path) if os.path.exists(p90_cl_path) else None
    return m_hl, m_cl, m_p90_hl, m_p90_cl

model_hl, model_cl, model_p90_hl, model_p90_cl = load_models()

# Header Section
st.markdown('<p class="main-header">HVAC AI Optimizer: Dự Báo Phụ Tải Nhiệt & Tối Ưu Năng Lượng Công Trình</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Đồ án môn học: Trí tuệ Nhân tạo Ứng dụng & Khoa học Dữ liệu | <b>Nhóm tác giả: Nguyễn Huỳnh Anh Dũng (2042250036 - ĐH Công Thương TP.HCM - Khoa CN Cơ khí) — Đặng Yến Thư (2421001883 - ĐH Tài Chính Marketing - Khoa CN Tài chính Ngân hàng)</b></p>', unsafe_allow_html=True)

# Sidebar: Input parameters
st.sidebar.header("Thông số Hình học & Kiến trúc")

# Danh sách 12 khối hình học chuẩn mực ENB2012
STANDARD_SHAPES = [
    {"id": 1, "name": "Khối 1: RC=0.62 | 1 tầng (3.5m) | Mái 220.5m² | Tường 367.5m²", "rc": 0.62, "sa": 808.5, "wa": 367.5, "ra": 220.5, "height": 3.5},
    {"id": 2, "name": "Khối 2: RC=0.64 | 1 tầng (3.5m) | Mái 220.5m² | Tường 343.0m²", "rc": 0.64, "sa": 784.0, "wa": 343.0, "ra": 220.5, "height": 3.5},
    {"id": 3, "name": "Khối 3: RC=0.66 | 1 tầng (3.5m) | Mái 220.5m² | Tường 318.5m²", "rc": 0.66, "sa": 759.5, "wa": 318.5, "ra": 220.5, "height": 3.5},
    {"id": 4, "name": "Khối 4: RC=0.69 | 1 tầng (3.5m) | Mái 220.5m² | Tường 294.0m²", "rc": 0.69, "sa": 735.0, "wa": 294.0, "ra": 220.5, "height": 3.5},
    {"id": 5, "name": "Khối 5: RC=0.71 | 1 tầng (3.5m) | Mái 220.5m² | Tường 269.5m²", "rc": 0.71, "sa": 710.5, "wa": 269.5, "ra": 220.5, "height": 3.5},
    {"id": 6, "name": "Khối 6: RC=0.74 | 1 tầng (3.5m) | Mái 220.5m² | Tường 245.0m²", "rc": 0.74, "sa": 686.0, "wa": 245.0, "ra": 220.5, "height": 3.5},
    {"id": 7, "name": "Khối 7: RC=0.76 | 2 tầng (7.0m) | Mái 122.5m² | Tường 416.5m²", "rc": 0.76, "sa": 661.5, "wa": 416.5, "ra": 122.5, "height": 7.0},
    {"id": 8, "name": "Khối 8: RC=0.79 | 2 tầng (7.0m) | Mái 147.0m² | Tường 343.0m²", "rc": 0.79, "sa": 637.0, "wa": 343.0, "ra": 147.0, "height": 7.0},
    {"id": 9, "name": "Khối 9: RC=0.82 | 2 tầng (7.0m) | Mái 147.0m² | Tường 318.5m²", "rc": 0.82, "sa": 612.5, "wa": 318.5, "ra": 147.0, "height": 7.0},
    {"id": 10, "name": "Khối 10: RC=0.86 | 2 tầng (7.0m) | Mái 147.0m² | Tường 294.0m²", "rc": 0.86, "sa": 588.0, "wa": 294.0, "ra": 147.0, "height": 7.0},
    {"id": 11, "name": "Khối 11: RC=0.90 | 2 tầng (7.0m) | Mái 122.5m² | Tường 318.5m²", "rc": 0.90, "sa": 563.5, "wa": 318.5, "ra": 122.5, "height": 7.0},
    {"id": 12, "name": "Khối 12: RC=0.98 | 2 tầng (7.0m) | Mái 110.25m² | Tường 294.0m²", "rc": 0.98, "sa": 514.5, "wa": 294.0, "ra": 110.25, "height": 7.0}
]

shape_mode = st.sidebar.radio(
    "Chế độ thiết lập hình khối:",
    ["12 Khối hình học chuẩn ENB2012 (Khuyên dùng)", "Tùy chỉnh tự do (Custom)"],
    index=0,
    help="Khuyên dùng 12 khối chuẩn để đảm bảo công trình tuân thủ ràng buộc vật lý và không bị Out-of-Distribution."
)

def sync_slider_number(label, min_val, max_val, default_val, step, key, help_text=None, format="%.2f", decimals=2):
    k_slider = f"{key}_slider"
    k_num = f"{key}_num"
    
    if k_slider not in st.session_state:
        st.session_state[k_slider] = float(default_val)
    if k_num not in st.session_state:
        st.session_state[k_num] = float(default_val)
        
    def _from_slider():
        st.session_state[k_num] = round(st.session_state[k_slider], decimals)
        
    def _from_num():
        st.session_state[k_slider] = round(st.session_state[k_num], decimals)
        
    st.sidebar.markdown(f"**{label}**", help=help_text)
    col_slide, col_num = st.sidebar.columns([1.6, 1.4])
    with col_slide:
        st.slider(
            label=label,
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=k_slider,
            on_change=_from_slider,
            label_visibility="collapsed"
        )
    with col_num:
        st.number_input(
            label=label,
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            format=format,
            key=k_num,
            on_change=_from_num,
            label_visibility="collapsed"
        )
    return st.session_state[k_num]

if shape_mode == "12 Khối hình học chuẩn ENB2012 (Khuyên dùng)":
    sel_idx = st.sidebar.selectbox(
        "Chọn khối hình học tiêu chuẩn:",
        options=list(range(len(STANDARD_SHAPES))),
        format_func=lambda i: STANDARD_SHAPES[i]["name"],
        index=6
    )
    selected = STANDARD_SHAPES[sel_idx]
    rc = selected["rc"]
    sa = selected["sa"]
    wa = selected["wa"]
    ra = selected["ra"]
    height = selected["height"]
    st.sidebar.success(f"Khối đã chọn: RC = {rc} | Bề mặt = {sa} m² | Tường = {wa} m² | Mái = {ra} m² | Chiều cao = {height} m")
else:
    st.sidebar.markdown("Điều chỉnh hoặc gõ trực tiếp thông số bao che:")
    rc = sync_slider_number("1. Độ nén tương đối (Relative Compactness)", min_val=0.62, max_val=0.98, default_val=0.76, step=0.02, key="rc", help_text="Tỷ số giữa thể tích và diện tích bề mặt bao che (V/A)", format="%.2f", decimals=2)
    sa = sync_slider_number("2. Diện tích bề mặt bao che (Surface Area - m²)", min_val=514.5, max_val=808.5, default_val=661.5, step=12.25, key="sa", format="%.2f", decimals=2)
    wa = sync_slider_number("3. Diện tích tường (Wall Area - m²)", min_val=245.0, max_val=416.5, default_val=318.5, step=12.25, key="wa", format="%.2f", decimals=2)
    ra = sync_slider_number("4. Diện tích mái (Roof Area - m²)", min_val=110.25, max_val=220.5, default_val=171.5, step=12.25, key="ra", format="%.2f", decimals=2)
    height = st.sidebar.selectbox("5. Chiều cao tổng thể (Overall Height - m)", options=[3.5, 7.0], index=1, help="3.5m tương đương nhà 1 tầng; 7.0m tương đương 2 tầng")
    
    # Kiểm tra OOD
    is_matched = any(
        abs(rc - s['rc']) < 0.01 and abs(sa - s['sa']) < 1.0 and abs(wa - s['wa']) < 1.0 and abs(ra - s['ra']) < 1.0 and abs(height - s['height']) < 0.1
        for s in STANDARD_SHAPES
    )
    if not is_matched:
        st.sidebar.warning("**Cảnh báo Ngoài phân phối (OOD):** Cấu hình hình học này không thuộc 12 khối hình học chuẩn ENB2012. Sai số dự báo có thể tăng cao do vi phạm giả định phân phối dữ liệu!")

orient = st.sidebar.selectbox("6. Hướng nhà đón nắng (Orientation)", options=[2, 3, 4, 5], format_func=lambda x: {2: "Bắc (2)", 3: "Đông (3)", 4: "Nam (4)", 5: "Tây (5)"}[x], index=1)

# Glazing area: allow both slider & direct number (% or decimal)
def sync_glazing():
    if "glazing_pct_slider" not in st.session_state:
        st.session_state["glazing_pct_slider"] = 25
    if "glazing_pct_num" not in st.session_state:
        st.session_state["glazing_pct_num"] = 25
        
    def _from_slider():
        st.session_state["glazing_pct_num"] = int(st.session_state["glazing_pct_slider"])
    def _from_num():
        st.session_state["glazing_pct_slider"] = int(st.session_state["glazing_pct_num"])
        
    st.sidebar.markdown("**7. Tỷ lệ diện tích cửa kính (Glazing Area - %)**", help="Tỷ lệ diện tích kính trên bề mặt tường (0%, 10%, 25%, 40%)")
    col_slide, col_num = st.sidebar.columns([1.6, 1.4])
    with col_slide:
        st.select_slider(
            label="Glazing slider",
            options=[0, 10, 25, 40],
            key="glazing_pct_slider",
            on_change=_from_slider,
            label_visibility="collapsed"
        )
    with col_num:
        st.number_input(
            label="Glazing number",
            min_value=0,
            max_value=40,
            step=5,
            key="glazing_pct_num",
            on_change=_from_num,
            label_visibility="collapsed"
        )
    return float(st.session_state["glazing_pct_num"]) / 100.0

glazing = sync_glazing()
glazing_dist = st.sidebar.selectbox("8. Phân bố kính (Glazing Distribution)", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: {0: "Không có kính", 1: "Đều 4 hướng", 2: "Lệch Bắc", 3: "Lệch Đông", 4: "Lệch Nam", 5: "Lệch Tây"}[x], index=1)

st.sidebar.markdown("---")
# Diện tích sàn tính chuẩn theo hình học (Mái x Số tầng)
auto_floor = float(ra * (height / 3.5))
floor_area = st.sidebar.number_input(
    "Diện tích sàn sử dụng (m²)",
    min_value=50.0,
    max_value=2000.0,
    value=auto_floor,
    step=25.0,
    help="Tự động tính từ Diện tích mái x Số tầng. Giới hạn công trình dân dụng thấp tầng 1–2 tầng."
)

# Automatic Feature Engineering
w_to_r = wa / ra
glazing_real = wa * glazing
volume = ra * height
s_to_v = sa / volume

# Create feature vector
input_features = pd.DataFrame([{
    'Relative_Compactness': rc,
    'Surface_Area': sa,
    'Wall_Area': wa,
    'Roof_Area': ra,
    'Overall_Height': height,
    'Orientation': orient,
    'Glazing_Area': glazing,
    'Glazing_Distribution': glazing_dist,
    'Wall_to_Roof_Ratio': w_to_r,
    'Glazing_Area_Real': glazing_real,
    'Surface_to_Volume_Ratio': s_to_v
}])

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Dự Báo Phụ Tải & Khuyến Nghị", 
    "Mô Phỏng Kịch Bản 'What-If'", 
    "Minh Bạch Hóa AI (XAI)", 
    "Thông Tin Đề Tài & Quy Trình"
])

# ----------------- TAB 1 -----------------
with tab1:
    st.subheader("1. Kết Quả Dự Báo Phụ Tải & Khuyến Nghị Kích Cỡ Thiết Bị HVAC")
    
    if model_hl is None or model_cl is None:
        st.error("Không tìm thấy tệp mô hình đã lưu (`best_model_hl.joblib` / `best_model_cl.joblib`). Vui lòng chạy lại `code/project_ai_notebook.ipynb`!")
    else:
        # Predict
        pred_hl = float(model_hl.predict(input_features)[0])
        pred_cl = float(model_cl.predict(input_features)[0])
        p90_hl = float(model_p90_hl.predict(input_features)[0]) if model_p90_hl is not None else pred_hl * 1.05
        p90_cl = float(model_p90_cl.predict(input_features)[0]) if model_p90_cl is not None else pred_cl * 1.05
        
        # ASHRAE / HVAC Engineering Calculations:
        # ENB2012: kWh/m2 là chỉ số nhu cầu năng lượng hàng năm (Annual Thermal Energy Demand)
        COP_DEFAULT = 3.5 # Hệ số hiệu quả năng lượng chiller/VRV tiêu chuẩn
        annual_cooling_kwh = pred_cl * floor_area # Năng lượng lạnh yêu cầu mỗi năm (kWh/năm)
        annual_elec_kwh = annual_cooling_kwh / COP_DEFAULT # Điện năng tiêu thụ làm lạnh (kWh/năm)
        
        # Ước lượng công suất đỉnh (Peak Cooling Load) dựa trên giờ vận hành đầy tải tương đương (EFLH ~ 2000 giờ/năm)
        peak_cooling_kw = annual_cooling_kwh / 2000.0 # kW công suất lạnh
        total_cl_rt = peak_cooling_kw / 3.517          # Ton of Refrigeration (1 RT = 3.517 kW lạnh)
        total_cl_hp = peak_cooling_kw / 2.64           # Mã lực lạnh (1 HP lạnh ~ 2.64 kW lạnh)
        
        # Khuyến nghị với hệ số an toàn thiết kế khoa học 10% (dựa trên phân vị thiếu tải P95 = 4.11%)
        SAFETY_FACTOR = 1.10
        rec_chiller_rt = total_cl_rt * SAFETY_FACTOR
        rec_type = "Hệ thống điều hòa Multi-split / VRV gia dụng" if rec_chiller_rt < 5.0 else "Hệ thống điều hòa trung tâm VRV/VRF (hoặc Chiller công suất nhỏ)"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #EF4444;">
                <p class="card-title">Phụ Tải Sưởi Ấm Dự Báo (Heating Load - Y1)</p>
                <p class="card-value" style="color: #991B1B;">{pred_hl:.2f} <span style="font-size: 1.1rem; font-weight: 500;">kWh/m²</span></p>
                <p class="card-subtitle">Năng lượng sưởi hàng năm: {pred_hl * floor_area:,.0f} kWh/năm (Cận trên P90: {p90_hl:.2f} kWh/m²)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3B82F6;">
                <p class="card-title">Phụ Tải Làm Lạnh Dự Báo (Cooling Load - Y2)</p>
                <p class="card-value" style="color: #1E3A8A;">{pred_cl:.2f} <span style="font-size: 1.1rem; font-weight: 500;">kWh/m²</span></p>
                <p class="card-subtitle">Nhu cầu năng lượng lạnh: {annual_cooling_kwh:,.0f} kWh/năm (Điện tiêu thụ: {annual_elec_kwh:,.0f} kWh/năm | Cận trên P90: {p90_cl:.2f} kWh/m²)</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #10B981; background-color: #F0FDF4;">
            <p class="card-title" style="color: #065F46;">Khuyến Nghị Thiết Kế Hệ Thống Làm Lạnh (HVAC Sizing Recommendation)</p>
            <p class="card-value" style="color: #166534;">{rec_chiller_rt:.1f} <span style="font-size: 1.1rem; font-weight: 500;">RT (Tấn lạnh)</span></p>
            <p class="card-subtitle">Đã tính hệ số an toàn 10% ({peak_cooling_kw*1.1:.1f} kW lạnh / {total_cl_hp*1.1:.1f} HP lạnh)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Phân Tích & Hướng Dẫn Vận Hành Cho Kỹ Sư:")
        st.info(f"""
        * **Diện tích sàn tính toán:** **{floor_area:.1f} m²** (Công trình {1 if height < 5 else 2} tầng).
        * **Kích cỡ thiết bị đề xuất:** Nên lựa chọn **{rec_type}** có công suất lạnh danh định **{rec_chiller_rt:.1f} RT** (tương đương **{peak_cooling_kw*1.1:.1f} kW lạnh** / **{total_cl_hp*1.1:.1f} HP lạnh**).
        * **Độ tin cậy thiết kế & Chống thiếu tải:** Hệ thống kết hợp đồng thời phân vị cận trên **P90** ({p90_cl:.2f} kWh/m²) cùng hệ số an toàn khoa học **10%** (xác lập dựa trên phân vị $P_{{95}} = 4.11\%$ sai số thiếu tải), giúp bảo vệ công trình không bị thiếu lạnh trong đợt nắng nóng đỉnh điểm mà vẫn tối ưu chi phí đầu tư ban đầu (CapEx).
        """)

# ----------------- TAB 2 -----------------
with tab2:
    st.subheader("2. Mô Phỏng Kịch Bản Tối Ưu Hóa Năng Lượng ('What-If' Analysis)")
    st.write("So sánh phương án thiết kế hiện tại với phương án cải tiến nhằm định lượng điện năng và chi phí tiền điện tiết kiệm:")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("**Phương án Cải tiến (Alternative Scenario):**")
        whatif_glazing = st.select_slider("Thay đổi tỷ lệ kính (Glazing Area)", options=[0.0, 0.10, 0.25, 0.40], value=0.10)
        whatif_orient = st.selectbox("Đổi hướng nhà (Orientation)", options=[2, 3, 4, 5], format_func=lambda x: {2: "Bắc (2)", 3: "Đông (3)", 4: "Nam (4)", 5: "Tây (5)"}[x], index=0)
    with col_w2:
        st.markdown("**Thông số kinh tế:**")
        elec_price = st.number_input("Đơn giá điện năng (VNĐ / kWh)", value=2500, step=100)
        chiller_cop = st.number_input("Hệ số hiệu quả COP hệ thống", value=3.5, step=0.1)
        
    alt_features = input_features.copy()
    alt_features['Glazing_Area'] = whatif_glazing
    alt_features['Orientation'] = whatif_orient
    alt_features['Glazing_Area_Real'] = wa * whatif_glazing
    
    if model_cl is not None:
        alt_cl = float(model_cl.predict(alt_features)[0])
        diff_cl = alt_cl - pred_cl
        pct_cl = (diff_cl / pred_cl) * 100
        
        st.markdown("---")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Phụ tải làm lạnh sau tối ưu", f"{alt_cl:.2f} kWh/m²", f"{diff_cl:.2f} kWh/m² ({pct_cl:.1f}%)", delta_color="inverse")
        
        # Tiết kiệm năng lượng lạnh (kWh/năm)
        annual_saved_cooling_kwh = max(0, -diff_cl) * floor_area
        annual_saved_kwh = annual_saved_cooling_kwh / chiller_cop
        annual_saved_vnd = annual_saved_kwh * elec_price
        
        m_col2.metric("Tiền điện tiết kiệm hàng năm", f"{annual_saved_vnd:,.0f} VNĐ/năm", f"{annual_saved_kwh:,.0f} kWh điện/năm")
        
        if diff_cl < 0:
            st.success(f"Phương án cải tiến giúp giảm **{-diff_cl:.2f} kWh/m²** phụ tải làm lạnh, tương đương tiết kiệm khoảng **{annual_saved_vnd:,.0f} VNĐ** tiền điện vận hành mỗi năm!")
        else:
            st.info("Phương án này không làm giảm phụ tải lạnh so với thiết kế hiện tại.")

# ----------------- TAB 3 -----------------
with tab3:
    st.subheader("3. Minh Bạch Hóa Mô Hình AI với SHAP (Explainable AI)")
    st.write("Giải thích cơ chế đóng góp của các đặc trưng vào quyết định của mô hình máy học:")
    
    shap_cl_img = get_asset_path("figures/fig8_shap_summary_cl.png")
    fi_img = get_asset_path("figures/fig7_feature_importance.png")
    
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.markdown("**Biểu đồ SHAP Beeswarm Summary (Cooling Load):**")
        if os.path.exists(shap_cl_img):
            st.image(shap_cl_img, use_container_width=True)
        else:
            st.warning("Chưa tìm thấy `figures/fig8_shap_summary_cl.png`.")
    with col_x2:
        st.markdown("**Độ quan trọng đặc trưng (Feature Importance):**")
        if os.path.exists(fi_img):
            st.image(fi_img, use_container_width=True)
        else:
            st.warning("Chưa tìm thấy `figures/fig7_feature_importance.png`.")
            
    st.info("""
    * **Độ nén tương đối (Relative Compactness):** Là yếu tố quyết định hàng đầu trong việc phân loại quy mô và cấu trúc hình học công trình.
    * **Diện tích kính thực tế (Glazing Area Real):** Đóng góp trực tiếp làm gia tăng phụ tải làm lạnh do bức xạ nhiệt mặt trời xuyên qua bao che kính.
    """)

# ----------------- TAB 4 -----------------
with tab4:
    st.subheader("4. Thông Tin Đề Tài & Quy Trình Khai Thác Chuẩn CRISP-DM")
    st.markdown("""
    * **Học phần:** Trí tuệ Nhân tạo Ứng dụng & Khoa học Dữ liệu (Học kỳ 1 - Năm 2026)
    * **Đề tài:** Ứng dụng học máy dự báo phụ tải nhiệt công trình và tối ưu hóa năng lượng HVAC
    * **Nhóm sinh viên thực hiện:**
      1. **Nguyễn Huỳnh Anh Dũng** — MSSV: 2042250036 (Trường Đại học Công Thương TP.HCM, Khoa Công nghệ Cơ khí)
      2. **Đặng Yến Thư** — MSSV: 2421001883 (Trường Đại học Tài chính - Marketing, Khoa Công nghệ Tài chính Ngân hàng)
    * **Quy trình nghiên cứu CRISP-DM:**
      1. *Business Understanding:* Xác định nỗi đau kinh doanh chọn thừa/thiếu tải HVAC, xác lập mục tiêu SMART & ROI.
      2. *Data Understanding & EDA:* Khám phá phân phối 2 đỉnh, ma trận tương quan, kiểm định ngoại lai IQR.
      3. *Data Preparation:* Kỹ thuật tạo đặc trưng (Tường/Mái, Kính thực tế, Bao che/Thể tích).
      4. *Modeling:* Baseline Ridge + 3 mô hình Ensemble (Random Forest, LightGBM, XGBoost) tối ưu siêu tham số với Optuna (5-Fold CV) và kiểm định ngoại suy Leave-One-Shape-Out (LOSO).
      5. *Evaluation & XAI:* Đánh giá đa chiều (R², RMSE, MAE, MAPE), giải thích vật lý với SHAP Summary & Waterfall.
      6. *Deployment:* Đóng gói sản phẩm mẫu tương tác Streamlit Web App có cơ chế chặn Out-of-Distribution và giám sát Drift.
    """)
