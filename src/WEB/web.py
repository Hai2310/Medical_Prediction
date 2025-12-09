import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
import pandas as pd
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
import pickle
import random


st.sidebar.title("🩺 Medical Diagnosis App")
st.sidebar.markdown("Ứng dụng hỗ trợ dự đoán bệnh từ ảnh X-ray và dữ liệu lâm sàng.")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Điều hướng nhanh")
st.sidebar.markdown("""
- **📷 Dự đoán từ ảnh X-ray**
- **🧬 Dự đoán BERT từ mô tả triệu chứng**
- **📊 Phân tích & biểu đồ Data Mining**
""")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Cài đặt")
theme_choice = st.sidebar.selectbox("Giao diện", ["Light", "Dark", "Auto"])

st.sidebar.markdown("---")

st.sidebar.info(
    "**📌 Gợi ý:**\n"
    "• Upload ảnh và bấm **Predict** để xem kết quả\n"
    "• Sang tab Data Mining để xem phân tích dữ liệu\n"
)

# ============================
# PATH
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CV_MODEL_PATH = os.path.join(MODELS_DIR, "CV", "cv_model.keras")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "TF-IDF", "tfidf_vectorizer.pkl")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "TF-IDF", "tfidf_random_forest.pkl")
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "BERT", "bert_random_forest.pkl")
CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA"]

MAX_LEN = 64
MODEL_NAME = "bert-base-uncased"

metadata_path = os.path.join(os.path.dirname(BASE_DIR), "data", "metadata.csv")
df_meta = pd.read_csv(metadata_path)
LABEL_NAMES = list(df_meta["finding"].astype("category").cat.categories)


# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_cv_model():
    return load_model(CV_MODEL_PATH)

@st.cache_resource
def load_tfidf():
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    model = joblib.load(TFIDF_MODEL_PATH)
    return vectorizer, model

def preprocess_image(image_data):
    target_size = (224, 224)

    image = image_data.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)

    return img_array

@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME)
    bert.eval()
    return tokenizer, bert

@st.cache_resource
def load_rf():
    with open(BERT_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def get_embedding(text, tokenizer, bert):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        out = bert(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return cls.numpy()

def predict_text(text, rf_model, tokenizer, bert):
    emb = get_embedding(text, tokenizer, bert)  

    probs = rf_model.predict_proba(emb)[0]
    cls_ids = rf_model.classes_

    prob_dict = {LABEL_NAMES[c]: float(probs[i]) for i, c in enumerate(cls_ids)}

    final_probs = np.array([prob_dict.get(lbl, 0.0) for lbl in LABEL_NAMES])

    final_label = LABEL_NAMES[np.argmax(final_probs)]
    return final_label, final_probs

TREATMENTS = {
    "NORMAL": [
        "🩺 Tổng quan: Không cần thuốc đặc hiệu. Tập trung PHÒNG NGỪA và PHỤC HỒI thể lực.",
        "💊 Bổ sung (nếu cần): Vitamin D3 1000–2000 IU/ngày nếu thiếu; vitamin C 500 mg/ngày nếu ăn uống kém; kẽm 20–30 mg/ngày trong 7–14 ngày nếu vừa ốm.",
        "🧪 Kiểm tra cơ bản (1 lần/năm nếu cần): công thức máu, chức năng gan-thận, đường huyết, lipid; nếu có triệu chứng hô hấp bất thường thì X-quang phổi.",
        "🏃‍♂️ Chương trình vận động (4 tuần, dành cho người bình thường muốn tăng sức bền):\n"
        "  • Tuần 1–2: Đi bộ 20–30 phút/ngày (chia 2 lần), 5 ngày/tuần.\n"
        "  • Tuần 3–4: Tăng lên 30–45 phút/ngày hoặc thêm 2 buổi tập cường độ nhẹ (đi bộ nhanh, xe đạp đạp nhẹ).\n"
        "🫁 Bài tập thở cơ bản (hàng ngày):\n"
        "  • Thở bụng (diaphragmatic breathing): nằm ngửa, 10 phút, 3 lần/ngày, mỗi lần 10 nhịp.\n"
        "  • Pursed-lip breathing (thở mím môi): tập 5 phút, 3 lần/ngày, khi gắng sức.\n"
        "  • Nếu có dụng cụ: sử dụng incentive spirometer 10 lần/lần, 3 lần/ngày (hít sâu giữ 2–3 giây mỗi lần).",
        "🔔 Khi cần khám: Ho kéo dài >2 tuần, khó thở khi gắng sức, sốt kéo dài."
    ],

    "COVID": [
        "🔷 Mục tiêu: giảm triệu chứng, phòng biến chứng, đảm bảo oxy mô.",
        "💊 Thuốc hỗ trợ tại nhà (mức nhẹ):\n"
        "  • Paracetamol 500–1000 mg mỗi 4–6 giờ khi sốt hoặc đau (tối đa 3 g/ngày), tránh quá liều.\n"
        "  • Long đờm: Acetylcysteine 200 mg x 2–3 lần/ngày (uống) nếu có đờm đặc.\n"
        "  • Giảm ho: Dextromethorphan 10–20 mg x 3–4 lần/ngày (không dùng kéo dài nếu vẫn có đờm nhiều).\n"
        "  • Nếu chỉ định thuốc kháng virus ở bệnh nhân nguy cơ cao: Molnupiravir hoặc Paxlovid theo chỉ định (tham khảo bác sĩ), KHÔNG tự mua dùng bừa bãi.",
        "🫁 Hỗ trợ oxy & chăm sóc:\n"
        "  • Đo SpO₂ tại nhà: 4–6 giờ/lần; mục tiêu SpO₂ ≥ 94% (người có bệnh phổi mạn có mục tiêu khác theo BS).\n"
        "  • Bù dịch: uống đủ 1.5–2.5 L/ngày tùy tình trạng.\n"
        "  • Khi SpO₂ < 94% hoặc khó thở tăng → liên hệ BS/nhập viện.\n",
        "🧪 Xét nghiệm tham khảo (nếu triệu chứng nặng hoặc có yếu tố nguy cơ):\n"
        "  • Công thức máu, CRP, D-dimer, Ferritin, chức năng gan-thận;\n"
        "  • X-quang phổi (nghi viêm phổi), CT ngực nếu cần.\n",
        "🫁 Chương trình phục hồi hô hấp tại nhà (kèm theo minh họa):\n"
        "  • **Thở bụng (Diaphragmatic breathing)**: nằm ngửa, tay đặt lên bụng, hít sâu bằng mũi cho bụng phồng, thở ra mím môi; 10 lần/lần, 3 lần/ngày.\n"
        "  • **Thở mím môi (Pursed-lip breathing)**: hít 2 giây, mím môi thở ra chậm 4–6 giây; 5 phút, 3 lần/ngày; khi thấy hụt hơi.\n"
        "  • **Kỹ thuật ACBT (Active Cycle of Breathing Technique)** — dùng khi có đờm nhiều:\n"
        "      1. Thở thư giãn 3–4 nhịp.\n"
        "      2. Hít sâu (thoracic expansion) 3 lần, giữ 2–3 giây mỗi lần.\n"
        "      3. Thở ra mạnh để tống đờm (huff) 1–2 lần.\n" 
        "      => Lặp 3–4 chu kỳ; thực hiện 2–3 lần/ngày nếu có đờm.\n"
        "  • **Sử dụng incentive spirometer** (nếu có): 10 nhịp mỗi lần, 3 lần/ngày.\n"
        "  • **Đi bộ ngắn tăng dần**: bắt đầu 5–10 phút x 2 lần/ngày, tăng dần theo khả năng.\n",
        "🏥 Điều trị bệnh viện (trung bình → nặng):\n"
        "  • Oxy liệu pháp (gọng kính → mask → HFNC → thở máy tùy tình trạng).\n"
        "  • Corticosteroid (ví dụ Dexamethasone 6 mg/ngày × 10 ngày) khi bệnh nhân cần oxy (theo guideline và BS).\n"
        "  • Kháng đông dự phòng (Heparin trọng lượng thấp) cho BN nằm liệt/ít vận động hoặc tăng D-dimer.\n"
        "  • Theo dõi chặt: SpO₂, khí máu nếu cần, lọc máu các chỉ số (CRP, D-dimer, Ferritin).\n",
        "⏱ Lộ trình phục hồi 4 tuần (gợi ý):\n"
        "  • **Tuần 0–1 (giai đoạn cấp/ít vận động)**: Thở bụng 3×10 nhịp, sử dụng incentive 3 lần/ngày, nghỉ ngơi nhiều.\n"
        "  • **Tuần 2 (bắt đầu hồi phục)**: Đi bộ 10–15 phút x 2 lần/ngày; thở mím môi 3 lần/ngày; ACBT khi có đờm.\n"
        "  • **Tuần 3–4 (tăng sức bền)**: Đi bộ 20–30 phút/ngày hoặc tập aerobic nhẹ 3–5 lần/tuần; bổ sung bài tập tăng cường cơ hô hấp (IMT nếu có dụng cụ).\n",
        "🔍 Theo dõi & tái khám: Nếu sau 7–14 ngày chưa cải thiện hoặc có dấu hiệu nặng (khó thở, SpO₂ giảm) → nhập viện. Tái khám sau 4–6 tuần nếu ho kéo dài >4 tuần.",
        "⚠️ Cảnh báo: Không tự ý dùng kháng sinh trừ khi có chỉ định; không tự dùng steroid; thận trọng kháng virus (cần đơn BS)."
    ],

    "PNEUMONIA": [
        "🔷 Mục tiêu: eradication of pathogen, hỗ trợ hô hấp, ngăn biến chứng (áp xe, tràn mủ).",
        "💊 Kháng sinh (liều tham khảo người lớn, cần điều chỉnh theo BS và cân nặng):\n"
        "  • **Amoxicillin–Clavulanate (Augmentin)**: 1 g uống 2 lần/ngày (1 g every 12h) cho CAP mức nhẹ–vừa, thời gian 7–10 ngày.\n"
        "  • **Azithromycin**: 500 mg uống ngày 1, sau đó 250 mg/ngày x 4 ngày (dùng khi nghi vi khuẩn không điển hình).\n"
        "  • **Ceftriaxone**: 1–2 g IV mỗi 24h, dùng cho bệnh nhân nhập viện hoặc nặng.\n"
        "  • **Levofloxacin**: 500 mg uống/IV mỗi 24h, dùng khi nghi ngờ kháng thuốc hoặc bệnh nhân dị ứng beta-lactam.\n"
        "  • **Lưu ý**: Chọn kháng sinh theo kết quả cấy đàm/kháng sinh đồ nếu có; chỉnh liều khi suy thận/gan.\n",
        "🩺 Thuốc triệu chứng:\n"
        "  • Paracetamol 500–1000 mg khi sốt/đau (tối đa 3 g/ngày).\n"
        "  • Nếu đờm đặc: Acetylcysteine 200 mg x 2–3 lần/ngày.\n"
        "  • Bronchodilators (Salbutamol inhaler) nếu có co thắt phế quản: 100–200 mcg x 4–6 lần/ngày theo chỉ dẫn.\n",
        "🫁 Hỗ trợ oxy & chăm sóc:\n"
        "  • Oxy đường mũi nếu SpO₂ < 94% (mục tiêu thường 92–96% tùy BS và bệnh nền).\n"
        "  • Nếu giảm oxy nặng → HFNC hoặc thở máy xâm nhập theo chỉ định ICU.\n",
        "🧪 CLS cần làm sớm:\n"
        "  • X-quang phổi thẳng (PA) → xác định vùng viêm.\n"
        "  • Công thức máu, CRP, Procalcitonin (đánh giá nhiễm trùng).\n"
        "  • Cấy đàm (nếu có đờm) hoặc huyết thanh chẩn đoán vi khuẩn/virus.\n  • Khí máu động mạch nếu bệnh nhân khó thở nặng.\n",
        "🫁 Chương trình vật lý trị liệu phổi (thực hiện hàng ngày, cụ thể):\n"
        "  • **Postural drainage (Đặt dẫn lưu tư thế)**: trong 10–15 phút mỗi vị trí (dựa theo vùng tổn thương), 2–3 lần/ngày — thực hiện theo hướng dẫn PT hoặc BS.\n"
        "  • **Percussion & vibration** (đập lồng ngực nhẹ nhàng) kết hợp với drainage nếu có đờm đặc — do PT thực hiện hoặc hướng dẫn gia đình kỹ thuật an toàn.\n"
        "  • **ACBT (Active Cycle of Breathing Technique)**: 3–4 chu kỳ, 2 lần/ngày khi có đờm.\n"
        "  • **Incentive spirometry**: 10 nhịp/lần, 3 lần/ngày.\n"
        "  • **Diaphragmatic breathing**: 10–15 phút, 3 lần/ngày.\n"
        "  • **Progressive ambulation (tăng vận động dần)**: nếu khả năng cho phép, bắt đầu 5–10 phút đi bộ 2 lần/ngày, tăng dần 5 phút mỗi ngày.\n",
        "⏱ Kế hoạch phục hồi (mốc 6 tuần):\n"
        "  • **Tuần 0–1 (cấp)**: ưu tiên oxy, kháng sinh, physiotherapy ngắn; nghỉ ngơi nhiều.\n"
        "  • **Tuần 2–3**: bắt đầu chương trình đi bộ nhẹ, thở cơ hoành, tập tăng sức bền 10–20 phút/ngày.\n"
        "  • **Tuần 4–6**: tăng dần hoạt động lên 30–45 phút/ngày (đi bộ, đạp xe nhẹ), tập tăng cường cơ hô hấp nếu cần.\n",
        "🚨 Khi nhập viện ngay:\n"
        "  • SpO₂ < 92% (không đáp ứng với oxy đơn giản).\n"
        "  • Thở nhanh > 30/phút, huyết áp tụt, lú lẫn.\n  • Suy đa tạng hoặc cần hỗ trợ hô hấp xâm lấn.\n",
        "🔍 Theo dõi & tái khám:\n"
        "  • Tái khám sau 48–72 giờ nếu điều trị ngoại trú; nếu không cải thiện → nhập viện.\n"
        "  • Chụp X-quang lặp lại sau 4–6 tuần để xác nhận hồi phục phổi.\n",
        "⚠️ Lưu ý an toàn:\n"
        "  • Tránh dùng kháng sinh kéo dài không cần thiết.\n"
        "  • Kiểm tra dị ứng penicillin/cephalosporin trước khi dùng.\n"
        "  • Điều chỉnh liều thuốc khi suy thận/suy gan. Luôn tham vấn BS."
    ]
}



def get_treatment(disease):
    disease = disease.upper()
    if disease in TREATMENTS:
        return random.choice(TREATMENTS[disease])
    return "Không có khuyến cáo điều trị."

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Medical Prediction App", layout="wide")

st.title("🩺 Medical Prediction System (COVID - Pneumonia - Normal)")
st.write("Ứng dụng dự đoán bệnh dựa trên **CV model**, **NLP model**, và hiển thị **Data Mining insights**.")

tabs = st.tabs(["📷 CV Prediction (Image)", "✍️ NLP Prediction (Text)", "📊 Data Mining"])

# ==============================================================
#  TAB 1 — CV PREDICTION
# ==============================================================
with tabs[0]:
    st.header("📷 Dự đoán bệnh từ ảnh X-ray (Keras Model)")

    uploaded_file = st.file_uploader("Tải ảnh X-ray (.jpg, .png)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Ảnh đã tải", width=300)

        # Nút predict
        if st.button("🔮 Predict X-ray"):
            cv_model = load_cv_model()
            img_arr = preprocess_image(img)

            preds = cv_model.predict(img_arr)[0]
            pred_idx = np.argmax(preds)

            # ⭐ Lưu vào session_state
            st.session_state.cv_pred_idx = pred_idx
            st.session_state.cv_preds = preds

    # ⭐ KHỐI NÀY PHẢI ĐỂ BÊN NGOÀI nút predict
    if "cv_pred_idx" in st.session_state:
        pred_idx = st.session_state.cv_pred_idx
        preds = st.session_state.cv_preds

        st.subheader(f"🔍 **Kết quả: {CLASS_NAMES[pred_idx]}**")
        st.write(f"Độ tin cậy: `{preds[pred_idx]:.4f}`")

        st.write("### 📌 Xác suất từng lớp")
        for cls, p in zip(CLASS_NAMES, preds):
            st.write(f"- **{cls}**: `{p:.4f}`")

        # ⭐ Nút điều trị không làm mất kết quả nữa
        if st.button("💊 Gợi ý phương pháp điều trị"):
            treatment = get_treatment(CLASS_NAMES[pred_idx])
            st.info(f"**Phương pháp điều trị đề xuất:**\n\n{treatment}")



# ==============================================================
#  TAB 2 — NLP (BERT)
# ==============================================================
# ==== TAB 2: NLP BERT ====
with tabs[1]:
    st.header("✍️ NLP Prediction (BERT + Random Forest)")

    text = st.text_area("Nhập ghi chú bác sĩ / mô tả triệu chứng")

    tokenizer, bert = load_bert_model()
    rf_model = load_rf()

    if st.button("Predict Text"):
        if not text.strip():
            st.warning("Vui lòng nhập văn bản trước.")
        else:
            label, probs = predict_text(text, rf_model, tokenizer, bert)

            st.session_state.nlp_label = label
            st.session_state.nlp_probs = probs

    if "nlp_label" in st.session_state:
        label = st.session_state.nlp_label
        probs = st.session_state.nlp_probs

        st.subheader(f"Kết quả dự đoán: **{label}**")

        df = pd.DataFrame({"Class": LABEL_NAMES, "Probability": probs})
        st.bar_chart(df.set_index("Class"))

        if st.button("💊 Điều trị phù hợp"):
            treatment = get_treatment(label)
            st.info(f"**Phương pháp điều trị đề xuất:**\n\n{treatment}")

# ==============================================================
#  TAB 3 — Data Mining Visualizations
# ==============================================================

with tabs[2]:
    st.header("📊 Data Mining Insights")
    st.write("Các biểu đồ phân tích dữ liệu được nhóm theo từng chủ đề.")

    # =============================
    # 1️⃣ HISTOGRAMS (Age – Temp – SpO2)
    # =============================
    st.subheader("📌 1. Phân phối các chỉ số quan trọng (Histograms)")

    hist_images = [
        "hist_age.png",
        "hist_temperature.png",
        "hist_pO2_saturation.png"
    ]

    cols = st.columns(3)
    for i, img in enumerate(hist_images):
        path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(path):
            cols[i].image(path, caption=img.replace(".png", ""), use_container_width=True)

    st.markdown("---")

    # =============================
    # 2️⃣ COVID Distribution Charts
    # =============================
    st.subheader("📌 2. Phân bố bệnh nhân COVID")

    covid_imgs = [
        "covid_distribution_age.png",
        "covid_distribution_gender.png",
        "covid_scatter_temperature_vs_o2.png"
    ]

    cols = st.columns(2)
    for i, img in enumerate(covid_imgs):
        path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=img.replace('.png', ''), use_container_width=True)

    st.markdown("---")

    # =============================
    # 3️⃣ Pneumonia Distribution Charts
    # =============================
    st.subheader("📌 3. Phân bố bệnh nhân VIÊM PHỔI (Pneumonia)")

    pneu_imgs = [
        "pneu_distribution_gender.png",
        "pneu_scatter_temperature_vs_o2.png"
    ]

    cols = st.columns(2)
    for i, img in enumerate(pneu_imgs):
        path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=img.replace('.png', ''), use_container_width=True)

    st.markdown("---")

    # =============================
    # 4️⃣ Correlation & Boxplots
    # =============================
    st.subheader("📌 4. Phân tích tương quan & biến số")

    misc_imgs = [
        "correlation_map.png",
        "boxplot_variables.png"
    ]

    cols = st.columns(2)
    for i, img in enumerate(misc_imgs):
        path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=img.replace(".png", ""), use_container_width=True)

    st.markdown("---")

    # =============================
    # 5️⃣ CV Model Evaluation
    # =============================
    st.subheader("📌 5. Đánh giá mô hình CV")

    cv_imgs = [
        "cv_confusion_matrix.png",
        "cv_roc.png"
    ]

    cols = st.columns(2)
    for i, img in enumerate(cv_imgs):
        path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=img.replace(".png", ""), use_container_width=True)

    # CSV report nếu có
    report_path = os.path.join(OUTPUT_DIR, "cv_classification_report.csv")
    if os.path.exists(report_path):
        st.write("📄 **Báo cáo chi tiết (CSV):**")
        st.dataframe(pd.read_csv(report_path))

    st.markdown("---")

    # =============================
    # 6️⃣ Association Rules (Luật kết hợp)
    # =============================
    st.subheader("📌 6. Luật kết hợp (Association Rules) — Phân tích mẫu bệnh")

    rules_imgs = [
        "rules_network.png",
        "rules_scatter.png"
    ]

    cols = st.columns(2)
    for i, img in enumerate(rules_imgs):
        path = os.path.join(OUTPUT_DIR, img)
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=img.replace(".png", ""), use_container_width=True)

    # CSV rules
    rules_csv = os.path.join(OUTPUT_DIR, "association_rules.csv")
    if os.path.exists(rules_csv):
        st.write("📄 **Bảng luật (CSV):**")
        st.dataframe(pd.read_csv(rules_csv))
