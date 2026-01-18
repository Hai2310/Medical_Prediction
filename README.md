# 🏥 MEDICAL_PREDICTION

Hệ thống **Medical Prediction** là một project Machine Learning tổng hợp, kết hợp **NLP (xử lý văn bản y tế)**, **Computer Vision (ảnh X-ray)** và **Data Mining (khai phá luật y tế)** nhằm hỗ trợ **dự đoán và phân tích bệnh** từ nhiều nguồn dữ liệu khác nhau.

---

## 🎯 Mục tiêu dự án

- Xây dựng pipeline phân tích dữ liệu y tế hoàn chỉnh bằng Machine Learning.
- Kết hợp **NLP + Computer Vision + Data Mining** để:
  - 📄 Dự đoán bệnh từ **bệnh án / mô tả triệu chứng dạng text**.
  - 🩻 Phân loại bệnh từ **ảnh X-ray**.
  - 📊 Khai phá **luật kết hợp triệu chứng – bệnh** để hỗ trợ giải thích.
- Triển khai **demo Web/App** phục vụ chẩn đoán nhanh.

---

## 🏥 Đối tượng & phạm vi ứng dụng

- Bệnh viện, phòng khám
- Hệ thống hỗ trợ bác sĩ
- Nghiên cứu và học tập về AI trong y tế

Hệ thống có thể:
- Gợi ý chẩn đoán nhanh
- Phân tích hình ảnh X-ray
- Cung cấp luật kết hợp giúp hiểu rõ đặc điểm bệnh

---

## 📦 Dữ liệu sử dụng

- **Text dataset**: bệnh án, mô tả triệu chứng (tiếng Việt / tiếng Anh) :
- **Image dataset**: ảnh X-ray đã phân lớp (COVID / NORMAL / PNEUMONIA…)
- **Structured dataset**: bảng triệu chứng – bệnh phục vụ Data Mining

---

## 🧰 Công nghệ & thư viện

- **Ngôn ngữ**: Python
- **Xử lý dữ liệu**: pandas, numpy
- **Machine Learning**: scikit-learn
- **NLP**: TF-IDF, BERT
- **Computer Vision**: PyTorch / TensorFlow / Keras
- **Data Mining**: Apriori, FP-Growth
- **Visualization**: matplotlib, seaborn
- **Demo Web**: Streamlit / Flask

---

## 📁 Cấu trúc thư mục

```
MEDICAL_PREDICTION/
│
├── data/                     # Dataset (text, image, structured)
│
├── src/
│   ├── data_cleaning/        # Làm sạch & tiền xử lý dữ liệu
│   │   ├── clean_metadata.py
│   │   ├── disease_data_processing.py
│   │   └── metadata_processing.py
│   │
│   ├── NLP/                  # Xử lý văn bản y tế
│   │   ├── analyze_text_tfidf.py
│   │   ├── analyze_text_bert.py
│   │   └── evaluate_text_models.py
│   │
│   ├── CV/                   # Phân tích ảnh X-ray
│   │   ├── analyze_image.ipynb
│   │   ├── learning_curve.ipynb
│   │   └── benchmark.ipynb
│   │
│   └── DM/                   # Data Mining
│       ├── analyze_rules.ipynb
│       ├── visualize.ipynb
│       └── BTL_IntroductionToML.ipynb
│
├── models/                   # Model đã huấn luyện
│   ├── BERT/
│   ├── TF-IDF/
│   ├── CV/
│   └── DM/
│
├── outputs/                  # Kết quả, hình ảnh, metric
│
├── REPORT/
│   └── report.py             # Sinh báo cáo tổng hợp
│
├── WEB/                      # Demo Web/App
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Pipeline triển khai

### Bước 1. Thu thập & chuẩn bị dữ liệu (1 tuần)

- Thu thập dataset text, ảnh X-ray và dữ liệu bảng.
- Làm sạch dữ liệu:
  - **Text**: loại ký tự nhiễu, stopwords, lemmatization/stemming.
  - **Image**: resize 224×224, normalization, augmentation, cân bằng dữ liệu.

---

### Bước 2. Phân tích văn bản y tế – NLP (2 tuần)

- Vector hóa văn bản:
  - TF-IDF (baseline)
  - BERT embeddings (advanced)
- Mô hình huấn luyện:
  - Logistic Regression
  - Random Forest
  - (Optional) SVM
- Đánh giá:
  - Accuracy
  - F1-score
  - Confusion Matrix

---

### Bước 3. Khai phá dữ liệu y tế – Data Mining (1 tuần)

- Biến dữ liệu triệu chứng – bệnh thành dạng giao dịch.
- Thuật toán:
  - Apriori
  - FP-Growth
- Đánh giá luật:
  - Support
  - Confidence
  - Lift

**Ví dụ luật**:
```
{Ho, Khó thở} → {Viêm phổi}
Support: 0.12 | Confidence: 0.81 | Lift: 2.5
```

---

### Bước 4. Phân tích ảnh y tế – Computer Vision (2 tuần)

- Mô hình:
  - CNN cơ bản
  - Transfer Learning (ResNet, EfficientNet)
- Quy trình:
  - Load & preprocess ảnh
  - Huấn luyện + fine-tuning
  - Early stopping
- Metric:
  - Accuracy
  - AUC
  - ROC curve
  - Confusion matrix

---

### Bước 5. Tích hợp & đánh giá tổng thể (1 tuần)

- Tổng hợp kết quả từ:
  - NLP (text)
  - CV (ảnh)
  - Data Mining (luật)
- Visualization:
  - Confusion matrix
  - ROC curve
  - Biểu đồ luật kết hợp

---

### Bước 6. Demo hệ thống (Optional)

- Xây dựng Web/App bằng **Streamlit**:
  - Upload bệnh án (text) → dự đoán bệnh
  - Upload ảnh X-ray → phân loại
  - Hiển thị luật kết hợp

---

## 📊 Kết quả đạt được

- NLP: mô hình BERT cho kết quả tốt hơn TF-IDF baseline.
- CV: Transfer Learning đạt accuracy và AUC cao.
- Data Mining: trích xuất được nhiều luật y tế có ý nghĩa.
- Demo Web hoạt động ổn định.

---

## ⚠️ Khó khăn

- Dataset nhỏ và không đồng nhất
- Văn bản y tế nhiều thuật ngữ chuyên ngành
- Mất cân bằng lớp
- Ảnh X-ray nhiễu, độ phân giải thấp

---

## 🔮 Hướng phát triển

- Sử dụng **BioBERT / ClinicalBERT**
- Áp dụng **Vision Transformer (ViT)**
- Mô hình **đa-modal (text + image)**
- Bổ sung dữ liệu xét nghiệm
- Tối ưu hiệu năng Web/App

---

## 👨‍💻 Tác giả

**Nhóm 6**
**Hoàng Minh Hải**
**Trần Phúc Long**
**Nguyễn Quang Huy**
**Mai Tiến Hoàng**
**Nguyễn Khánh Lâm**
- Project học tập & nghiên cứu Machine Learning trong y tế

---

## 📌 Ghi chú

>  Hệ thống chỉ mang tính **hỗ trợ nghiên cứu – học tập**, **không thay thế chẩn đoán y khoa chuyên nghiệp**.

---



