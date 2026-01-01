import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, make_scorer
)
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack, csr_matrix
import joblib
import matplotlib.pyplot as plt

#  Xử lý dữ liệu & Tạo đặc trưng TF-IDF
def create_features_and_labels(dataset_path: str):
    df = pd.read_csv(dataset_path)
    
    numeric_cols = ["age", "temperature", "pO2 saturation"]
    categorical_cols = ["sex"]
    symptom_cols = ["cough", "fever", "healthy", "fatigue",
                    "shortness_of_breath", "chest_pain"]
    target_col = "finding"

    # Tạo text từ các triệu chứng
    def make_symptom_text(row):
        tokens = [col for col in symptom_cols if row[col] == 1]
        return " ".join(tokens) if tokens else "no_symptoms"

    df["symptoms_text"] = df.apply(make_symptom_text, axis=1)

    # Chuẩn hóa dữ liệu số
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_cols])

    # Mã hóa giới tính
    ohe = OneHotEncoder(sparse_output=True)
    X_cat = ohe.fit_transform(df[categorical_cols])

    # TF-IDF cho cột triệu chứng
    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(df["symptoms_text"])

    # Ghép các đặc trưng lại
    X = hstack([X_cat, csr_matrix(X_num), X_text], format="csr")

    # Encode nhãn
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    return X, y, le, scaler, ohe, vectorizer

def plot_logistic_loss_curve(X_train, y_train):
    # Khởi tạo model với warm_start để huấn luyện nối tiếp
    model = LogisticRegression(max_iter=1, solver='saga', warm_start=True, random_state=42, C=1)
    
    epochs = 50 # Số lần lặp
    losses = []
    
    # Giả lập quá trình huấn luyện qua từng epoch để lấy loss
    for epoch in range(epochs):
        model.fit(X_train, y_train)
        # Tính Cross-Entropy Loss (Log Loss)
        probs = model.predict_proba(X_train)
        from sklearn.metrics import log_loss
        loss = log_loss(y_train, probs)
        losses.append(loss)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), losses, color='blue', label='Log Loss')
    plt.xlabel('Epochs (Iterations)')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Logistic Regression Learning Curve (Loss over Epochs)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Huấn luyện + Đánh giá model
def train_model(dataset_path: str):
    X, y, le, scaler, ohe, vectorizer = create_features_and_labels(dataset_path)

    # Chia tập train/test 80–20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n📉 Plotting Logistic Loss Curve...")
    plot_logistic_loss_curve(X_train, y_train)
    # ================== GRID SEARCH ==================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_macro = make_scorer(f1_score, average="macro")

    param_grids = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "params": {
                "C": [0.01, 0.1, 1, 10],
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }
        }
    }

    best_models = {}
    results = {}

    print("🔹 GridSearchCV trên tập train (scoring = F1-macro):")

    for name, cfg in param_grids.items():
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring=f1_macro,
            cv=cv,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_

        print(f"\n{name}")
        print("Best F1-macro (CV):", grid.best_score_)
        print("Best params:", grid.best_params_)

        results[name + " (CV)"] = {
            "Best_F1_macro": float(grid.best_score_),
            "Best_Params": grid.best_params_
        }


    # ================== TEST EVALUATION ==================
    print("\n📊 Evaluation on test set:")

    best_model_name = None
    best_f1 = -1

    for name, model in best_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-weighted: {f1_weighted:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        results[name + " (Test)"] = {
            "Accuracy": float(acc),
            "F1_weighted": float(f1_weighted),
            "Classification_Report": report
        }        
        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_model_name = name

        model_filename = f"../models/TF-IDF/tfidf_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_filename)

    # ================== SAVE ENCODERS ==================
    joblib.dump(vectorizer, "../models/TF-IDF/tfidf_vectorizer.pkl")
    joblib.dump(scaler, "../models/TF-IDF/tfidf_scaler.pkl")
    joblib.dump(ohe, "../models/TF-IDF/tfidf_onehot_encoder.pkl")
    joblib.dump(le, "../models/TF-IDF/tfidf_label_encoder.pkl")

    with open("../models/TF-IDF/results_tfidf.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n🏆 BEST MODEL (based on F1-weighted on test):", best_model_name)
    print("\n Hoàn tất huấn luyện!")

    
if __name__ == "__main__": train_model("../../data/metadata.csv")