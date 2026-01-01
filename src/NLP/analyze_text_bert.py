import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_NAME = "bert-base-uncased"  
BATCH_SIZE = 16
MAX_LEN = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42

csv_path = "../../data/metadata.csv"
print(f" Đang load dataset: {csv_path}")
df = pd.read_csv(csv_path)


texts = df["notes"].astype(str).tolist()
labels = df["finding"].astype("category").cat.codes
label_names = list(df["finding"].astype("category").cat.categories)

print(f" Số mẫu: {len(texts)} | Số nhãn: {len(label_names)} | Nhãn: {label_names}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f" Dùng thiết bị: {device}")

def plot_learning_curve_fixed_test(model, X_train, y_train, X_test, y_test, title):
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    val_scores = []

    for frac in train_sizes:
        size = int(len(X_train) * frac)

        X_sub = X_train[:size]
        y_sub = y_train[:size]

        model.fit(X_sub, y_sub)

        y_train_pred = model.predict(X_sub)
        y_test_pred = model.predict(X_test)

        train_scores.append(
            f1_score(y_sub, y_train_pred, average="weighted")
        )
        val_scores.append(
            f1_score(y_test, y_test_pred, average="weighted")
        )

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes * len(X_train), train_scores, 'o-', label="Training score")
    plt.plot(train_sizes * len(X_train), val_scores, 'o-', label="Validation score")

    plt.title(title)
    plt.xlabel("Number of training samples")
    plt.ylabel("F1-score")
    plt.ylim(0.85, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()



def get_bert_embeddings(texts, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    """Tạo CLS embedding cho danh sách văn bản."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔹Đang tạo embedding"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            cls_embeds = outputs.last_hidden_state[:, 0, :]  # vector CLS
            embeddings.append(cls_embeds.cpu().numpy())

    return np.vstack(embeddings)

print(" Đang tính toán BERT embeddings...")
X = get_bert_embeddings(texts)
y = labels.values
print(f" Embeddings shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

results = {}
trained_models = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, y_pred_lr),
    "f1": f1_score(y_test, y_pred_lr, average="weighted"),
    "report": classification_report(y_test, y_pred_lr, target_names=label_names, digits=4)
}
trained_models["Logistic Regression"] = lr

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, y_pred_rf),
    "f1": f1_score(y_test, y_pred_rf, average="weighted"),
    "report": classification_report(y_test, y_pred_rf, target_names=label_names, digits=4)
}
trained_models["Random Forest"] = rf

print("\n====  KẾT QUẢ (BERT + ML) ====\n")
for name, res in results.items():
    print(f"--- {name} ---")
    print("Accuracy:", res["accuracy"])
    print("F1-weighted:", res["f1"])
    print(res["report"])

# ========== Logistic Regression ==========
plot_learning_curve_fixed_test(
    LogisticRegression(max_iter=1000),
    X_train, y_train,
    X_test, y_test,
    "Learning Curve - BERT + Logistic Regression"
)



# ========== Random Forest ==========
plot_learning_curve_fixed_test(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    X_train, y_train,
    X_test, y_test,
    "Learning Curve - BERT + Random Forest"
)


save_dir = "../models"
os.makedirs(save_dir, exist_ok=True)

for name, model_obj in trained_models.items():
    file_path = os.path.join(save_dir, f"bert_{name.replace(' ', '_').lower()}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f" Saved model: {file_path}")

np.save(os.path.join(save_dir, "bert_embeddings.npy"), X)
print(" Saved embeddings.")

save_results_path = os.path.join(save_dir, "results_bert.json")
with open(save_results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f" Saved evaluation results to {save_results_path}")

for name, model_obj in trained_models.items():
    print(f"\n Model: {name}")
    print("Classes learned:", model_obj.classes_)
    print("Number of classes:", len(model_obj.classes_))
