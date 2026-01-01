import matplotlib.pyplot as plt
import os
import json

#  Đọc kết quả đánh giá từ các file analyze_text.py và
# analyze_text_bert.py, sau đó tổng hợp - so sánh - trực quan hóa   

#  Đọc kết quả đánh giá từ các file analyze_text.py và
# analyze_text_bert.py, sau đó tổng hợp - so sánh - trực quan hóa   

def load_results(BASE_PATH):
    results = {}

    tfidf_path = os.path.join(BASE_PATH, "TF-IDF\\results_tfidf.json")
    bert_path = os.path.join(BASE_PATH, "BERT\\results_bert.json")

    
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "r", encoding="utf-8") as f:
            tfidf_results = json.load(f)
            for model_name, metrics in tfidf_results.items():
                results[f"TF-IDF + {model_name}"] = metrics
    else:
        print("Không tìm thấy results_tfidf.json")

   
    if os.path.exists(bert_path):
        with open(bert_path, "r", encoding="utf-8") as f:
            bert_results = json.load(f)
            for model_name, metrics in bert_results.items():
                results[f"BERT + {model_name}"] = metrics
    else:
        print("Không tìm thấy results_bert.json")

    return results







def plot_results(results, metric="accuracy"):
    models = list(results.keys())
    scores = [results[m][metric] for m in models]

    plt.figure(figsize=(10, 5))
    plt.barh(models, scores, color="lightseagreen")
    plt.xlabel(metric.capitalize())
    plt.title(f"So sánh mô hình theo {metric}")
    plt.xlim(0, 1)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()




from sklearn.model_selection import learning_curve
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(C=1, max_iter=1000),
    X, y,
    cv=5,
    scoring="neg_log_loss",
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_loss = -np.mean(train_scores, axis=1)
val_loss = -np.mean(val_scores, axis=1)

plt.plot(train_sizes, train_loss, label="Training Loss")
plt.plot(train_sizes, val_loss, label="Validation Loss")
plt.xlabel("Training Set Size")
plt.ylabel("Log Loss")
plt.title("Learning Curve - Logistic Regression")
plt.legend()
plt.show()
