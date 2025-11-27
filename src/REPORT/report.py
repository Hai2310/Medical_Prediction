import os
import sys
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

sns.set(style="whitegrid")

# ===========================
# Cấu hình đường dẫn
# ===========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(SRC_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
CV_TEST_DIR = os.path.join(DATA_DIR, "images", "test")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
CV_MODEL_PATH = os.path.join(MODELS_DIR, "CV\\cv_model.keras")
NLP_MODEL_PATH = os.path.join(MODELS_DIR, "BERT\\bert_random_forest.pkl")
RULES_PATH = os.path.join(MODELS_DIR, "DM\\rules_apriori.pkl")
OUTPUT_DIR = os.path.join(SRC_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# Hàm hỗ trợ hiển thị & thống kê
# ===========================
def clf_report_to_df(y_true, y_pred, target_names=None):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return pd.DataFrame(report).transpose()

def plot_confusion_mat(y_true, y_pred, classes, normalize=False, title="Confusion Matrix", savepath=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def plot_roc(y_true, y_probs, classes, savepath=None):
    n_classes = len(classes)
    plt.figure(figsize=(8,6))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:,1])
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')
    else:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                plt.plot(fpr, tpr, lw=1.5, label=f'{classes[i]} (AUC={auc(fpr, tpr):.3f})')
            except ValueError:
                continue
        plt.plot([0,1],[0,1],'--',color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def plot_rules_scatter(rules, top_n=50, savepath=None):
    rules_plot = rules.sort_values(by="lift", ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(
        rules_plot['support'], rules_plot['confidence'],
        s=(rules_plot['lift'] - rules_plot['lift'].min() + 0.5) * 80,
        c=rules_plot['lift'], cmap="viridis", alpha=0.8, edgecolors="k"
    )
    plt.colorbar(scatter, label='Lift')
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title(f"Association Rules (Top {top_n})")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def plot_rules_network(rules, top_n=30, savepath=None):
    rules_plot = rules.sort_values(by="lift", ascending=False).head(top_n)
    G = nx.DiGraph()
    for _, row in rules_plot.iterrows():
        ants = sorted(list(row['antecedents'])) if hasattr(row['antecedents'], '__iter__') else [str(row['antecedents'])]
        cons = sorted(list(row['consequents'])) if hasattr(row['consequents'], '__iter__') else [str(row['consequents'])]
        for a in ants:
            G.add_node(a)
            for c in cons:
                G.add_edge(a, c, weight=row.get('lift', 1.0))
    plt.figure(figsize=(12,8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
    plt.title("Association Rule Network")
    plt.axis('off')
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

# ===========================
# Tải mô hình & dữ liệu
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Luật kết hợp ---
rules_df = None
if os.path.exists(RULES_PATH):
    try:
        rules_df = joblib.load(RULES_PATH)
    except Exception:
        with open(RULES_PATH, "rb") as f:
            rules_df = pickle.load(f)
    print("Association rules loaded.")

# --- Metadata ---
if not os.path.exists(METADATA_CSV):
    raise FileNotFoundError(f"Không tìm thấy metadata.csv tại {METADATA_CSV}")
metadata = pd.read_csv(METADATA_CSV, sep=",", encoding="utf-8", low_memory=False)
print("Metadata loaded:", metadata.shape)

# --- Mô hình NLP ---
nlp_model = None
if os.path.exists(NLP_MODEL_PATH):
    nlp_model = joblib.load(NLP_MODEL_PATH)
    print("NLP model loaded.")

# Thử tải module embedder BERT
bert_embedder = None
sys.path.insert(0, MODELS_DIR)
try:
    import bert_embeddings as be
    bert_embedder = getattr(be, "embed_texts", None) or getattr(be, "get_embeddings", None)
    if bert_embedder:
        print("Using embedding function from bert_embeddings.py.")
except Exception:
    print("No bert_embeddings.py found — using TF-IDF fallback.")

# --- Mô hình CV ---
cv_model = None
cv_label_names = None

if os.path.exists(CV_MODEL_PATH):
    try:
        cv_model = load_model(CV_MODEL_PATH)
        print("Loaded CV model (Keras).")
    except Exception as e:
        raise RuntimeError(f"Không thể load model Keras: {e}")
else:
    print("CV model not found.")


# ===========================
# TẠO DATASET TEST
# ===========================
cv_trues = cv_preds = cv_probs = None

if cv_model and os.path.isdir(CV_TEST_DIR):

    IMG_SIZE = (224, 224)   # với VGG16, MobileNet, ResNet của TF → 224x224 chuẩn nhất

    test_ds = image_dataset_from_directory(
        CV_TEST_DIR,
        labels="inferred",
        label_mode="categorical",   # lấy one-hot để khớp softmax
        batch_size=32,
        image_size=IMG_SIZE,
        shuffle=False
    )

    cv_label_names = test_ds.class_names
    print("Classes:", cv_label_names)

    # Chuẩn hóa y như MobileNet/VGG16: ảnh từ [0,255] → [0,1]
    def preprocess(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    test_ds = test_ds.map(preprocess)

    # ===========================
    # RUN INFERENCE
    # ===========================
    probs_list = []
    labels_list = []

    for batch_imgs, batch_labels in test_ds:
        p = cv_model.predict(batch_imgs, verbose=0)
        probs_list.append(p)
        labels_list.append(batch_labels.numpy())

    cv_probs = np.concatenate(probs_list, axis=0)
    cv_trues = np.argmax(np.concatenate(labels_list, axis=0), axis=1)
    cv_preds = np.argmax(cv_probs, axis=1)

    print("CV prediction completed:", len(cv_trues))

# ===========================
# Chạy dự đoán NLP
# ===========================
nlp_trues = nlp_preds = nlp_probs = classes_nlp = None
if nlp_model is not None and {'notes','finding'}.issubset(metadata.columns):
    df_nlp = metadata.dropna(subset=['notes']).copy()
    texts = df_nlp['notes'].astype(str).tolist()
    labels = df_nlp['finding'].astype(str).tolist()
    classes_nlp = sorted(df_nlp['finding'].unique())
    label_to_idx = {l:i for i,l in enumerate(classes_nlp)}
    y_true_nlp = np.array([label_to_idx[l] for l in labels])

    X_feats = None
    if bert_embedder:
        X_feats = np.array(bert_embedder(texts))
    else:
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            vec = joblib.load(tfidf_path)
            X_feats = vec.transform(texts)
    if X_feats is None:
        raise RuntimeError("Thiếu vectorizer hoặc hàm embed_texts().")

    try:
        y_probs_nlp = nlp_model.predict_proba(X_feats)
        nlp_trues, nlp_preds, nlp_probs = y_true_nlp, np.argmax(y_probs_nlp, axis=1), y_probs_nlp
    except Exception:
        nlp_trues, nlp_preds = y_true_nlp, nlp_model.predict(X_feats)
    print("NLP prediction completed:", len(nlp_trues))

# ===========================
# Xuất kết quả và biểu đồ
# ===========================
if cv_trues is not None:
    df_cv_report = clf_report_to_df(cv_trues, cv_preds, target_names=cv_label_names)
    df_cv_report.to_csv(os.path.join(OUTPUT_DIR, "cv_classification_report.csv"))
    plot_confusion_mat(cv_trues, cv_preds, cv_label_names, title="CV Confusion Matrix",
                       savepath=os.path.join(OUTPUT_DIR,"cv_confusion_matrix.png"))
    if cv_probs is not None:
        plot_roc(cv_trues, cv_probs, cv_label_names,
                 savepath=os.path.join(OUTPUT_DIR,"cv_roc.png"))

if nlp_trues is not None:
    df_nlp_report = clf_report_to_df(nlp_trues, nlp_preds, target_names=classes_nlp)
    df_nlp_report.to_csv(os.path.join(OUTPUT_DIR, "nlp_classification_report.csv"))
    plot_confusion_mat(nlp_trues, nlp_preds, classes_nlp, title="NLP Confusion Matrix",
                       savepath=os.path.join(OUTPUT_DIR,"nlp_confusion_matrix.png"))
    if nlp_probs is not None:
        plot_roc(nlp_trues, nlp_probs, classes_nlp,
                 savepath=os.path.join(OUTPUT_DIR,"nlp_roc.png"))

if rules_df is not None:
    rules_df.to_csv(os.path.join(OUTPUT_DIR, "association_rules.csv"), index=False)
    plot_rules_scatter(rules_df, savepath=os.path.join(OUTPUT_DIR,"rules_scatter.png"))
    plot_rules_network(rules_df, savepath=os.path.join(OUTPUT_DIR,"rules_network.png"))

print("Reports generated at:", OUTPUT_DIR)
