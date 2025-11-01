import joblib
from preprocess_features import y_tf_idf_create, X_tf_idf_create
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

# Load tập dữ liệu
y_test = y_tf_idf_create("./data/test.csv")
X_test = X_tf_idf_create("./data/test.csv")

#Load model
best_tf_idf_model = joblib.load("best_tf_idf_model.pkl")
y_pred = best_tf_idf_model.predict(X_test)

#Tính Accuracy, F1-score
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")


print("Final:")
print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test F1-macro: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
