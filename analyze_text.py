
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
import numpy as np
import joblib
from preprocess_features import y_tf_idf_create, X_tf_idf_create

# Load tập train
y_train = y_tf_idf_create("./data/train.csv")
X_train = X_tf_idf_create("./data/train.csv")

# Model Training
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, solver = "lbfgs"),
    'RandomForest(10)': RandomForestClassifier(max_depth= 10, n_estimators= 100, random_state=42),
    'RandomForest(20)': RandomForestClassifier(max_depth= 20, n_estimators= 100, random_state=42),
}
# In ra kết quả f1 của từng model
for name, model in models.items():
    f1 = make_scorer(f1_score, average='macro')
    score_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring=f1)
    score_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"{name:30s} | F1_macro: {np.mean(score_f1):.4f} ± {np.std(score_f1):.4f}"
          f" | Accuracy: {np.mean(score_acc):.4f} ± {np.std(score_acc):.4f}")

# Train lần cuối với đầy đủ tập train và lưu lại
best_model = RandomForestClassifier(max_depth=20, n_estimators=100, random_state=42)
best_model.fit(X_train, y_train) 
joblib.dump(best_model, "best_tf_idf_model.pkl")