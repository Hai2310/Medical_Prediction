import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

def X_tf_idf_create(dataset : str):
       
    df = pd.read_csv(dataset)
    # Xác định các loại cột
    numeric_cols = ["age", "temperature", "pO2 saturation"]
    categorical_cols = ["sex"]
    symptom_cols = ["cough", "fever", "healthy", "fatigue", "shortness_of_breath", "chest_pain"]
    target_col = "finding"

    # Tạo cột text từ các triệu chứng 
    def make_symptom_text(row):
        tokens = [col for col in symptom_cols if row[col] == 1]
        return " ".join(tokens) if tokens else "no_symptoms"

    df["symptoms_text"] = df.apply(make_symptom_text, axis=1)

    # Chuẩn hóa số (StandardScaler) 
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_cols])

    # Mã hóa giới tính (OneHotEncoder)
    ohe = OneHotEncoder(sparse_output=True, drop=None)
    X_cat = ohe.fit_transform(df[categorical_cols])

    # TF-IDF trên cột text (triệu chứng) 
    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(df["symptoms_text"])

    # Ghép tất cả thành ma trận đặc trưng 
    X_num_sparse = csr_matrix(X_num)
    X_test = hstack([X_cat, X_num_sparse, X_text], format="csr")
    return X_test

# Chuyển cột label thành vector
def y_tf_idf_create(dataset: str):
    df = pd.read_csv(dataset)   
    y = df["finding"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded