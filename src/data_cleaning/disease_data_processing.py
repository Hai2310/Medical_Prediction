import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nltk.download("stopwords")
nltk.download("wordnet")
# 1. XỬ LÝ TEXT DATA
def load_and_clean_text(csv_path):
    df = pd.read_csv(csv_path)

    # Nếu cột mô tả khác tên thì đổi lại 
    text_cols = [col for col in df.columns if col not in ["Outcome Variable"]]

    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        tokens = [w for w in tokens if w not in stopwords.words("english")]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    # Làm sạch tất cả cột triệu chứng
    for col in text_cols:
        df[col] = df[col].apply(lambda x: clean_text(x))

    return df

# 2. XỬ LÝ IMAGE DATA

def load_and_preprocess_images(image_dir, img_size=(128, 128), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    generator = datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )
    return generator


    # TEXT 
text_path = r"C:\Users\Lenovo\Desktop\Project ML\Medical Prediction - Copy\data\notes\Disease_symptom_and_patient_profile_dataset.csv" 
df_text = load_and_clean_text(text_path)
print("Text data sample:")
print(df_text.head())

    # IMAGE 
image_path = r"C:\Users\Lenovo\Desktop\Project ML\Medical Prediction - Copy\data\images\train"
train_gen = load_and_preprocess_images(image_path)
imgs, labels = next(train_gen)
print("Image batch shape:", imgs.shape)
print("Label batch shape:", labels.shape)