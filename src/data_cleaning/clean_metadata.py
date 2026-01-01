import pandas as pd
import re
import missingno as mso
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch



df = pd.read_csv("../../data/metadata.csv")
df.head()
mso.matrix(df)

df["notes"] = (
    df["clinical_notes"].fillna("") + " " +
    df["other_notes"].fillna("")
).str.strip()

# Giữ các cột cần thiết
keep_cols = [
    "notes",
    "age",
    "sex",
    "temperature",
    "pO2_saturation",
    "finding"
]

# Drop toàn bộ cột thừa
df_final = df[keep_cols]

# === Làm sạch cột notes  ===
def clean_notes(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["notes"] = df["notes"].astype(str).apply(clean_notes)




# === Hàm trích xuất triệu chứng  ===
def extract_symptoms(text):
    text = str(text).lower()

    return {
        "cough": int(re.search(r"\bcough\b", text) is not None),
        "fever": int(re.search(r"\bfever\b", text) is not None or "high temperature" in text),
        "shortness_of_breath": int("shortness of breath" in text or "breathing" in text),
        "fatigue": int("fatigue" in text or "tired" in text or "weak" in text),
        "chest_pain": int("chest pain" in text),
        "healthy": int(any(x in text for x in [
            "healthy", "normal", "no symptoms", "normal health", 
            "normal condition", "no sign of disease", "normal chest", "routine check"
        ]))
    }


symptom_df = df["notes"].apply(extract_symptoms).apply(pd.Series)



# === Tần suất xuất hiện và chọn top triệu chứng phổ biến ===
symptom_sums = symptom_df.sum().sort_values(ascending=False)
top_symptoms = symptom_sums.head(6).index.tolist()


print(symptom_sums.head(6))



selected_symptoms = symptom_df[top_symptoms]

clean_df = pd.concat([df, selected_symptoms], axis=1)

# === Chuẩn hóa ===
if "sex" in clean_df.columns:
    clean_df["sex"] = clean_df["sex"].astype(str).str.upper().str.strip()
    clean_df["sex"] = clean_df["sex"].map({"M": 1, "F": 0})


if "finding" in clean_df.columns:
    clean_df["finding"] = clean_df["finding"].astype(str).str.upper().str.strip()

# ===  Giữ lại các cột quan trọng ===
cols = ["patientid", "sex", "age", "finding" ,"temperature" ,	"pO2 saturation" , "notes"] + top_symptoms
clean_df = clean_df[cols]



# === Xuất kết quả ===
print("Dữ liệu sau khi làm sạch:")
print(clean_df.head())


clean_df.head(20)
clean_df = pd.get_dummies(clean_df, columns=["finding"], prefix="", prefix_sep="")

cols = ['patientid', 'sex', 'age', 'temperature', 'pO2 saturation', "notes" ,
        'cough', 'fever', 'healthy', 'fatigue', 'shortness_of_breath', 'chest_pain',
        'COVID-19', 'PNEUMONIA', 'NORMAL']
clean_df =clean_df[cols]
clean_df.head(5)
import re

def normalize_notes(text):
    text = str(text).lower().strip()
    text = re.sub(r'[\u00A0\u200B]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(
        r'\b('
        r'no\s*symptom[s]?|'                     
        r'no\s*sign[s]?\s*(of)?\s*(infection|disease|illness)?|' 
        r'no\s*issues?\s*(detected|found)?|'     
        r'no\s*infection[s]?\s*(signs?|detected)?|' 
        r'no\s*illness|'                         
        r'no\s*fever|'                           
        r'no\s*health\s*(issue|problem)s?|'      
        r'healthy(\s*patient)?|'                 
        r'normal(\s*(health|condition|chest|lungs|xray|report))?|' 
        r'normal\s*check(\s*-?\s*up)?|'          
        r'normal\s*normal|'                      
        r'routine\s*check(\s*-?\s*up)?'          
        r')\b',
        'normal',
        text
    )

    text = re.sub(r'\b(normal\s+){2,}', 'normal ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


clean_df["notes"] = clean_df["notes"].apply(normalize_notes)
synonyms = {
    "fever": ["high temperature", "pyrexia", "temperature rise"],
    "cough": ["dry cough", "persistent cough", "light cough"],
    "fatigue": ["tiredness", "exhaustion", "feeling weak"],
    "shortness of breath": ["breathing difficulty", "short breath", "low oxygen"],
    "chest pain": ["chest discomfort", "tight chest", "pain in chest"],
    "normal": [
        "no symptoms",
        "no symptom",
        "no signs of infection",
        "no illness",
        "no health issues",
        "no health problems",
        "no disease detected",
        "no fever",
        "no infection found",
        "healthy",
        "healthy patient",
        "fit and well",
        "normal condition",
        "normal health",
        "normal report",
        "normal chest x-ray",
        "normal xray",
        "routine check-up",
        "routine examination",
        "standard medical check",
        "normal check-up",
        "no abnormality detected"
    ]
}

def augment_text(text, n=2):
    augmented = []
    for _ in range(n):
        aug = text
        for word, alt_list in synonyms.items():
            if word in aug:
                if random.random() < 0.7: 
                    aug = re.sub(rf'\b{word}\b', random.choice(alt_list), aug)
        if random.random() < 0.3:
            parts = aug.split()
            random.shuffle(parts)
            aug = " ".join(parts)
        augmented.append(aug.strip())
    return augmented


augmented_rows = []
for i, row in clean_df.iterrows():
    new_texts = augment_text(row["notes"], n=2)  
    for t in new_texts:
        new_row = row.copy()
        new_row["notes"] = t
        augmented_rows.append(new_row)

aug_df = pd.DataFrame(augmented_rows)


combined_df = pd.concat([clean_df, aug_df], ignore_index=True)


combined_df["patientid"] = [
    f"P{str(i+1).zfill(3)}" for i in range(len(combined_df))
]

# ===  Xuất kết quả ===
print(f"Dữ liệu gốc: {len(clean_df)} dòng")
print(f"Dữ liệu sau augmentation: {len(combined_df)} dòng\n")

print(combined_df.sample(10, random_state=42)[["patientid", "notes", "COVID-19", "PNEUMONIA", "NORMAL"]])


combined_df
# ===  Tải Pegasus paraphraser  ===
model_name = "tuner007/pegasus_paraphrase"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

paraphraser = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

print("Pegasus paraphraser đã sẵn sàng!")

# === Template sinh nội dung theo nhóm bệnh ===
templates = {
    "COVID-19": [
        "Patient tested positive for COVID-19 with {symptom}.",
        "Symptoms include {symptom}, suggesting COVID-19 infection.",
        "Reports of {symptom} and loss of smell/taste indicate COVID-19.",
        "Mild {symptom} observed, confirmed COVID-19 case.",
        "COVID-19 suspected due to {symptom} and fatigue."
    ],
    "PNEUMONIA": [
        "Chest X-ray shows signs of pneumonia with {symptom}.",
        "Patient suffers from {symptom}, consistent with pneumonia.",
        "Presence of {symptom} and chest pain indicates pneumonia.",
        "Findings reveal {symptom} and breathing difficulty.",
        "Pneumonia diagnosed; patient has {symptom} and cough."
    ],
    "NORMAL": [
        "No signs of illness detected, patient appears healthy.",
        "Routine check-up, normal findings throughout examination.",
        "Patient reports feeling fine, no abnormal symptoms.",
        "All test results normal, no health issues identified.",
        "Normal condition observed, no treatment required."
    ]
}

symptoms = [
    "fever", "dry cough", "fatigue", "shortness of breath",
    "chest pain", "tiredness", "sore throat", "loss of taste", "loss of smell"
]


new_rows = []
target_n = 1600

for _ in range(target_n):
    base = clean_df.sample(1).iloc[0].copy()

    # Xác định nhãn bệnh
    if base["COVID-19"] == 1:
        label = "COVID-19"
    elif base["PNEUMONIA"] == 1:
        label = "PNEUMONIA"
    else:
        label = "NORMAL"

    symptom = random.choice(symptoms)
    template = random.choice(templates[label])
    note = template.format(symptom=symptom)
    
    try:
        paraphrases = paraphraser(note, num_return_sequences=2, num_beams=5, truncation=True)
        new_note = random.choice(paraphrases)["generated_text"]
    except Exception:
        new_note = note  

    base["notes"] = new_note
    new_rows.append(base)


aug_df = pd.DataFrame(new_rows)
combined_df = pd.concat([clean_df, aug_df], ignore_index=True)
combined_df["patientid"] = [f"P{str(i+1).zfill(4)}" for i in range(len(combined_df))]

combined_df
# === Xuất file ===
combined_df.to_csv("../../data/metadata_clean_02.csv", index=False)

print("Đã tạo file 'clean_data.csv' với các cột triệu chứng chính:")
print(combined_df.head(10))
print(f"\nTổng số cột sau xử lý: {len(combined_df.columns)}")
