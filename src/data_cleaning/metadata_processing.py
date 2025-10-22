import pandas as pd
import re

def clean_age(age):
    if pd.isna(age):
        return ""
    age = str(age).lower().strip()
    if "+" in age:
        return re.sub(r"\+", "", age)
    if "early" in age:
        return re.sub(r"\D", "", age)
    if "mid" in age:
        num = int(re.sub(r"\D", "", age))
        return str(num + 5)  # mid 50s → 55
    if age.endswith("s"):  # e.g., 40s
        return re.sub(r"\D", "", age)
    return age

def clean_sex(sex):
    if pd.isna(sex): return ""
    sex = str(sex).strip().upper()
    if sex.startswith("M"): return "M"
    if sex.startswith("F"): return "F"
    return ""

def clean_temp(temp):
    if pd.isna(temp): return ""
    temp = str(temp)
    temp = temp.replace("+", "")
    match = re.findall(r"\d+(\.\d+)?", temp)
    return match[0] if match else ""

def clean_po2(val):
    if pd.isna(val): return ""
    val = str(val).lower()
    if "-" in val:  # 91-92 → 91
        return val.split("-")[0]
    if "low 90" in val: return "90"
    if "80" in val: return "80"
    match = re.findall(r"\d+", val)
    return match[0] if match else ""

def clean_notes(row):
    # ví dụ: "61M, fever and dry cough, desaturation to 88% RA. Tested positive for COVID-19."
    parts = []
    if row["age"] and row["sex"]:
        parts.append(f"{row['age']}{row['sex']}")
    elif row["age"]:
        parts.append(f"{row['age']}yo")
    if row["finding"]:
        parts.append(row["finding"])
    if row["temperature"]:
        parts.append(f"fever {row['temperature']}°C")
    if row["pO2 saturation"]:
        parts.append(f"desaturation to {row['pO2 saturation']}% RA")
    notes = ", ".join(parts)
    if "COVID" in (row["finding"] or "").upper():
        notes += ". Tested positive for COVID-19."
    return notes

def standardize_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df["sex"] = df["sex"].apply(clean_sex)
    df["age"] = df["age"].apply(clean_age)
    df["temperature"] = df["temperature"].apply(clean_temp)
    df["pO2 saturation"] = df["pO2 saturation"].apply(clean_po2)
    df["finding"] = df["finding"].fillna("No finding")
    df["notes"] = df.apply(clean_notes, axis=1)
    
    df.to_csv(output_file, index=False)

# Ví dụ chạy
standardize_csv(f"C:\Users\Lenovo\Desktop\Project ML\Medical Prediction - Copy\data\metadata.csv", f"C:\Users\Lenovo\Desktop\Project ML\Medical Prediction - Copy\data\metadata_clean.csv")
