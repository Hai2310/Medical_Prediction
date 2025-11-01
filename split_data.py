from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("merged_finding.csv")
y = df["finding"]

# chia dữ liệu ra 2 tập train và test, chia đều lớp theo nhãn y
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=y
)

df.loc[train_idx].to_csv("./data/train.csv", index=False)
df.loc[test_idx].to_csv("./data/test.csv", index=False)