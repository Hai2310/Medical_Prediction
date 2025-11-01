import pandas as pd

df = pd.read_csv("./data/metadata.csv")
# Tạo cột 'finding' dựa vào 3 cột bệnh

def get_finding(row):
    if row['COVID-19']:
        return 'COVID-19'
    elif row['PNEUMONIA']:
        return 'PNEUMONIA'
    elif row['NORMAL']:
        return 'NORMAL'
    else:
        return 'UNKNOWN'

df['finding'] = df.apply(get_finding, axis=1)

# Nếu muốn xóa 3 cột cũ
df.drop(['COVID-19', 'PNEUMONIA', 'NORMAL', 'notes'], axis=1, inplace=True)

# Xuất kết quả ra file mới (tuỳ chọn)
df.to_csv("./data/merged_finding.csv", index=False)
print(df.head())
