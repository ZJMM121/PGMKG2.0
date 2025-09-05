import pandas as pd

with open("bacteria_names.txt", "r", encoding="utf-8") as f:
    keep_names = set(line.strip() for line in f if line.strip())

df = pd.read_csv("quintuple0724.csv", sep=",", dtype=str)

def bacteria_in_keep(row):
    if row["Subject_type"] == "Bacteria" and row["Subject_name"] not in keep_names:
        return False
    if row["Object_type"] == "Bacteria" and row["Object_name"] not in keep_names:
        return False
    return True

df_filtered = df[df.apply(bacteria_in_keep, axis=1)]

df_filtered.to_csv("final_use.tsv", sep="\t", index=False)
