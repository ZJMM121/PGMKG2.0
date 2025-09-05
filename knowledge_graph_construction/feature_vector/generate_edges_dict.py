import pandas as pd
import pickle
from collections import defaultdict

df = pd.read_csv("final_use.tsv", sep="\t")
entity2id = defaultdict(dict)
count = defaultdict(int)


for _, row in df.iterrows():
    for etype, name in [(row['Subject_type'], row['Subject_name']), (row['Object_type'], row['Object_name'])]:
        if name not in entity2id[etype]:
            entity2id[etype][name] = count[etype]
            count[etype] += 1


edge_index_dict = defaultdict(list)
for _, row in df.iterrows():
    src_id = entity2id[row['Subject_type']][row['Subject_name']]
    tgt_id = entity2id[row['Object_type']][row['Object_name']]
    edge_index_dict[(row['Subject_type'], row['relation'], row['Object_type'])].append((src_id, tgt_id))


with open("edges_dict.pkl", "wb") as f:
    pickle.dump(edge_index_dict, f)

print("✅ Saved edges_dict.pkl")
