import pandas as pd
from ete3 import NCBITaxa
import networkx as nx
import numpy as np
import json
from tqdm import tqdm

df = pd.read_csv("Bacteria_names.txt")  # 含 'name' 列
names = df['name'].tolist()
ncbi = NCBITaxa()

name_to_taxid = {}
for name in tqdm(names, desc="Mapping to TaxID"):
    try:
        taxid = ncbi.get_name_translator([name])[name][0]
        name_to_taxid[name] = taxid
    except:
        name_to_taxid[name] = None

with open("bacteria_taxid.json", "w") as f:
    json.dump(name_to_taxid, f)


taxids = [tid for tid in name_to_taxid.values() if tid is not None]
G = nx.DiGraph()

for taxid in tqdm(taxids, desc="Building taxonomy DAG"):
    try:
        lineage = ncbi.get_lineage(taxid)
        for i in range(len(lineage) - 1):
            parent, child = lineage[i], lineage[i + 1]
            G.add_edge(parent, child)
    except:
        continue

def semantic_contribution(graph, node, weight_decay=0.8):
    contributions = {}
    queue = [(node, 1.0)]
    while queue:
        current_node, weight = queue.pop(0)
        if current_node in contributions:
            contributions[current_node] += weight
        else:
            contributions[current_node] = weight
        for parent in graph.predecessors(current_node):
            queue.append((parent, weight * weight_decay))
    return contributions

all_terms = set()
microbiota_contributions = {}

for name, taxid in tqdm(name_to_taxid.items(), desc="Computing Wang vectors"):
    if taxid is None or taxid not in G:
        continue
    contrib = semantic_contribution(G, taxid)
    microbiota_contributions[name] = contrib
    all_terms.update(contrib.keys())

term_list = sorted(list(all_terms))
term_index = {term: i for i, term in enumerate(term_list)}


vectors = []
valid_names = []

for name in names:
    contrib = microbiota_contributions.get(name)
    if not contrib:
        continue
    vec = np.zeros(len(term_list))
    for term, score in contrib.items():
        vec[term_index[term]] = score
    vectors.append(vec)
    valid_names.append(name)

vectors = np.array(vectors)
np.save("bacteria_wang_features.npy", vectors)

with open("bacteria_wang_names.txt", "w") as f:
    for name in valid_names:
        f.write(name + "\n")
