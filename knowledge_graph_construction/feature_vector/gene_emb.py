import json
import time
import numpy as np
from goatools.obo_parser import GODag
from goatools.semantic import TermCounts, semantic_similarity
import gzip
from tqdm import tqdm

with open("Gene_names.txt") as f:
    gene_list = set(line.strip() for line in f if line.strip())

gene2go = {gene: [] for gene in gene_list}


with gzip.open("gene2go.gz", "rt") as f:
    for line in tqdm(f, desc="parse gene2go.gz"):
        if line.startswith("#"):
            continue
        tax_id, gene_id, go_id, _ = line.strip().split("\t")[:4]
        if tax_id != "9823":
            continue
        if gene_id in gene2go:
            gene2go[gene_id].append(go_id)

for gene in tqdm(gene2go, desc="duplicate"):
    gene2go[gene] = list(set(gene2go[gene]))

with open("gene_go_cache.json", "w") as f:
    json.dump(gene2go, f)

obodag = GODag("go-basic.obo")
with open("gene_go_cache.json") as f:
    gene2go = json.load(f)
genes = list(gene2go.keys())
termcounts = TermCounts(obodag, gene2go)

def gene_sim(g1, g2):
    gos1 = set(gene2go[g1])
    gos2 = set(gene2go[g2])
    if not gos1 or not gos2:
        return 0.0
    sims = []
    for go1 in gos1:
        max_sim = max([semantic_similarity(go1, go2, obodag, termcounts, 'Wang') for go2 in gos2 if go2 in obodag], default=0)
        sims.append(max_sim)
    for go2 in gos2:
        max_sim = max([semantic_similarity(go2, go1, obodag, termcounts, 'Wang') for go1 in gos1 if go1 in obodag], default=0)
        sims.append(max_sim)
    return np.mean(sims) if sims else 0.0

N = len(genes)
features = np.zeros((N, N))
for i, g1 in enumerate(tqdm(genes, desc="Calculation of gene similarity")):
    for j, g2 in enumerate(genes):
        if i <= j:
            sim = gene_sim(g1, g2)
            features[i, j] = sim
            features[j, i] = sim  

np.save("gene_wang_similarity.npy", features)
with open("gene_wang_similarity_name.txt", "w") as f:
    for gene in genes:
        f.write(gene + "\n")