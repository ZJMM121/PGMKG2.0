import pandas as pd
import numpy as np
from scipy import sparse

df = pd.read_csv("Pathway_names.txt")
pathway_names = df['name'].unique().tolist()

n = len(pathway_names)
one_hot_matrix = sparse.eye(n, dtype=np.float32, format='csr')

sparse.save_npz("pathway_features.npz", one_hot_matrix)

with open("pathway_names.txt", "w") as f:
    for name in pathway_names:
        f.write(name + "\n")
