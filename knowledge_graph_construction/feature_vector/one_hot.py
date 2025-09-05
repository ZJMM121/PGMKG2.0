import pandas as pd
import numpy as np
from scipy import sparse

entity_files = {
     "PMID": "PMID_names.txt",
     "ID": "ID_names.txt",
     "Segment": "Segment_names.txt",
     "Trait": "Trait_names.txt",
     "Taxonomy": "Taxonomy_names.txt"
}

for entity, path in entity_files.items():
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    n = len(names)
    one_hot = sparse.eye(n, dtype=np.float32, format='csr')
    sparse.save_npz(f"{entity.lower()}_onehot_features0725.npz", one_hot)
    with open(f"{entity.lower()}_names0725.txt", "w") as f2:
        for name in names:
            f2.write(name + "\n")
    print(f"{entity}: {n} saved")