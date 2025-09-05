import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_relation_entropy(kg_triples):
    relations = [triple[1] for triple in kg_triples]
    relation_counts = Counter(relations)

    total_triples = len(relations)

    probabilities = [count / total_triples for count in relation_counts.values()]

    entropy = -np.sum([p * np.log(p) for p in probabilities if p > 0])

    num_relations = len(relation_counts)
    if num_relations > 1:
        max_entropy = np.log(num_relations)
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0

    return entropy, normalized_entropy, relation_counts

if __name__ == "__main__":
    import csv
    tsv_path = "final_use.tsv"
    triples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            subj_type, subj, rel, obj_type, obj = row
            triples.append((subj, rel, obj))

    entropy, normalized_entropy, relation_counts = calculate_relation_entropy(triples)
    print(f"Relation Entropy: {entropy:.4f}")
    print(f"Normalized Relation Entropy: {normalized_entropy:.4f}")
    print(f"Number of Relations: {len(relation_counts)}")
    print(f"Most common relations: {relation_counts.most_common(5)}")
