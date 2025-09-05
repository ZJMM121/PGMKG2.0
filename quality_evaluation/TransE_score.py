import sys
sys.path.insert(0, "OpenKE")

import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from openke.module.model.TransE import TransE
from openke.data.TrainDataLoader import TrainDataLoader
import csv

import os
triple_set = set()
entity_set = set()
relation_set = set()
file_path = "final_use.tsv"
with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for row in reader:
        if len(row) < 5:
            continue
        subj = row[1].strip()
        rel = row[2].strip()
        obj = row[4].strip()
        triple_set.add((subj, rel, obj))
        entity_set.add(subj)
        entity_set.add(obj)
        relation_set.add(rel)

triples = list(triple_set)
unique_entities = list(entity_set)
unique_relations = list(relation_set)
entity2id = {entity: i for i, entity in enumerate(unique_entities)}
relation2id = {rel: i for i, rel in enumerate(unique_relations)}

tmp_dir = "./tmp_openke_data/"
os.makedirs(tmp_dir, exist_ok=True)
entity2id_path = os.path.join(tmp_dir, "entity2id.txt")
relation2id_path = os.path.join(tmp_dir, "relation2id.txt")
train2id_path = os.path.join(tmp_dir, "train2id.txt")

with open(entity2id_path, "w", encoding="utf-8") as f:
    f.write(f"{len(unique_entities)}\n")
    for e in unique_entities:
        f.write(f"{e}\t{entity2id[e]}\n")
with open(relation2id_path, "w", encoding="utf-8") as f:
    f.write(f"{len(unique_relations)}\n")
    for r in unique_relations:
        f.write(f"{r}\t{relation2id[r]}\n")
with open(train2id_path, "w", encoding="utf-8") as f:
    f.write(f"{len(triples)}\n")
    for h, r, t in triples:
        f.write(f"{entity2id[h]} {entity2id[t]} {relation2id[r]}\n")

from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.config import Trainer

loader = TrainDataLoader(
    in_path=tmp_dir,
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0
)

MODEL_PATH = os.path.join(tmp_dir, "transe_model_100.ckpt")

transe = TransE(
    ent_tot=len(entity2id),
    rel_tot=len(relation2id),
    dim=100,
    p_norm=1,
    norm_flag=True
)

if os.path.exists(MODEL_PATH):
    print(f"loading: {MODEL_PATH}")
    transe.load_checkpoint(MODEL_PATH)
else:
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=1.0),
        batch_size=loader.get_batch_size()
    )
    trainer = Trainer(
        model=model,
        data_loader=loader,
        train_times=100,
        alpha=0.01,
        use_gpu=False
    )
    trainer.run()
    transe.save_checkpoint(MODEL_PATH)

all_scores = []
for h, r, t in triples:
    h_id, r_id, t_id = entity2id[h], relation2id[r], entity2id[t]
    h_tensor = torch.tensor([h_id])
    r_tensor = torch.tensor([r_id])
    t_tensor = torch.tensor([t_id])
    score = -transe._calc(
        transe.ent_embeddings(h_tensor),
        transe.ent_embeddings(t_tensor),
        transe.rel_embeddings(r_tensor),
        mode='normal'
    ).item()
    all_scores.append(score)

mean_score = np.mean(all_scores)
var_score = np.var(all_scores)
standardized_scores = [(s - mean_score) / np.sqrt(var_score) for s in all_scores]
ic_scores = [norm.cdf(s) for s in standardized_scores]
kg_ic_score = np.mean(ic_scores)

print(f"kg_ic_score：{kg_ic_score:.4f}")

from scipy.stats import norm
ic_values = [norm.cdf(z) for z in standardized_scores]  
ic_variance = np.var(ic_values)  

print(f"ic_variance: {ic_variance:.6f}")
print(f"np.sqrt(ic_variance): {np.sqrt(ic_variance):.6f}")
print(f"1 - ic_variance: {1 - ic_variance:.6f}") 