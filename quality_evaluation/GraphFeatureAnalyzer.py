import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
from itertools import islice



class GraphFeatureAnalyzer:

    def __init__(self, kg_triples: List[Tuple[str, str, str]] = None):
        self.kg_triples = kg_triples or []
        self.G = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        self.G = nx.DiGraph()
        for h, r, t in tqdm(self.kg_triples, desc="construct", ncols=80):
            self.G.add_edge(h, t, relation=r)
    
    def update_triples(self, new_triples: List[Tuple[str, str, str]]) -> None:
        self.kg_triples.extend(new_triples)
        self._build_graph()
    
    def calculate_basic_features(self) -> Dict[str, float]:

        if not self.G:
            return {}
        
        degrees = dict(self.G.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if len(degrees) > 0 else 0
        
        G_undirected = self.G.to_undirected()
        connected_components = list(nx.connected_components(G_undirected))
        if not connected_components:
            largest_cc_ratio = 0
        else:
            largest_cc = max(connected_components, key=len)
            largest_cc_ratio = len(largest_cc) / len(G_undirected.nodes()) if len(G_undirected.nodes()) > 0 else 0
        
        path_richness = self._calculate_path_richness()
        
        return {
            "avg_degree": avg_degree,
            "largest_cc_ratio": largest_cc_ratio,
            "path_richness": path_richness
        }
    
    def _calculate_path_richness(self, cutoff: int = 3, sample_size: Optional[int] = 200, max_paths_per_pair: int = 10, G=None) -> float:
        G = G if G is not None else self.G
        if not G:
            return 0
        path_counts = 0
        entity_pairs = 0
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        if num_nodes < 2:
            return 0
        sample_size = min(sample_size or 200, num_nodes * (num_nodes - 1) // 2)
        for _ in tqdm(range(sample_size), desc="Sampling path diversity", ncols=80):
            source, target = np.random.choice(nodes, 2, replace=False)
            entity_pairs += 1
            try:
                paths = list(islice(nx.all_simple_paths(G, source, target, cutoff=cutoff), max_paths_per_pair))
                path_counts += len(paths)
            except nx.NetworkXNoPath:
                continue
        total_pairs = num_nodes * (num_nodes - 1) // 2
        path_counts = path_counts * (total_pairs / entity_pairs) if entity_pairs > 0 else 0
        entity_pairs = total_pairs
        return path_counts / entity_pairs if entity_pairs > 0 else 0
    
    def calculate_relation_aware_features(self) -> Dict[str, Dict[str, float]]:
        if not self.G or not self.kg_triples:
            return {}
        
        relation_subgraphs = defaultdict(nx.DiGraph)
        for h, r, t in tqdm(self.kg_triples, desc="Construct the relationship subgraph", ncols=80):
            relation_subgraphs[r].add_edge(h, t)
        
        relation_features = {}
        for rel, G_rel in relation_subgraphs.items():
            if len(G_rel.nodes()) < 2:
                continue
            
            degrees = dict(G_rel.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if len(degrees) > 0 else 0
            
            G_rel_undirected = G_rel.to_undirected()
            connected_components = list(nx.connected_components(G_rel_undirected))
            if not connected_components:
                largest_cc_ratio = 0
            else:
                largest_cc = max(connected_components, key=len)
                largest_cc_ratio = len(largest_cc) / len(G_rel_undirected.nodes()) if len(G_rel_undirected.nodes()) > 0 else 0
            
            path_richness = self._calculate_path_richness(G=G_rel, cutoff=3, sample_size=1000)
            
            relation_features[rel] = {
                "avg_degree": avg_degree,
                "largest_cc_ratio": largest_cc_ratio,
                "path_richness": path_richness,
                "node_count": len(G_rel.nodes()),
                "edge_count": len(G_rel.edges())
            }
        
        return relation_features

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


    analyzer = GraphFeatureAnalyzer(triples)

    features = analyzer.calculate_basic_features()
    for feature, value in features.items():
        print(f"  - {feature}: {value:.4f}")

    num_nodes = len(analyzer.G.nodes()) if analyzer.G is not None else 0
    norm_features = {}
    if num_nodes > 1:
        norm_features['avg_degree'] = features['avg_degree'] / (num_nodes - 1)
    else:
        norm_features['avg_degree'] = 0.0
    norm_features['largest_cc_ratio'] = features['largest_cc_ratio']
    norm_features['path_richness'] = features['path_richness']
    for feature, value in norm_features.items():
        print(f"  - {feature}: {value:.4f}")

    rel_features = analyzer.calculate_relation_aware_features()
    for rel, stats in rel_features.items():
        print(f"  - {rel}:")
        for feature, value in stats.items():
            print(f"    - {feature}: {value:.4f}")
