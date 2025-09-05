import torch
import numpy as np
import os
import pickle
from collections import defaultdict, Counter
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

def convert_hetero_data_to_hgb(data_path, output_dir):
    data = torch.load(data_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    node_type_map = {ntype: i for i, ntype in enumerate(data.node_types)}
    node_shifts = {}
    node_counts = {}
    current_shift = 0
    
    for ntype in data.node_types:
        node_shifts[ntype] = current_shift
        node_counts[ntype] = data[ntype].x.shape[0]
        current_shift += node_counts[ntype]
    
    with open(os.path.join(output_dir, 'node.dat'), 'w', encoding='utf-8') as f:
        for ntype in data.node_types:
            features = data[ntype].x.numpy()
            type_id = node_type_map[ntype]
            shift = node_shifts[ntype]
            
            for i in range(features.shape[0]):
                node_id = shift + i
                node_name = f"{ntype}_{i}"
                feature_str = ','.join([f"{x:.6f}" for x in features[i]])
                f.write(f"{node_id}\t{node_name}\t{type_id}\t{feature_str}\n")
    
    edge_type_map = {etype: i for i, etype in enumerate(data.edge_types)}
    all_edges = []
    edge_statistics = {}
    
    for etype in data.edge_types:
        edge_index = data[etype].edge_index.numpy()
        src_type, rel_type, dst_type = etype
        
        src_indices = edge_index[0] + node_shifts[src_type]
        dst_indices = edge_index[1] + node_shifts[dst_type]
        
        rel_id = edge_type_map[etype]
        edge_count = edge_index.shape[1]
        
        edge_statistics[etype] = {
            'count': edge_count,
            'src_type': src_type,
            'dst_type': dst_type,
            'rel_id': rel_id
        }
        
        for i in range(edge_count):
            all_edges.append((src_indices[i], dst_indices[i], rel_id, 1.0))  
    
    edges_by_type = defaultdict(list)
    for edge in all_edges:
        src, dst, rel_id, weight = edge
        edges_by_type[rel_id].append(edge)
    
    train_edges = []
    valid_edges = []
    test_edges = []
    
    for rel_id, edges in edges_by_type.items():
        if len(edges) < 10:  
            train_edges.extend(edges)
            continue
            
        train_e, temp_e = train_test_split(edges, test_size=0.3, random_state=42)
        valid_e, test_e = train_test_split(temp_e, test_size=0.5, random_state=42)
        
        train_edges.extend(train_e)
        valid_edges.extend(valid_e)
        test_edges.extend(test_e)
    

    with open(os.path.join(output_dir, 'link.dat'), 'w', encoding='utf-8') as f:
        for edge in train_edges + valid_edges:
            src, dst, rel_id, weight = edge
            f.write(f"{src}\t{dst}\t{rel_id}\t{weight}\n")

    with open(os.path.join(output_dir, 'link.dat.test'), 'w', encoding='utf-8') as f:
        for edge in test_edges:
            src, dst, rel_id, weight = edge
            f.write(f"{src}\t{dst}\t{rel_id}\t{weight}\n")
    
    metadata = {
        'node_type_map': node_type_map,
        'edge_type_map': edge_type_map,
        'node_shifts': node_shifts,
        'node_counts': node_counts,
        'edge_statistics': edge_statistics,
        'original_node_types': data.node_types,
        'original_edge_types': data.edge_types,
        'total_nodes': sum(node_counts.values()),
        'total_edges': len(all_edges),
        'train_edges': len(train_edges),
        'valid_edges': len(valid_edges),
        'test_edges': len(test_edges)
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata

if __name__ == "__main__":
    data_path = "hetero_graph.pt"
    output_dir = "PGMKG_HGB_format"
    
    metadata = convert_hetero_data_to_hgb(data_path, output_dir)
    