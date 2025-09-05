import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import argparse

sys.path.append('pyHGT')
from pyHGT.data import Graph

def load_pgmkg_data(data_path):

    if data_path.endswith('.pt'):
        data = torch.load(data_path, map_location='cpu')
    elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"no: {data_path}")
    
    print(f"type: {type(data)}")
    if hasattr(data, 'node_types'):
        print(f"node_types: {data.node_types}")
        print(f"edge_types: {data.edge_types}")
    
    return data

def convert_hetero_to_graph(hetero_data):
    graph = Graph()
    
    node_feature = {}
    node_forward = {}
    node_bacward = {}
    node_id_counter = 0
    node_id_mapping = {}
    
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
            num_nodes = hetero_data[node_type].x.size(0)
            features = hetero_data[node_type].x.numpy()
        elif hasattr(hetero_data[node_type], 'num_nodes'):
            num_nodes = hetero_data[node_type].num_nodes
            features = np.random.randn(num_nodes, 128)  
        else:

            max_node_id = 0
            for edge_type in hetero_data.edge_types:
                if edge_type[0] == node_type or edge_type[2] == node_type:
                    edge_index = hetero_data[edge_type].edge_index
                    if edge_type[0] == node_type:
                        max_node_id = max(max_node_id, edge_index[0].max().item())
                    if edge_type[2] == node_type:
                        max_node_id = max(max_node_id, edge_index[1].max().item())
            num_nodes = max_node_id + 1
            features = np.random.randn(num_nodes, 128)
        
        print(f"  {node_type}: {num_nodes} ")
        
        node_feature[node_type] = pd.DataFrame(features)
        
        node_forward[node_type] = {}
        node_bacward[node_type] = {}
        
        for i in range(num_nodes):
            node_forward[node_type][i] = node_id_counter
            node_bacward[node_type][node_id_counter] = i
            node_id_mapping[node_id_counter] = (node_type, i)
            node_id_counter += 1
    
    edge_list = {}
    times = {}
    
    for edge_type in hetero_data.edge_types:
        src_type, relation, dst_type = edge_type
        
        edge_index = hetero_data[edge_type].edge_index
        num_edges = edge_index.size(1)
        
        if src_type not in edge_list:
            edge_list[src_type] = {}
        if relation not in edge_list[src_type]:
            edge_list[src_type][relation] = {}
        if dst_type not in edge_list[src_type][relation]:
            edge_list[src_type][relation][dst_type] = []
        
        for i in range(num_edges):
            src_id = edge_index[0, i].item()
            dst_id = edge_index[1, i].item()
            
            src_internal_id = node_forward[src_type][src_id]
            dst_internal_id = node_forward[dst_type][dst_id]
            
            edge_list[src_type][relation][dst_type].append((src_internal_id, dst_internal_id, 2020))
    
    for node_type in node_feature:
        times[node_type] = {nid: 2020 for nid in node_bacward[node_type].keys()}
    
    graph.node_feature = node_feature
    graph.node_forward = node_forward
    graph.node_bacward = node_bacward
    graph.edge_list = edge_list
    graph.times = times
    
    return graph, node_id_mapping

def create_task_labels(hetero_data, node_id_mapping):
    bacteria_taxonomy_labels = {}

    if 'Bacteria' in hetero_data.node_types:
        if hasattr(hetero_data['Bacteria'], 'y') and hetero_data['Bacteria'].y is not None:
            labels = hetero_data['Bacteria'].y.numpy()
            for i, label in enumerate(labels):
                for internal_id, (node_type, orig_id) in node_id_mapping.items():
                    if node_type == 'Bacteria' and orig_id == i:
                        bacteria_taxonomy_labels[internal_id] = int(label)
                        break
        else:
            bacteria_count = 0
            for internal_id, (node_type, orig_id) in node_id_mapping.items():
                if node_type == 'Bacteria':
                    bacteria_taxonomy_labels[internal_id] = bacteria_count % 6
                    bacteria_count += 1

    bacteria_metabolite_links = []
    for edge_type in hetero_data.edge_types:
        src_type, relation, dst_type = edge_type
        if (src_type == 'Bacteria' and dst_type == 'Metabolite') or \
           (src_type == 'Metabolite' and dst_type == 'Bacteria'):

            edge_index = hetero_data[edge_type].edge_index

            for i in range(edge_index.size(1)):
                if src_type == 'Bacteria':
                    bacteria_orig_id = edge_index[0, i].item()
                    metabolite_orig_id = edge_index[1, i].item()
                else:
                    metabolite_orig_id = edge_index[0, i].item()
                    bacteria_orig_id = edge_index[1, i].item()

                bacteria_metabolite_links.append((bacteria_orig_id, metabolite_orig_id, relation))

    bacteria_trait_links = []
    for edge_type in hetero_data.edge_types:
        src_type, relation, dst_type = edge_type
        if (src_type == 'Bacteria' and dst_type == 'Trait') or \
           (src_type == 'Trait' and dst_type == 'Bacteria'):

            edge_index = hetero_data[edge_type].edge_index
            for i in range(edge_index.size(1)):
                if src_type == 'Bacteria':
                    bacteria_orig_id = edge_index[0, i].item()
                    trait_orig_id = edge_index[1, i].item()
                else:
                    trait_orig_id = edge_index[0, i].item()
                    bacteria_orig_id = edge_index[1, i].item()

                bacteria_trait_links.append((bacteria_orig_id, trait_orig_id, relation))

    return bacteria_taxonomy_labels, bacteria_metabolite_links, bacteria_trait_links

def save_hgt_format(graph, node_id_mapping, bacteria_taxonomy_labels, bacteria_metabolite_links, bacteria_trait_links, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'graph_PGMKG.pk'), 'wb') as f:
        pickle.dump(graph, f)

    with open(os.path.join(output_dir, 'node_id_mapping.pk'), 'wb') as f:
        pickle.dump(node_id_mapping, f)

    with open(os.path.join(output_dir, 'bacteria_taxonomy_labels.pk'), 'wb') as f:
        pickle.dump(bacteria_taxonomy_labels, f)

    with open(os.path.join(output_dir, 'bacteria_metabolite_links.pk'), 'wb') as f:
        pickle.dump(bacteria_metabolite_links, f)

    with open(os.path.join(output_dir, 'bacteria_trait_links.pk'), 'wb') as f:
        pickle.dump(bacteria_trait_links, f)

    meta_graph_path = os.path.join(output_dir, 'meta_graph.txt')
    with open(meta_graph_path, 'w') as f:
        f.write("# PGMKG Meta Graph Structure\n\n")
        f.write("## Node Types:\n")
        for node_type in graph.node_feature.keys():
            f.write(f"- {node_type}: {len(graph.node_feature[node_type])} nodes\n")
        f.write("\n## Edge Types:\n")
        for src_type in graph.edge_list:
            for relation in graph.edge_list[src_type]:
                for dst_type in graph.edge_list[src_type][relation]:
                    num_edges = len(graph.edge_list[src_type][relation][dst_type])
                    f.write(f"- {src_type} --[{relation}]--> {dst_type}: {num_edges} edges\n")

def main():
    parser = argparse.ArgumentParser(description='PGMKG to HGT format converter')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./hgt_format')

    args = parser.parse_args()

    try:
        hetero_data = load_pgmkg_data(args.input)
        graph, node_id_mapping = convert_hetero_to_graph(hetero_data)
        bacteria_taxonomy_labels, bacteria_metabolite_links, bacteria_trait_links = create_task_labels(hetero_data, node_id_mapping)
        save_hgt_format(graph, node_id_mapping, bacteria_taxonomy_labels, bacteria_metabolite_links, bacteria_trait_links, args.output)

    except Exception as e:
        print(f"❌ error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
