import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import logging
from datetime import datetime
from collections import defaultdict, deque
import copy
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class MetaPathExtractor:
    
    def __init__(self, hetero_data, max_length=3):
        self.hetero_data = hetero_data
        self.max_length = max_length
        self.edge_types = list(hetero_data.edge_types)
        
    def extract_bacteria_trait_metapaths(self):
        metapaths = []
        
        for edge_type in self.edge_types:
            src_type, relation, dst_type = edge_type
            if src_type == 'Bacteria' and dst_type == 'Trait':
                metapaths.append([edge_type])
        for first_edge in self.edge_types:
            src1, rel1, dst1 = first_edge
            if src1 == 'Bacteria':
                for second_edge in self.edge_types:
                    src2, rel2, dst2 = second_edge
                    if src2 == dst1 and dst2 == 'Trait':
                        metapaths.append([first_edge, second_edge])
       
        for first_edge in self.edge_types:
            src1, rel1, dst1 = first_edge
            if src1 == 'Bacteria':
                for second_edge in self.edge_types:
                    src2, rel2, dst2 = second_edge
                    if src2 == dst1:
                        for third_edge in self.edge_types:
                            src3, rel3, dst3 = third_edge
                            if src3 == dst2 and dst3 == 'Trait':
                                metapaths.append([first_edge, second_edge, third_edge])
        
        for i, path in enumerate(metapaths[:10]):  
            path_str = " -> ".join([f"{edge[0]}-{edge[1]}->{edge[2]}" for edge in path])
            print(f"  path {i+1}: {path_str}")
        
        return metapaths
    
    def compute_metapath_adjacency(self, metapath):
        try:
            first_edge = metapath[0]
            edge_index = self.hetero_data[first_edge].edge_index
            
            num_bacteria = self.hetero_data['Bacteria'].num_nodes
            intermediate_nodes = self.hetero_data[first_edge[2]].num_nodes
            
            adj = torch.zeros(num_bacteria, intermediate_nodes)
            adj[edge_index[0], edge_index[1]] = 1.0
            
            for edge in metapath[1:]:
                edge_index = self.hetero_data[edge].edge_index
                next_nodes = self.hetero_data[edge[2]].num_nodes
                
                next_adj = torch.zeros(intermediate_nodes, next_nodes)
                next_adj[edge_index[0], edge_index[1]] = 1.0
                
                adj = torch.mm(adj, next_adj)
                intermediate_nodes = next_nodes
            
            return adj
            
        except Exception as e:
            print(f"error: {e}")
            num_bacteria = self.hetero_data['Bacteria'].num_nodes
            num_traits = self.hetero_data['Trait'].num_nodes
            return torch.zeros(num_bacteria, num_traits)

class HCMGNNLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, node_types, edge_types, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.node_transforms = nn.ModuleDict()
        for node_type in node_types:
            self.node_transforms[node_type] = nn.Linear(in_dim, out_dim)
       
        conv_dict = {}
        valid_edge_types = []
        
        for edge_type in edge_types:
            src_type, relation, dst_type = edge_type
            if src_type in node_types and dst_type in node_types:
                conv_dict[edge_type] = SAGEConv(
                    in_channels=(-1, -1),
                    out_channels=out_dim,
                    normalize=True
                )
                valid_edge_types.append(edge_type)
            else:
                print(f"skip {edge_type}")
        
        print(f"valid_edge: {len(valid_edge_types)}/{len(edge_types)}")
        self.hetero_conv = HeteroConv(conv_dict, aggr='mean')
        
        self.layer_norms = nn.ModuleDict()
        for node_type in node_types:
            self.layer_norms[node_type] = nn.LayerNorm(out_dim)
        
        self.attention = nn.MultiheadAttention(out_dim, num_heads=4, dropout=dropout)
        
    def forward(self, x_dict, edge_index_dict):
        transformed_x = {}
        for node_type, x in x_dict.items():
            if x is not None:
                transformed_x[node_type] = self.node_transforms[node_type](x)
            else:
                print(f"waring:  {node_type}  None")
        
        valid_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type in transformed_x and dst_type in transformed_x:
                valid_edge_index_dict[edge_type] = edge_index
        
        if not valid_edge_index_dict:
            return transformed_x
        
        try:
            conv_out = self.hetero_conv(transformed_x, valid_edge_index_dict)
        except Exception as e:
            print(f"transformed_x keys: {list(transformed_x.keys())}")
            print(f"valid_edge_index_dict keys: {list(valid_edge_index_dict.keys())}")
            return transformed_x
        
        output = {}
        for node_type in conv_out:
            if node_type in transformed_x:
                residual = transformed_x[node_type] + conv_out[node_type]
                output[node_type] = self.layer_norms[node_type](residual)
                # Dropout
                output[node_type] = F.dropout(output[node_type], p=self.dropout, training=self.training)
            else:
                output[node_type] = conv_out[node_type]
        
        return output

class MetaPathAggregator(nn.Module):  
    
    def __init__(self, embedding_dim, num_metapaths, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_metapaths = num_metapaths
        
        self.metapath_weights = nn.Parameter(torch.ones(num_metapaths))
        
        self.metapath_transform = nn.Linear(embedding_dim, embedding_dim)
        
        self.attention_weights = nn.Parameter(torch.randn(num_metapaths, embedding_dim))
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, bacteria_embeddings, trait_embeddings, metapath_matrices):
        metapath_features = []
        
        for i, metapath_adj in enumerate(metapath_matrices):
            if metapath_adj.size(0) == bacteria_embeddings.size(0) and metapath_adj.size(1) == trait_embeddings.size(0):
                aggregated = torch.mm(metapath_adj, trait_embeddings)
                reverse_aggregated = torch.mm(metapath_adj.t(), bacteria_embeddings)
                
                combined = bacteria_embeddings + aggregated
                metapath_features.append(combined)
        
        if metapath_features:
            stacked_features = torch.stack(metapath_features, dim=0)  # [num_metapaths, num_bacteria, dim]
            
            weights = F.softmax(self.metapath_weights, dim=0)
            weighted_features = torch.sum(stacked_features * weights.view(-1, 1, 1), dim=0)
            
            output = self.metapath_transform(weighted_features)
            output = self.layer_norm(output)
            
            return output
        else:
            return bacteria_embeddings

class HCMGNNBasedMetaPathModel(nn.Module):
    
    def __init__(self, hetero_data, metapaths, embedding_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.hetero_data = hetero_data
        self.metapaths = metapaths
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.node_embeddings = nn.ModuleDict()
        for node_type in hetero_data.node_types:
            num_nodes = hetero_data[node_type].num_nodes
            if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                input_dim = hetero_data[node_type].x.size(1)
                self.node_embeddings[node_type] = nn.Linear(input_dim, embedding_dim)
            else:
                self.node_embeddings[node_type] = nn.Embedding(num_nodes, embedding_dim)
        
        self.hcmgnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = HCMGNNLayer(
                in_dim=embedding_dim,
                out_dim=embedding_dim,
                node_types=hetero_data.node_types,
                edge_types=hetero_data.edge_types,
                dropout=dropout
            )
            self.hcmgnn_layers.append(layer)
        
        self.metapath_aggregator = MetaPathAggregator(
            embedding_dim=embedding_dim,
            num_metapaths=len(metapaths),
            dropout=dropout
        )
        
        self.bacteria_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # *2因为结合了GNN和元路径特征
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        self.trait_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        self.temperature = nn.Parameter(torch.tensor(20.0))  # 更高的温度用于sharper分布
        self.similarity_bias = nn.Parameter(torch.tensor(0.0))
        
        self.metapath_matrices = self._precompute_metapath_matrices()
        
        self._init_weights()
    
    def _precompute_metapath_matrices(self):
        extractor = MetaPathExtractor(self.hetero_data)
        matrices = []
        
        for metapath in self.metapaths:
            adj_matrix = extractor.compute_metapath_adjacency(metapath)
            matrices.append(adj_matrix)
        
        return matrices
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)
    
    def get_initial_embeddings(self):
        x_dict = {}
        device = next(self.parameters()).device
        
        for node_type in self.hetero_data.node_types:
            try:
                if hasattr(self.hetero_data[node_type], 'x') and self.hetero_data[node_type].x is not None:
                    x = self.hetero_data[node_type].x.to(device)
                    if x.dim() == 2 and x.size(1) > 0:
                        x_dict[node_type] = self.node_embeddings[node_type](x)
                    else:
                
                        num_nodes = self.hetero_data[node_type].num_nodes
                        node_indices = torch.arange(num_nodes, device=device)
                        x_dict[node_type] = self.node_embeddings[node_type](node_indices)
                else:
                    num_nodes = self.hetero_data[node_type].num_nodes
                    node_indices = torch.arange(num_nodes, device=device)
                    x_dict[node_type] = self.node_embeddings[node_type](node_indices)
            except Exception as e:

                num_nodes = self.hetero_data[node_type].num_nodes
                node_indices = torch.arange(num_nodes, device=device)
                x_dict[node_type] = self.node_embeddings[node_type](node_indices)
        
        return x_dict
    
    def forward(self):
        device = next(self.parameters()).device
        
        x_dict = self.get_initial_embeddings()
        edge_index_dict = {k: v.to(device) for k, v in self.hetero_data.edge_index_dict.items()}
        
        for layer in self.hcmgnn_layers:
            x_dict = layer(x_dict, edge_index_dict)
       
        bacteria_gnn_emb = x_dict['Bacteria']
        trait_gnn_emb = x_dict['Trait']
        
        metapath_matrices_device = [m.to(device) for m in self.metapath_matrices]
        bacteria_metapath_emb = self.metapath_aggregator(
            bacteria_gnn_emb, trait_gnn_emb, metapath_matrices_device
        )
        
        bacteria_combined = torch.cat([bacteria_gnn_emb, bacteria_metapath_emb], dim=1)
        
        return bacteria_combined, trait_gnn_emb
    
    def compute_ranking_scores(self, bacteria_embeddings, trait_embeddings):
        bacteria_proj = self.bacteria_proj(bacteria_embeddings)
        trait_proj = self.trait_proj(trait_embeddings)
        
        scores = torch.mm(bacteria_proj, trait_proj.t())
        
        scores = scores * self.temperature + self.similarity_bias
        
        return scores

class EnhancedRankingLoss(nn.Module):
    
    def __init__(self, margin=2.0, lambda_param=0.5):
        super().__init__()
        self.margin = margin
        self.lambda_param = lambda_param  # 类似HCMGNN中的γ参数
        
    def forward(self, scores, positive_pairs):
        device = scores.device
        num_bacteria, num_traits = scores.shape
        
        labels = torch.zeros_like(scores)
        for bacteria_id, trait_id in positive_pairs:
            labels[bacteria_id, trait_id] = 1.0
        
        predictions = torch.sigmoid(scores)
        diff = labels - predictions
        
        positive_loss = torch.sum(labels * (diff ** 2))
        
        negative_loss = torch.sum((1 - labels) * (diff ** 2))
        
        total_samples = num_bacteria * num_traits
        positive_loss = positive_loss / total_samples
        negative_loss = negative_loss / total_samples
        
        loss = (1 - self.lambda_param) * positive_loss + self.lambda_param * negative_loss
        
        ranking_loss = 0.0
        num_ranking_pairs = 0
        
        for bacteria_id, pos_trait_id in positive_pairs:
            pos_score = scores[bacteria_id, pos_trait_id]
            neg_scores = scores[bacteria_id, labels[bacteria_id] == 0]
            
            if len(neg_scores) > 0:
                top_neg_scores = torch.topk(neg_scores, min(5, len(neg_scores))).values
                for neg_score in top_neg_scores:
                    ranking_loss += F.relu(self.margin - (pos_score - neg_score))
                    num_ranking_pairs += 1
        
        if num_ranking_pairs > 0:
            ranking_loss = ranking_loss / num_ranking_pairs
            loss = loss + 0.3 * ranking_loss
        
        return loss

class HCMGNNTrainer:
 
    def __init__(self, hetero_data, positive_pairs, device='cuda'):
        self.hetero_data = hetero_data.to(device)
        self.positive_pairs = positive_pairs
        self.device = device
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"hcmgnn_metapath_{timestamp}.log"
        
        self.save_dir = f"hcmgnn_models_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, self.log_filename)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        extractor = MetaPathExtractor(hetero_data)
        self.metapaths = extractor.extract_bacteria_trait_metapaths()
        
        self._split_data()
    
    def _split_data(self):
        random.shuffle(self.positive_pairs)
        
        split_idx = int(0.8 * len(self.positive_pairs))
        self.train_pairs = self.positive_pairs[:split_idx]
        self.val_pairs = self.positive_pairs[split_idx:]
    
    def compute_ranking_metrics(self, model):
        model.eval()
        
        with torch.no_grad():
            bacteria_embeddings, trait_embeddings = model()
            scores = model.compute_ranking_scores(bacteria_embeddings, trait_embeddings)
            
            metrics = self._compute_metrics(scores, self.val_pairs)
        
        return metrics
    
    def _compute_metrics(self, scores, positive_pairs):
        metrics = {
            'hit_1': 0.0,
            'hit_3': 0.0,
            'hit_5': 0.0,
            'hit_10': 0.0,
            'mrr': 0.0,
            'mean_rank': 0.0
        }
        
        bacteria_groups = defaultdict(list)
        for bacteria_id, trait_id in positive_pairs:
            bacteria_groups[bacteria_id].append(trait_id)
        
        total_queries = 0
        
        for bacteria_id, positive_traits in bacteria_groups.items():
            bacteria_scores = scores[bacteria_id]
            sorted_trait_indices = torch.argsort(bacteria_scores, descending=True)
            
            for trait_id in positive_traits:
                rank_tensor = (sorted_trait_indices == trait_id).nonzero(as_tuple=True)[0]
                if len(rank_tensor) > 0:
                    rank = rank_tensor.item() + 1
                    
                    metrics['hit_1'] += 1 if rank <= 1 else 0
                    metrics['hit_3'] += 1 if rank <= 3 else 0
                    metrics['hit_5'] += 1 if rank <= 5 else 0
                    metrics['hit_10'] += 1 if rank <= 10 else 0
                    metrics['mrr'] += 1.0 / rank
                    metrics['mean_rank'] += rank
                    
                    total_queries += 1
        
        if total_queries > 0:
            for key in metrics:
                metrics[key] /= total_queries
        
        return metrics
    
    def train_epoch(self, model, optimizer, loss_fn):
        model.train()
        
        bacteria_embeddings, trait_embeddings = model()
        scores = model.compute_ranking_scores(bacteria_embeddings, trait_embeddings)
        
        loss = loss_fn(scores, self.train_pairs)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def train(self, config):
        
        model = HCMGNNBasedMetaPathModel(
            hetero_data=self.hetero_data,
            metapaths=self.metapaths,
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        loss_fn = EnhancedRankingLoss(
            margin=config.get('margin', 2.0),
            lambda_param=config.get('lambda_param', 0.5)
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01
        )
        
        best_mrr = 0.0
        best_metrics = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            train_loss = self.train_epoch(model, optimizer, loss_fn)
            scheduler.step()
            
            if epoch % 20 == 0:
                val_metrics = self.compute_ranking_metrics(model)
                
                if val_metrics['mrr'] > best_mrr:
                    best_mrr = val_metrics['mrr']
                    best_metrics = val_metrics.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                self.logger.info(
                    f"Epoch {epoch:3d}: Loss={train_loss:.4f}, MRR={val_metrics['mrr']:.4f}, "
                    f"Hit@1={val_metrics['hit_1']:.4f}, Hit@3={val_metrics['hit_3']:.4f}"
                )
                
                if patience_counter >= config.get('patience', 25):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return best_metrics

def main():
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    data_path = "hetero_graph.pt"
    hetero_data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    all_bacteria_trait_edges = []
    
    bacteria_trait_relations = [
        ('Bacteria', 'related_to', 'Trait'),
        ('Bacteria', 'positive_relate', 'Trait'),
        ('Bacteria', 'negative_relate', 'Trait')
    ]
    
    for edge_type in bacteria_trait_relations:
        if edge_type in hetero_data.edge_types:
            edges = hetero_data[edge_type].edge_index.t()
            all_bacteria_trait_edges.append(edges)
            print(f"  {edge_type}: {edges.shape[0]} 条边")
    
    bacteria_trait_edges = torch.cat(all_bacteria_trait_edges, dim=0)
    positive_pairs = [(int(edge[0]), int(edge[1])) for edge in bacteria_trait_edges]
    
    logger = logging.getLogger(__name__)

    target_metrics = {
        'hit_1': 0.7947,
        'hit_3': 0.9417,
        'hit_5': 0.9641,
        'mrr': 0.8712,
        'mean_rank': 1.5 
    }
    
    for metric, value in target_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
   
    configs = [
        {
            'name': 'HCMGNN-Large',
            'embedding_dim': 512,
            'num_layers': 4,
            'dropout': 0.1,
            'lr': 0.0005,
            'weight_decay': 1e-5,
            'epochs': 400,
            'patience': 30,
            'margin': 2.0,
            'lambda_param': 0.7
        },
        {
            'name': 'HCMGNN-Medium', 
            'embedding_dim': 256,
            'num_layers': 3,
            'dropout': 0.15,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'epochs': 300,
            'patience': 25,
            'margin': 1.5,
            'lambda_param': 0.5
        },
        {
            'name': 'HCMGNN-Compact',
            'embedding_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'lr': 0.002,
            'weight_decay': 5e-4,
            'epochs': 200,
            'patience': 20,
            'margin': 1.0,
            'lambda_param': 0.3
        }
    ]
    
    trainer = HCMGNNTrainer(hetero_data, positive_pairs, device)
    
    best_overall = None
    best_config_name = None
    
    for config in configs:
        logger.info(f"\n======== {config['name']} ========")
        
        try:
            metrics = trainer.train(config)
            
            score = 0.0
            exceeded_count = 0
            
            for metric, value in metrics.items():
                target_value = target_metrics.get(metric, 0)
                exceeded = value >= target_value if metric != 'mean_rank' else value <= target_value
                status = "✅" if exceeded else "❌"
                
                if exceeded:
                    exceeded_count += 1
                
                logger.info(f"  {metric}: {value:.4f} ( {target_value:.4f}) {status}")
                
                if metric == 'mrr':
                    score += value * 0.4
                elif metric == 'hit_1':
                    score += value * 0.3
                elif metric == 'hit_3':
                    score += value * 0.2
                elif metric == 'hit_5':
                    score += value * 0.1
            
            logger.info(f"  score: {score:.4f}")
            logger.info(f"  exceeded_count: {exceeded_count}/{len(target_metrics)}")
            
            if best_overall is None or score > best_overall.get('score', 0):
                best_overall = metrics.copy()
                best_overall['score'] = score
                best_overall['name'] = config['name']
                best_overall['exceeded_count'] = exceeded_count
                best_config_name = config['name']
        
        except Exception as e:

            continue
    
    logger.info("\n" + "="*60)
    logger.info("="*60)
    
    if best_overall:
        logger.info(f"best_config_name: {best_config_name}")

        for metric, value in best_overall.items():
            if metric in target_metrics:
                target_value = target_metrics[metric]
                diff = value - target_value
                pct_diff = (diff / target_value) * 100 if target_value != 0 else 0
                exceeded = value >= target_value if metric != 'mean_rank' else value <= target_value
                status = "✅" if exceeded else "❌"
                logger.info(f"  {metric}: {value:.4f} vs HCMGNN {target_value:.4f} {status} {diff:+.4f} ({pct_diff:+.1f}%)")
        
    mrr_good = best_overall['mrr'] >= target_metrics['mrr'] * 0.95  
        hit1_good = best_overall['hit_1'] >= target_metrics['hit_1'] * 0.95
        
        if mrr_good and hit1_good:
            logger.info("  success")
        else:
            logger.info("  Further optimization is required")
    
    logger.info(f"\n end: {trainer.log_filename}")

if __name__ == "__main__":
    main()
