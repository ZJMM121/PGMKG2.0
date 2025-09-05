from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set, Optional

class TypeConstraintLearner:
    def __init__(self, min_support: int = 5, confidence_threshold: float = 0.8):
       
        self.min_support = min_support
        self.confidence_threshold = confidence_threshold
        self.entity_types = {}          
        self.rel_head_type_counts = defaultdict(Counter)  
        self.rel_tail_type_counts = defaultdict(Counter)  
        self.total_rel_triples = defaultdict(int)         
        self.learned_constraints = {}                    
    
    def learn_from_triples(self, triples: List[Tuple[str, str, str]]) -> None:
      
        self._infer_entity_types(triples)
        
        for head, rel, tail in triples:
            if head in self.entity_types and tail in self.entity_types:
                head_type = self.entity_types[head]
                tail_type = self.entity_types[tail]
                
                self.rel_head_type_counts[rel][head_type] += 1
                self.rel_tail_type_counts[rel][tail_type] += 1
                self.total_rel_triples[rel] += 1
        
        self._extract_constraints()
    
    def _infer_entity_types(self, triples: List[Tuple[str, str, str]]) -> None:

        global entity_type_dict
        self.entity_types = entity_type_dict.copy()
    
    def _extract_constraints(self) -> None:
        for rel in self.rel_head_type_counts:
            total = self.total_rel_triples[rel]
            if total < self.min_support:
                continue
                
            head_types = self.rel_head_type_counts[rel]
            main_head_type = max(head_types, key=head_types.get)
            head_confidence = head_types[main_head_type] / total
            
            tail_types = self.rel_tail_type_counts[rel]
            main_tail_type = max(tail_types, key=tail_types.get)
            tail_confidence = tail_types[main_tail_type] / total
            
            if head_confidence >= self.confidence_threshold and tail_confidence >= self.confidence_threshold:
                self.learned_constraints[rel] = (main_head_type, main_tail_type, head_confidence, tail_confidence)
    
    def get_constraints(self) -> Dict[str, Tuple[str, str, float, float]]:
        return self.learned_constraints
    
    def evaluate_kg(self, triples: List[Tuple[str, str, str]]) -> float:
        if not self.learned_constraints:
            raise ValueError("You haven't learned any constraint rules yet. Please call the "learn_from_triples" method first.")
        
        valid_count = 0
        total_count = 0
        
        for head, rel, tail in triples:
            if rel in self.learned_constraints:
                head_type = self.entity_types.get(head, None)
                tail_type = self.entity_types.get(tail, None)
                expected_head, expected_tail, _, _ = self.learned_constraints[rel]
                
                if head_type == expected_head and tail_type == expected_tail:
                    valid_count += 1
                total_count += 1
        
        return valid_count / total_count if total_count > 0 else 1.0
    
    def visualize_constraints(self, top_n: int = 10) -> None:
        if not self.learned_constraints:
            print("There are no visible constraints or rules.")
            return
        
        sorted_constraints = sorted(
            self.learned_constraints.items(), 
            key=lambda x: self.total_rel_triples[x[0]], 
            reverse=True
        )[:top_n]
        


# 示例用法
if __name__ == "__main__":
    import csv
    tsv_path = "final_use.tsv"
    triples = []
    entity_type_dict = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            subj_type, subj, rel, obj_type, obj = row
            triples.append((subj, rel, obj))
            entity_type_dict[subj] = subj_type
            entity_type_dict[obj] = obj_type

    class PatchedTypeConstraintLearner(TypeConstraintLearner):
        def _infer_entity_types(self, triples):
            self.entity_types = entity_type_dict.copy()

    learner = PatchedTypeConstraintLearner(min_support=10, confidence_threshold=0.8)
    learner.learn_from_triples(triples)
    constraints = learner.get_constraints() 
    consistency_rate = learner.evaluate_kg(triples)
    print(f"\nConsistency rate of knowledge graph types: {consistency_rate:.2%}")
