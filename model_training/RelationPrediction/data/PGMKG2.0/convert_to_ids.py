#!/usr/bin/env python3
"""
将知识图谱三元组从实体/关系名称转换为对应的ID
Convert knowledge graph triples from entity/relation names to corresponding IDs
"""

import os
from collections import defaultdict


def load_dict_file(filepath):
    """
    加载字典文件，返回name到id的映射
    Load dictionary file and return name to id mapping
    """
    name_to_id = {}
    id_to_name = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    id_val = int(parts[0])
                    name = parts[1]
                    name_to_id[name] = id_val
                    id_to_name[id_val] = name
    
    return name_to_id, id_to_name


def convert_triples_to_ids(input_file, output_file, entity_dict, relation_dict):
    """
    将三元组文件从名称转换为ID
    Convert triples file from names to IDs
    """
    print(f"Converting {input_file} -> {output_file}")
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Warning: Line {line_num} has {len(parts)} parts instead of 3: {line}")
                skipped_count += 1
                continue
            
            head_name, relation_name, tail_name = parts
            
            # 查找对应的ID
            head_id = entity_dict.get(head_name)
            relation_id = relation_dict.get(relation_name)
            tail_id = entity_dict.get(tail_name)
            
            if head_id is None:
                print(f"Warning: Entity '{head_name}' not found in entity dictionary (line {line_num})")
                skipped_count += 1
                continue
            
            if relation_id is None:
                print(f"Warning: Relation '{relation_name}' not found in relation dictionary (line {line_num})")
                skipped_count += 1
                continue
                
            if tail_id is None:
                print(f"Warning: Entity '{tail_name}' not found in entity dictionary (line {line_num})")
                skipped_count += 1
                continue
            
            # 写入转换后的三元组
            fout.write(f"{head_id}\t{relation_id}\t{tail_id}\n")
            converted_count += 1
    
    print(f"  Converted: {converted_count} triples")
    print(f"  Skipped: {skipped_count} triples")
    return converted_count, skipped_count


def build_dictionaries_from_data():
    """
    从数据文件构建实体和关系字典
    Build entity and relation dictionaries from data files
    """
    print("Building dictionaries from data files...")
    
    # 收集所有实体和关系
    entities = set()
    relations = set()
    
    # 从训练、验证、测试集收集
    for filename in ['train.txt', 'valid.txt', 'test.txt']:
        if os.path.exists(filename):
            print(f"  Processing {filename}")
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) == 3:
                            head, relation, tail = parts
                            entities.add(head)
                            entities.add(tail)
                            relations.add(relation)
    
    # 保存实体字典
    entities_list = sorted(list(entities))
    with open('entities.dict', 'w', encoding='utf-8') as f:
        for i, entity in enumerate(entities_list):
            f.write(f"{i}\t{entity}\n")
    
    # 保存关系字典
    relations_list = sorted(list(relations))
    with open('relations.dict', 'w', encoding='utf-8') as f:
        for i, relation in enumerate(relations_list):
            f.write(f"{i}\t{relation}\n")
    
    # 保存所有实体和关系列表
    with open('entities_all.txt', 'w', encoding='utf-8') as f:
        for entity in entities_list:
            f.write(f"{entity}\n")
    
    with open('relations_all.txt', 'w', encoding='utf-8') as f:
        for relation in relations_list:
            f.write(f"{relation}\n")
    
    print(f"  Found {len(entities)} unique entities")
    print(f"  Found {len(relations)} unique relations")
    
    return len(entities), len(relations)


def main():
    """
    主函数
    Main function
    """
    print("Knowledge Graph Triples ID Conversion Tool")
    print("=" * 50)
    
    # 检查是否存在字典文件
    if not os.path.exists('entities.dict') or not os.path.exists('relations.dict'):
        print("Dictionary files not found. Building from data files...")
        build_dictionaries_from_data()
    
    # 加载字典
    print("\nLoading dictionaries...")
    entity_name_to_id, entity_id_to_name = load_dict_file('entities.dict')
    relation_name_to_id, relation_id_to_name = load_dict_file('relations.dict')
    
    print(f"Loaded {len(entity_name_to_id)} entities")
    print(f"Loaded {len(relation_name_to_id)} relations")
    
    # 转换文件
    print("\nConverting triples to IDs...")
    
    total_converted = 0
    total_skipped = 0
    
    # 转换训练集
    if os.path.exists('train.txt'):
        converted, skipped = convert_triples_to_ids(
            'train.txt', 'train_ids.txt',
            entity_name_to_id, relation_name_to_id
        )
        total_converted += converted
        total_skipped += skipped
    
    # 转换验证集
    if os.path.exists('valid.txt'):
        converted, skipped = convert_triples_to_ids(
            'valid.txt', 'valid_ids.txt',
            entity_name_to_id, relation_name_to_id
        )
        total_converted += converted
        total_skipped += skipped
    
    # 转换测试集
    if os.path.exists('test.txt'):
        converted, skipped = convert_triples_to_ids(
            'test.txt', 'test_ids.txt',
            entity_name_to_id, relation_name_to_id
        )
        total_converted += converted
        total_skipped += skipped
    
    print("\n" + "=" * 50)
    print("Conversion Summary:")
    print(f"Total converted: {total_converted} triples")
    print(f"Total skipped: {total_skipped} triples")
    print("Conversion completed!")


def verify_conversion():
    """
    验证转换结果
    Verify conversion results
    """
    print("\nVerifying conversion...")
    
    # 加载字典
    entity_name_to_id, entity_id_to_name = load_dict_file('entities.dict')
    relation_name_to_id, relation_id_to_name = load_dict_file('relations.dict')
    
    # 验证文件对
    file_pairs = [
        ('train.txt', 'train_ids.txt'),
        ('valid.txt', 'valid_ids.txt'),
        ('test.txt', 'test_ids.txt')
    ]
    
    for name_file, id_file in file_pairs:
        if os.path.exists(name_file) and os.path.exists(id_file):
            print(f"Verifying {name_file} <-> {id_file}")
            
            # 读取原始文件
            with open(name_file, 'r', encoding='utf-8') as f:
                name_lines = [line.strip() for line in f if line.strip()]
            
            # 读取ID文件
            with open(id_file, 'r', encoding='utf-8') as f:
                id_lines = [line.strip() for line in f if line.strip()]
            
            if len(name_lines) != len(id_lines):
                print(f"  ERROR: Line count mismatch! {len(name_lines)} vs {len(id_lines)}")
                continue
            
            # 随机验证几行
            import random
            sample_indices = random.sample(range(len(name_lines)), min(5, len(name_lines)))
            
            for i in sample_indices:
                name_parts = name_lines[i].split('\t')
                id_parts = id_lines[i].split('\t')
                
                if len(name_parts) == 3 and len(id_parts) == 3:
                    head_name, rel_name, tail_name = name_parts
                    head_id, rel_id, tail_id = map(int, id_parts)
                    
                    # 验证转换
                    if (entity_name_to_id.get(head_name) == head_id and
                        relation_name_to_id.get(rel_name) == rel_id and
                        entity_name_to_id.get(tail_name) == tail_id):
                        print(f"  ✓ Line {i+1}: {head_name} -> {head_id}")
                    else:
                        print(f"  ✗ Line {i+1}: Conversion error!")
            
            print(f"  Verified {len(name_lines)} triples")


if __name__ == "__main__":
    main()
    
    # 可选：验证转换结果
    verify_conversion()
