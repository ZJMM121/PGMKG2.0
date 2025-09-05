import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def smiles_to_morgan(smiles, radius=2, nBits=1024):
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    except Exception as e:
        print(f"  Error")
        return None
    return None


def build_metabolite_features_from_csv(input_csv, output_npy, output_names_txt, output_stats_csv=None):
    
    try:
        df = pd.read_csv(input_csv)
        
        required_cols = ['Compound', 'CID', 'SMILES']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error：{missing_cols}")
            print(f"Actual: {list(df.columns)}")
            return
        
    except Exception as e:
        print(f"failure: {e}")
        return
    
 
    valid_df = df[
        (df['SMILES'].notna()) & 
        (df['SMILES'] != 'Not Found') & 
        (df['SMILES'] != '') &
        (df['Compound'].notna()) &
        (df['Compound'] != '')
    ].copy()
    

    for idx, row in valid_df.head().iterrows():
        print(f"  {row['Compound']} -> {row['SMILES'][:50]}{'...' if len(str(row['SMILES'])) > 50 else ''}")
    
    feature_list = []
    name_list = []
    stats_list = []
    
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Generate molecular fingerprints"):
        compound_name = row['Compound']
        smiles = row['SMILES']
        cid = row['CID']
        
        try:
            fp = smiles_to_morgan(smiles)
            
            if fp is not None:
                arr = np.zeros((1024,), dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                feature_list.append(arr)
                name_list.append(compound_name)
                
                stats_list.append({
                    'Compound': compound_name,
                    'CID': cid,
                    'SMILES': smiles,
                    'Status': 'Success',
                    'Error': ''
                })
                
            else:
                print(f"  skip {compound_name}")
                stats_list.append({
                    'Compound': compound_name,
                    'CID': cid,
                    'SMILES': smiles,
                    'Status': 'Failed',
                    'Error': 'Cannot generate fingerprint'
                })
                
        except Exception as e:
            stats_list.append({
                'Compound': compound_name,
                'CID': cid,
                'SMILES': smiles,
                'Status': 'Error',
                'Error': str(e)
            })
    

    if len(feature_list) > 0:
        
        feature_matrix = np.array(feature_list)
        np.save(output_npy, feature_matrix)
        
        with open(output_names_txt, 'w', encoding='utf-8') as f:
            for name in name_list:
                f.write(name + '\n')
        
        if output_stats_csv:
            stats_df = pd.DataFrame(stats_list)
            stats_df.to_csv(output_stats_csv, index=False)

            success_count = len(stats_df[stats_df['Status'] == 'Success'])
            failed_count = len(stats_df[stats_df['Status'] == 'Failed'])
            error_count = len(stats_df[stats_df['Status'] == 'Error'])
            
            print(f"  success rate: {success_count/len(stats_df)*100:.1f}%")
        
    else:
        print("Please verify whether the input data is in the correct SMILES format.")

if __name__ == "__main__":
    build_metabolite_features_from_csv(
        input_csv="compound_smiles_use.csv",  
        output_npy="./metabolite_features.npy",      
        output_names_txt="./metabolite_names.txt",  
        output_stats_csv="./metabolite_processing_stats.csv" 
    )