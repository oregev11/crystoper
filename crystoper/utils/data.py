import os
import json 
import pandas as pd

ENTITY_SPLIT_CHAR = '_'

def make_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def dump_json(obj, path):
    
    with open(path, 'w') as f:
        return json.dump(obj, f)
    
    
def filter_pdb_polymer_ids(polymer_ids, relevant_ids):
    """
    Use pandas for fast filtering of large data.

    Args:
        polymer_ids (list): lister of polymer ids like ['1AON-1', '1AON-2', '5CQ2-1',....]
        relevant_ids (list): relevant ids to filter by. like ['1AON1', '5CQ2',...]
    
    """
    
    df = pd.DataFrame([id.split(ENTITY_SPLIT_CHAR) for id in polymer_ids], columns=['id', 'entity'])
    filter_df = pd.DataFrame({'id': relevant_ids})
    
    df =  df.merge(filter_df, how='inner')
    
    return list(df['id'] + '-' + df['entity'])
    