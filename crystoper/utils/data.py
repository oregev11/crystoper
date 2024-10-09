import os
import json 
import pandas as pd
from os.path import isfile, dirname
import csv
from tqdm import tqdm


ENTITY_SPLIT_CHAR = '_'
CSV_BATCH_SIZE = 1000

def make_dir(path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def dump_json(obj, path):
    
    with open(path, 'w') as f:
        return json.dump(obj, f)
    


def write_to_csv_in_batches(data_generator, headers, output_file, batch_size=CSV_BATCH_SIZE, verbose=True, tqdm_total=None):
    
    buffer = []
    
    with open(output_file, mode='w', newline='') as file:
        
        writer = csv.writer(file)
        
        writer.writerow(headers)
        
        if verbose:
            for row in tqdm(data_generator, total=tqdm_total):
                buffer.append(row)
                
                if len(buffer) >= batch_size:
                    writer.writerows(buffer)
                    buffer.clear()  
        
        if buffer:
            writer.writerows(buffer)


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
    