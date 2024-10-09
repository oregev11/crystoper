from glob import glob
from os.path import join
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ..utils.data import load_json, write_to_csv_in_batches
from .. import config


ENTRY_FEATURES_NODES_PATHS = {'struct_method': ('exptl', 0, 'method'),
                              'crystal_method': ('exptl_crystal_grow', 0, 'method'),
                              'ph': ('exptl_crystal_grow', 0, 'p_h'),
                              'temp': ('exptl_crystal_grow', 0, 'temp'),
                              'pdbx_details': ('exptl_crystal_grow', 0, 'pdbx_details')}

def pdb_json_parser(entries_folder,
                    poly_entities_folder):
    
    if not entries_folder:
        raise ValueError('entries_folder MUST be passed')
    
    if not poly_entities_folder:
        raise ValueError('poly entities folder MUST be passed')
    
    df_entries = parse_entries(entries_folder)

     
def parse_entries(entries_folder):
    
    query = join(entries_folder, '**', '*.json')
    
    data = []
    
    print('Parsing Entries...')
    
    for entry_path in tqdm(glob(query, recursive=True)):
        
        pdb_id = Path(entry_path).stem
        
        entry = load_json(entry_path)
        
        data.append([pdb_id] + [get_item(entry, nodes_path) for nodes_path in ENTRY_FEATURES_NODES_PATHS.values()])
        
    return pd.DataFrame(data, headers=['pdb_id'] + list(ENTRY_FEATURES_NODES_PATHS.keys()))


def get_item(jsn, nodes_path):
    """recursive method to get item from a json like object (nested dicts and lists)

    Args:
        jsn (obj): a dict-like (or json like) nested object
        nodes_path (list): a list of nodes that represent the root to go when fetching the object of interest. can be either string (for dict access) or \
            int (for a list access).
            for example: given nodes_path = ['conditions', 'temp', 2] the object to fetch will be jsn.conditions.temp[2]
    """
    
    if not jsn:
        return None
    
    if len(nodes_path) == 0:
        return jsn
    
    #pop first item from tuple
    next_node = nodes_path[0] 
    nodes_path = nodes_path[1:]
    
    if type(next_node) == str:
        return get_item(jsn.get(next_node), nodes_path)
    
    if type(next_node) == int:
        return get_item(jsn[next_node], nodes_path)