from glob import glob
from os.path import join
import pandas as pd
from tqdm import tqdm
from ..utils.data import load_json, write_to_csv_in_batches
from .. import config


ENTRY_FEATURES_NODES_PATHS = {'struct_method': ['exptl', 0, 'method'],
                              'crystal_method': ['exptl_crystal_grow', 0, 'method'],
                              'ph': ['exptl_crystal_grow', 0, 'p_h'],
                              'temp': ['exptl_crystal_grow', 0, 'temp'],
                              'pdbx_details': ['exptl_crystal_grow', 0, 'pdbx_details']}

def pdb_json_parser(entries_folder,
                    poly_entities_folder):
    
    if entries_folder:
        parse_entries(entries_folder)
        

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
    
    next_node = nodes_path.pop(0) #pop first item
    
    if type(next_node) == str:
        return get_item(jsn.get(next_node), nodes_path)
    
    if type(next_node) == int:
        return get_item(jsn[next_node], nodes_path)