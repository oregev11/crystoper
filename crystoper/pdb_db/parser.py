from glob import glob
from os.path import join
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ..utils.data import load_json, write_to_csv_in_batches
from .. import config

#features to extract from each Entry json file
ENTRY_FEATURES_NODES_PATHS = {'struct_method': ('exptl', 0, 'method'),
                              'crystal_method': ('exptl_crystal_grow', 0, 'method'),
                              'ph': ('exptl_crystal_grow', 0, 'p_h'),
                              'temp': ('exptl_crystal_grow', 0, 'temp'),
                              'pdbx_details': ('exptl_crystal_grow', 0, 'pdbx_details'),
                              'deposit_date': ['rcsb_accession_info', 'deposit_date'],
                              'revision_date': ['rcsb_accession_info', 'revision_date']}

#features to extract from each Poly Entity json file
POLY_ENTITY_FEATURES_NODES_PATHS = {'sequence': ['entity_poly', 'pdbx_seq_one_letter_code_can'],
                                    'poly_type': ['entity_poly', 'rcsb_entity_polymer_type']}

def pdb_json_parser(entries_folder,
                    poly_entities_folder):
    
    if not entries_folder:
        raise ValueError('entries_folder MUST be passed')
    
    if not poly_entities_folder:
        raise ValueError('poly entities folder MUST be passed')
    
    df_entries = parse_entries(entries_folder)
    df_poly_entities = parse_poly_entities(poly_entities_folder)
    
    df = df_poly_entities.merge(df_entries, on='pdb_id')
    
    output_path = config.parsed_pdbs_path
    df.to_csv(output_path, index=False)
    
    print(f"Parsed PDB data was saved to {output_path}")
     
def parse_entries(folder):
    
    query = join(folder, '**', '*.json')
    
    data = []
    
    print('Parsing Entries...')
    
    for json_path in tqdm(glob(query, recursive=True)):
        
        pdb_id = Path(json_path).stem
        
        entry = load_json(json_path)
        
        data.append([pdb_id] + [get_item(entry, nodes_path) for nodes_path in ENTRY_FEATURES_NODES_PATHS.values()])
        
    return pd.DataFrame(data, columns=['pdb_id'] + list(ENTRY_FEATURES_NODES_PATHS.keys()))


def parse_poly_entities(folder):
    
    query = join(folder, '**', '*.json')
    
    data = []
    
    print('Parsing Poly Entities...')
    
    for json_path in tqdm(glob(query, recursive=True)):
        
        #poly entity filename format is <PDB-ID>-<POLY-ENTITY-INDEX>.json 
        pdb_id = Path(json_path).stem.split('-')[0]
        pe_index = Path(json_path).stem.split('-')[1]
        
        pe = load_json(json_path)
        
        data.append([pdb_id, pe_index] + [get_item(pe, nodes_path) for nodes_path in POLY_ENTITY_FEATURES_NODES_PATHS.values()])
        
    return pd.DataFrame(data, columns=['pdb_id', 'pe_index'] + list(POLY_ENTITY_FEATURES_NODES_PATHS.keys()))


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