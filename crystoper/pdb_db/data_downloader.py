from time import sleep
from os.path import exists
import argparse
import json
import requests
from Bio import SeqIO
from io import StringIO
import pandas as pd
from tqdm import tqdm

from .rest_api_methods import *
from ..utils.data import make_dir, load_json, dump_json, filter_pdb_polymer_ids
from .. import config

from rcsbsearchapi.search import TextQuery

OK_STATUS = 200
BASE_COLUMNS = ['id', 'download_status']
DOWNLOAD_BATCH_SIZE = 50
N_tries = 3

ENTRY_REST_API_URL = 'https://data.rcsb.org/rest/v1/core/entry/'
POLYMER_ENTITY_API_URL = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'
PDB_IDS_TEXT_QUERY = 'experimental_method:x-ray'

def download_pdbs_data(ids_path,
                       output_path='test_output.csv',
                       features=['method', 'ph', 'temp', 'details', 'sequence', 'n_chains'],
                       reset=False,
                       update_ids=False,
                       update_pe=False):
    
    if update_ids:
        update_ids_list(config.pdb_ids_path)
    
    if update_pe:
        
        relevant_ids = load_json(config.pdb_ids_path)
        update_polymer_entities_list(config.pdb_polymer_entities_path, relevant_ids)
    
    ids = load_pdb_ids_list(ids_path)
    errors = []
    
    #reset file
    if exists(output_path):
        if reset:
            #reset files with columns for all features
            with open(output_path, 'w') as f:
                columns = BASE_COLUMNS +  features
                f.write(','.join(columns) + '\n') 
        
        else:
            #filter ids that were not already downloaded
            existing_ids = pd.read_csv(output_path)['id'].to_list()
            ids = [id for id in ids \
                    if id not in existing_ids]
    else:
        
        make_dir(output_path)
        
        # if file dosnt exists reset files with columns for all features
        with open(output_path, 'w') as f:
            columns = BASE_COLUMNS +  features
            f.write(','.join(columns) + '\n') 
    
    buffer = ''
    
    print("Starting PDB data downloading....")
    
    for i, pdb_id in tqdm(enumerate(ids),
                          initial=len(existing_ids),
                          total=len(ids + existing_ids)):
        
        data = None
        
        for n in range(N_tries):
            
            data = get_features_dict_for_pdb_id(pdb_id, features)
            
            #if data was downloaded
            if data:
                buffer += f'{pdb_id}'
                buffer += f",{data.get('download_status')}"
                
                #write all acquired data to buffer according to features input order (which is the columns order in  the output csv)
                for feature in features:
                    buffer += f",{data.get(feature)}"

                buffer += '\n'
                break
            
            else:
                print(f"Failed downloading data for {pdb_id}. try {n+1} out of {N_tries}")
                sleep(n+1)
        
        if not data:
            errors.append(pdb_id)
        
        #dump buffer to file
        if i > 0 and i % DOWNLOAD_BATCH_SIZE == 0:
            
            with open(output_path, 'a') as f:
                f.write(buffer)
            
            print(f'Wrote batch {i//DOWNLOAD_BATCH_SIZE} out of {len(ids)//DOWNLOAD_BATCH_SIZE} to file')
            if len(errors) == 0:
                print('No errors found so far....')
            else:
                print(f"Could not download the following ids: {errors}")
                
            buffer = ''
            
                
    
    #dump rest of buffer to file    
    with open(output_path, 'a') as f:
                f.write(buffer)
    

    print('PDB download is Done!')
    print(f'The following pdb ids could not be downloaded: {errors}')

def update_ids_list(path):
    
    query = TextQuery(PDB_IDS_TEXT_QUERY)
    
    print(f"\nFetching all pdb ids with the following filter: '{PDB_IDS_TEXT_QUERY}'. This should take few minutes...")
    
    ids = list(query('entry')) 
    
    dump_json(ids, path)
        
    print(f'{len(ids)}  where fetched from PDB and dumped to {path}')


def update_polymer_entities_list(path, relevant_ids):
    
    query = TextQuery(PDB_IDS_TEXT_QUERY)
    
    print(f"\nFetching all pdb polymer entities with the following filter: '{PDB_IDS_TEXT_QUERY}'. This should take few minutes...")
    
    polymer_ids = []
    
    for id in tqdm(query('polymer_entity'), total=300000):
        polymer_ids.append(id)
    
    print(f'{len(polymer_ids)} polymer entities id were fetched from pdb')
    
    #validate data integrity by filtering polymer entities with the pdb entry list
    filtered_polymer_ids = filter_pdb_polymer_ids(polymer_ids, relevant_ids)
    
    dump_json(filtered_polymer_ids, path)
    
    print(f'After filtration for relevant ids (x-ray) {len(filtered_polymer_ids)} polymer entities ids were dumped to {path}')
    

def load_pdb_ids_list(path):
    "Load pdb ids json file as list. The file structure must be a list. for example: ['100D','1BQK','1DP5'...]"
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    assert type(data) == list, "The structure of the input json file must be a single list"
    
    return data

def get_features_dict_for_pdb_id(pdb_id, features):
    
        data = dict()
        
        #get all data from the 'entry' PDB page
        url = ENTRY_REST_API_URL + pdb_id 
        entry_response = get_url(url)
        
        if entry_response.status_code == OK_STATUS:
            method = get_experimental_method_from_entry_object(entry_response)
            data['method'] = method
            
            ph, temp, details = get_crystal_grow_cond_from_entry_object(entry_response)
        
            data['ph'] = ph
            data['temp'] = temp
            data['details'] = details
        
        else:
            data['method'] = "N/A"
            
        
                
        #get polymer chain sequences from the PDB polymer_entity pages
        url = POLYMER_ENTITY_API_URL + pdb_id
        sequence, n_chains = get_all_sequences_from_polymer_entity(url)
        
        data['sequence'] = sequence
        data['n_chains'] = n_chains
        
        return data
        
        
def main():
    download_pdbs_data('/home/ofir/ofir_code/crystoper/pdb_db/20240927_pdb_entry_ids.json',
                       'test_output.csv')
    

if __name__ == '__main__':
    main()