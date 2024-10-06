from time import sleep
from os.path import exists, join
from pathlib import Path
import argparse
import json
import requests
from Bio import SeqIO
from io import StringIO
import pandas as pd
from tqdm import tqdm
import glob
import os

from .rest_api_methods import *
from ..utils.data import make_dir, load_json, dump_json, filter_pdb_polymer_ids
from .. import config

from rcsbsearchapi.search import TextQuery

OK_STATUS = 200
BASE_COLUMNS = ['id', 'polymer_index'] 
DOWNLOAD_BATCH_SIZE = 10
N_tries = 3
PDB_HASH_RANGE = 1000

PDB_IDS_TEXT_QUERY = 'experimental_method:x-ray'

def download_pdbs_data(ids_path,
                       output_path='test_output.csv',
                       features=['method', 'ph', 'temp', 'details', 'sequence', 'polymer_type'],
                       reset=False,
                       fetch_entries_ids=False,
                       fetch_poly_entities_ids=False,
                       download_entries=False,
                       download_poly_entities=False):
    
    #fetch all relevant pdb ids from the PDB server (if not - they will be read from local file)
    if fetch_entries_ids:
        update_entry_ids_list(config.pdb_ids_path)
    
    #fetch all relevant polymer-entities from the PDB server (if not - they will be read from local file)
    if fetch_poly_entities_ids:
        relevant_ids = load_json(config.pdb_ids_path)
        update_polymer_entities_list(config.pdb_polymer_entities_path, relevant_ids)
    
    #download entires json files
    if download_entries:
        
        root_path = config.pdb_entries_path
        make_dir(root_path)
        
        entry_ids = load_json(config.pdb_entries_list_path)
        
        print('fetching all entries already downloaded...')
        existing_entry_ids = list_file_stems(root_path)
        
        print('{}')
        filtered_entry_ids = [id for id in entry_ids if not id in existing_entry_ids]
        
        print('\nStarting pdb entries download. this might take a few days... ¯\_(ツ)_/¯\nbut you can stop and resume automatically from last checkpoint')
        
        for i, entry_id in tqdm(enumerate(filtered_entry_ids), initial=len(existing_entry_ids),
                                                total=len(entry_ids)):
            
            #create a subdir (to avoid too much files in same folder)
            path = join(root_path, convert_id_to_subfolder(entry_id))
            make_dir(path)
            
            download_entry_as_json(entry_id, path)
    
    if download_poly_entities:
        
        root_path = config.pdb_polymer_entities_path
        make_dir(root_path)
        
        #filter for un-downloaded entities
        
        entities_ids = load_json(config.pdb_polymer_entities_list_path)
        existing_poly_entities_ids = list_file_stems(config.pdb_polymer_entities_path)
        filtered_poly_entity_ids = [id for id in entities_ids if id not in existing_poly_entities_ids]
        
        print('\nStarting pdb polymer entities download. this might take a few days... ¯\_(ツ)_/¯\nbut you can stop and resume automatically from last checkpoint')
        
        for poly_entity_id in tqdm(filtered_poly_entity_ids, initial=len(existing_poly_entities_ids),
                                                        total=len(entities_ids)):
            
            #create a subdir (to avoid too much files in same folder)
            path = join(root_path, convert_id_to_subfolder(poly_entity_id))
            make_dir(path)
            download_poly_entity_as_json(poly_entity_id, path)
        

    print('PDB data download is Done!')

def update_entry_ids_list(path):
    """update entry ids list with the latest polymer entities ids from the PDB"""
    
    query = TextQuery(PDB_IDS_TEXT_QUERY)
    
    print(f"\nFetching all pdb ids with the following filter: '{PDB_IDS_TEXT_QUERY}'. This should take few minutes...")
    
    ids = list(query('entry')) 
    
    dump_json(ids, path)
        
    print(f'{len(ids)}  where fetched from PDB and dumped to {path}')


def update_polymer_entities_list(path, relevant_ids):
    """update polimer entities list with the latest polymer entities ids from the PDB"""
    
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
    
def list_file_stems(path):
 
    files = list(glob.iglob(path + '/**/*.*', recursive=True))
 
    return {Path(file).stem for file in files}

def convert_id_to_subfolder(pdb_id):
    """convert pdb_id to a unique subfolder name by getting the two middle chars of the pdb_id"""
    return pdb_id[1:3]

# def get_already_downloaded_pe_ids():
    
#     df = pd.read_csv(config.downloaded_data_path)
#     already_downloaded = list(df['id'] + '-' + df['polymer_index'].astype(str))
#     already_downloaded = set(already_downloaded)
    
#     return already_downloaded


        
        
def main():
    download_pdbs_data('/home/ofir/ofir_code/crystoper/pdb_db/20240927_pdb_entry_ids.json',
                       'test_output.csv')
    

if __name__ == '__main__':
    main()