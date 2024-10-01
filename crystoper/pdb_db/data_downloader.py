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
        existing_entry_ids = list_file_stems(root_path)
        
        filtered_entry_ids = [id for id in entry_ids if id not in set(existing_entry_ids)]
        
        print('\nStarting pdb entries download. this might take a few days  ¯\_(ツ)_/¯\
            \nBut! if you stop and rerun - it will automatically continue from last checkpoint...')
               
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
        filtered_poly_entity_ids = [id for id in entities_ids if id not in set(existing_poly_entities_ids)]
        
        print('\nStarting pdb polymer entities download. this might take a few days, but you can stop and resume  ¯\_(ツ)_/¯  \
            \nBut! if you stop and rerun - it will automatically continue from last checkpoint...')
        
        for poly_entity_id in tqdm(filtered_poly_entity_ids, initial=len(existing_poly_entities_ids),
                                                        total=len(entities_ids)):
            
            #create a subdir (to avoid too much files in same folder)
            path = join(root_path, convert_id_to_subfolder(poly_entity_id))
            make_dir(path)
            download_poly_entity_as_json(poly_entity_id, path)
        
            
        
    # errors = []
    
    # make_dir(output_path)
    
    # #reset output file
    # if not exists(output_path) or reset:
        
    #         #reset files with columns for all features
    #         with open(output_path, 'w') as f:
    #             columns = BASE_COLUMNS +  features
    #             f.write(','.join(columns) + '\n') 
        
        
    # buffer = ''
    
    # print("Starting PDB data downloading....")
    
    # #get the full list of polymer entities ids 
    # pe_ids = load_json(config.pdb_polymer_entities_path)
    
    # #filter-out polymer entities for which data was already downloaded
    # already_downloaded = get_already_downloaded_pe_ids()
    # pe_ids = [pe_id for pe_id in pe_ids if pe_id not in already_downloaded]
    
    # #iterate all pe_ids and download data
    # for i, pe_id in tqdm(enumerate(pe_ids),
    #                      initial=len(already_downloaded),
    #                      total=len(pe_ids) - len(already_downloaded)):
        
    #     entry_data = None
    #     polymer_entity_data = None
        
    #     #pe_id has '1AON-1' format.
    #     pdb_id, polymer_index = pe_id.split('-')
        
    #     #gen all relevant data from the PDB server
    #     entry_data = get_pdb_entry_data(pdb_id)
    #     polymer_entity_data = get_pdb_polymer_entity_data(pe_id)
                        
    #     if not entry_data:
    #         print(f'Could not get {pdb_id} entry data')  
    #         errors.append(pdb_id)  
    #         continue            
        
    #     if not polymer_entity_data:
    #         print(f'Could not get {pe_id} polymer entity data data')     
    #         errors.append(pe_id)           
    #         continue
        
    #     #merge data dictionaries
    #     data = entry_data
    #     data.update(polymer_entity_data)
                
    #     #write data to buffer
    #     buffer += f'{pdb_id}'
    #     buffer += f',{polymer_index}'
    #     for feature in features:
    #         buffer += f",{data.get(feature)}"

    #     buffer += '\n'
        
    #     #dump buffer to file
    #     if i > 0 and i % DOWNLOAD_BATCH_SIZE == 0:
            
    #         with open(output_path, 'a') as f:
    #             f.write(buffer)
            
    #         print(f'Wrote batch {i//DOWNLOAD_BATCH_SIZE} out of {len(pe_ids)//DOWNLOAD_BATCH_SIZE} to file')
    #         if len(errors) == 0:
    #             print('No errors found so far....')
    #         else:
    #             print(f"Could not download the following ids: {errors}")
                
    #         buffer = ''
            
                
    
    # #dump rest of buffer to file    
    # with open(output_path, 'a') as f:
    #             f.write(buffer)
    

    print('PDB download is Done!')
    print(f'The following pdb ids could not be downloaded: {errors}')

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
 
    return [Path(file).stem for file in files]

def convert_id_to_subfolder(pdb_id):
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