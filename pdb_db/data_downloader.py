from time import sleep
from os.path import exists
import argparse
import json
import requests
from Bio import SeqIO
from io import StringIO
import pandas as pd
from tqdm import tqdm


PDB_REST_API_DOWNLOAD_URL = "https://files.rcsb.org/download/"
ENTRY_REST_API_DOWNLOAD_URL = "https://data.rcsb.org/rest/v1/core/entry/"

OK_STATUS = 200
BASE_COLUMNS = ['id', 'status']
BATCH_SIZE = 2

def load_pdb_ids_list(path):
    "Load pdb ids json file as list. The file structure must be a list. for example: ['100D','1BQK','1DP5'...]"
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    assert type(data) == list, "The structure of the input json file must be a single list"
    
    return data




            

def download_pdbs_data(ids_path,
                       output_path='test_output.csv',
                       features=['method', 'grow', 'ph', 'temp', 'details', 'sequence'],
                       type='xray',
                       reset_download=True):
    
    ids = load_pdb_ids_list(ids_path)
    errors = []
    
    #reset file
    if exists(output_path):
        if reset_download:
            #reset files with columns for all features
            with open(output_path, 'w') as f:
                columns = BASE_COLUMNS +  features
                f.write(','.join(columns) + '\n') 
        
        else:
            #filter ids that were not already downloaded
            ids = [id for id in ids \
                    if id not in pd.read_csv(output_path)['id'].to_list()]
    
    buffer = ''
    
    for i, pdb_id in tqdm(enumerate(ids), total=len(ids)):
        
        buffer += f'{pdb_id}'
        data = dict()
        
        #get entry
        entry_response = download_entry_object(pdb_id)
        
        if entry_response.status_code == OK_STATUS:
            method = get_experimental_method_from_entry_object(entry_response)
            data['method'] = method
        else:
            data['method'] = "could not reach record"
            
        ph, temp, details = get_crystal_grow_cond_from_entry_object(entry_response)
        
        data['ph'] = ph
        data['temp'] = temp
        data['details'] = details
                
        #get pdb as text
        pdb_response = download_pdb_object(pdb_id)
        
        data['status'] = pdb_response.status_code
        buffer += f",{data['status']}"
        
        if pdb_response.status_code == OK_STATUS:
            
            
            pdb_text = pdb_response.text
            
            if "sequence" in features:
                data['sequence'] = get_sequence_from_pdb_text(pdb_text)
                        
        
            #write all acquired data to buffer according to features input order (which is the columns order in  the output csv)
            for feature in features:
                buffer += f",{data[feature]}"
                
            
        
        else:
            errors.append(pdb_id)
            
            
        
        buffer += '\n'    
        
        #dump buffer to file
        if i > 0 and i % BATCH_SIZE == 0:
            
            with open(output_path, 'a') as f:
                f.write(buffer)
            
            print(f'Wrote batch {i//BATCH_SIZE} out of {len(ids)//BATCH_SIZE} to file')
            if len(errors) == 0:
                print('No errors found so far....')
            else:
                print(f"Could not download the following ids: {errors}")
                
            buffer = ''
                
        sleep(1)
    
    #dump rest of buffer to file    
    with open(output_path, 'a') as f:
                f.write(buffer)
    



def main():
    download_pdbs_data('/home/ofir/ofir_code/crystoper/pdb_db/20240927_pdb_entry_ids.json',
                       'test_output.csv')
    

if __name__ == '__main__':
    main()