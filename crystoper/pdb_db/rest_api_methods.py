import json 
import requests
from Bio import SeqIO
from io import StringIO
from time import sleep
from ..utils.data import dump_json
from os.path import join

N_TRIES = 4
OK_STATUS = 200

ENTRY_REST_API_URL = 'https://data.rcsb.org/rest/v1/core/entry/'
POLYMER_ENTITY_API_URL = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'
POLYMER_ENTITY_SEQUENCE_FIELD = 'pdbx_seq_one_letter_code'



def get_url(url):
        
    for i in range(N_TRIES):
        try:
            return requests.get(url)
        except:
            print(f"Could form not connection with PDB while fetching {url}")
            print(f'try {i+1} out of {N_TRIES}')
            sleep(i+1)
            
    return None

## download full record methods

def download_entry_as_json(pdb_id, folder):
    
    url = ENTRY_REST_API_URL + pdb_id
    response = get_url(url)
    
    if response:
        path = join(folder, pdb_id) + '.json'
        dump_json(response.json(), path)
    else:
        print(f'Could not get download entry from {url}')
        
def download_poly_entity_as_json(entity_id, folder):
    
    url = POLYMER_ENTITY_API_URL + entity_id.replace('-', '/') #convert, for example 1AON-1 to 1AON/1 to fit the address format
    response = get_url(url)
    
    if response:
        path = join(folder, entity_id) + '.json'
        dump_json(response.json(), path)
    else:
        print(f'Could not get download polymer entity from {url}')

def get_experimental_method_from_entry_object(response):
    """'method' refer to the method used to determine the structure ('X-RAY DIFFRACTION', 'SOLUTION NMR') """
    
    data = response.json()
    
    # Extract the experimental method(s)
    methods = [expt.get("method", "N/A") for expt in data.get("exptl", [])]
    
    # Return the experimental method(s)
    if methods:
        return methods[0]  # Return the first method (there can be multiple)
    else:
        return "N/A"

def get_crystal_grow_cond_from_entry_object(response):
     
    data = response.json()
     
    # Initialize variables for the conditions
    ph = "N/A"
    temp = "N/A"
    details = "N/A"

    # Extract crystallization conditions from the experimental section
    if 'exptl_crystal_grow' in data and len(data['exptl_crystal_grow']) > 0:
        exptl_crystal_grow = data['exptl_crystal_grow'][0]
        
        ph = exptl_crystal_grow.get('p_h', 'N/A')
        temp = exptl_crystal_grow.get('temp', 'N/A')
        details = exptl_crystal_grow.get('pdbx_details', 'N/A')
        
        details = details.replace(',', '|')
        details = details.replace('\n', '')
    
    
    
    return ph, temp, details

def get_pdb_entry_data(pdb_id):
    """get experiment features dict based on a PDB  entry page.
    """
        
    data = dict()
    
    #get all data from the 'entry' PDB page
    url = ENTRY_REST_API_URL + pdb_id 
    entry_response = get_url(url)
    
    if entry_response:
        if entry_response.status_code == OK_STATUS:
            method = get_experimental_method_from_entry_object(entry_response)
            data['method'] = method
            
            ph, temp, details = get_crystal_grow_cond_from_entry_object(entry_response)
        
            data['ph'] = ph
            data['temp'] = temp
            data['details'] = details
        
    return data



def get_pdb_polymer_entity_data(pe_id):
        
    """add info obtained from polymer entity page to the data"""
    
    data = dict()

    #get polymer chain sequences from the PDB polymer_entity pages
    url = POLYMER_ENTITY_API_URL + pe_id.replace('-', '/')
    polymer_entity = get_url(url)
    
    if polymer_entity:
        if polymer_entity.status_code == OK_STATUS:
            
            pe_data = polymer_entity.json()
            
            data['sequence'] = pe_data.get('entity_poly').get('pdbx_seq_one_letter_code')
            data['polymer_type'] = pe_data.get('entity_poly').get('rcsb_entity_polymer_type')
            
    return data
            

# def get_all_sequences_from_polymer_entity(root_url):
#     """iterates all polymer entities and retrieve sequences until polymer id is missing (error response)
    
#     ARGS:
#         root_url (str): a string with the polimer entity root path. 
                        
#     RETURN:
#         (str): a concat of all polymer chains for the pdb id in the root_url with ';' sep.
        
#     for example: given root_url='https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/'
#                         The function will download data from 'https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/1' ,
#                         'https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/2 and so on until error page is found - meaning no more eateries')
    
    
#     """
    
#     sequences = []
#     i = 0
        
#     while True:
        
#         i += 1
#         url = root_url + f'/{i}' 
#         response = get_url(url)
        
#         if response.status_code == OK_STATUS:
            
#             data = response.json()
            
#             #skip non protein
#             if data['entity_poly']['rcsb_entity_polymer_type'].lower() != 'protein':
#                 continue
            
#             else:
#                 sequences.append(data['entity_poly']['pdbx_seq_one_letter_code'])
            
    
#         #break on error status (when the polimer id is missing an error response is return)
#         if response.status_code != OK_STATUS:
#             return ';'.join(sequences), i
        
#         #stop condition 
#         if i > 20:
#             return '', 0
        
        
