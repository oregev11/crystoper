import json 
import requests
from Bio import SeqIO
from io import StringIO
from time import sleep
N_TRIES = 3
OK_STATUS = 200

POLYMER_ENTITY_SEQUENCE_FIELD = 'pdbx_seq_one_letter_code'



def get_url(url):
        
    for i in range(N_TRIES):
        try:
            return requests.get(url)
        except:
            print(f"Could form connection with PDB while fetching {url}")
            print(f'try {i+1} out of {N_TRIES}')
            sleep(i+1)


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


def get_all_sequences_from_polymer_entity(root_url):
    """iterates all polymer entities and retrieve sequences until polymer id is missing (error response)
    
    ARGS:
        root_url (str): a string with the polimer entity root path. 
                        
    RETURN:
        (str): a concat of all polymer chains for the pdb id in the root_url with ';' sep.
        
    for example: given root_url='https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/'
                        The function will download data from 'https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/1' ,
                        'https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/2 and so on until error page is found - meaning no more eateries')
    
    
    """
    
    sequences = []
    i = 0
        
    while True:
        
        i += 1
        url = root_url + f'/{i}' 
        response = get_url(url)
        
        if response.status_code == OK_STATUS:
            
            data = response.json()
            
            #skip non protein
            if data['entity_poly']['rcsb_entity_polymer_type'].lower() != 'protein':
                continue
            
            else:
                sequences.append(data['entity_poly']['pdbx_seq_one_letter_code'])
            
    
        #break on error status (when the polimer id is missing an error response is return)
        if response.status_code != OK_STATUS:
            return ';'.join(sequences), i
        
        #stop condition 
        if i > 20:
            return '', 0
        
        
