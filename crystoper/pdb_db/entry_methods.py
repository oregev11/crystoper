import json 
import requests
from Bio import SeqIO
from io import StringIO
ENTRY_REST_API_DOWNLOAD_URL = "https://data.rcsb.org/rest/v1/core/entry/"
N_TRIES = 3

def download_entry_object(pdb_id):
    
    url = ENTRY_REST_API_DOWNLOAD_URL + pdb_id 
    
    for i in range(N_TRIES):
        try:
            return requests.get(url)
        except:
            print(f"Could form connection with PDB while fetching {url}")
            print(f'try {i+1} out of {N_TRIES}')
            sleep(3)


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
    