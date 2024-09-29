import json 
import requests
from Bio import SeqIO
from io import StringIO

N_TRIES = 3
OK_STATUS = 200




def get_url(url):
        
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


def get_all_sequences_from_polymer_entity(root_url):
    """iterates all polymer entities and retrieve sequences until polymer id is missing (error response)
    
    ARGS:
        root_url (str): a string with the polimer entity root path. for example: 'https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/'
                        (given the above input, the function will download from 'https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/1' ,
                        'https://data.rcsb.org/rest/v1/core/polymer_entity/9F9L/2 and so on until error page is found')
                        
    RETURN:
        (str): a concat of all polymer chains for the pdb id in the root_url with ';' sep.
    
    
    """
    
    sequences = []
    i = 1
        
    while True:
        url = root_url + str(i) 
        response = get_url(url)
    
        #break on error status (when the polimer id is missing an error response is return)
        if response.status != OK_STATUS:
            return ';'.join(sequences)
        
        i += 1
