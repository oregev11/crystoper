import json 

def download_entry_object(pdb_id):
    
    url = ENTRY_REST_API_DOWNLOAD_URL + pdb_id 
    response = requests.get(url)
    
    return response


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
        temp = exptl_crystal_grow.get('temp', {}).get('value', 'N/A')
        details = exptl_crystal_grow.get('pdbx_details', 'N/A')
        
    return ph, temp, details
    