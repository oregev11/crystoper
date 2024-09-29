
import json 
import requests
from Bio import SeqIO
from io import StringIO

PDB_REST_API_DOWNLOAD_URL = "https://files.rcsb.org/download/"
N_TRIES = 3
def download_pdb_object(pdb_id):
    
    url = PDB_REST_API_DOWNLOAD_URL + pdb_id + '.pdb'
    
    for i in range(N_TRIES):
        try:
            return requests.get(url)
        except:
            print(f"Could form connection with PDB while fetching {url}")
            print(f'try {i+1} out of {N_TRIES}')
            sleep(3)


def get_sequence_from_pdb_text(pdb_text):
    "Get the FIRST sequence from a PDB file"
    
    pdb_io = StringIO(pdb_text)
    sequence = ''

    # Parse the PDB file and extract the sequence using SeqIO
    for record in SeqIO.parse(pdb_io, "pdb-seqres"):
        sequence = record.seq
        return sequence