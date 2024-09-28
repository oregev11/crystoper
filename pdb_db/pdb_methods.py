def download_pdb_object(pdb_id):
    
    url = PDB_REST_API_DOWNLOAD_URL + pdb_id + '.pdb'
    response = requests.get(url)
    
    return response


def get_sequence_from_pdb_text(pdb_text):
    "Get the FIRST sequence from a PDB file"
    
    pdb_io = StringIO(pdb_text)
    sequence = ''

    # Parse the PDB file and extract the sequence using SeqIO
    for record in SeqIO.parse(pdb_io, "pdb-seqres"):
        sequence = record.seq
        return sequence