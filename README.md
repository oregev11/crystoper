# crystoper - Crystallization Conditions Predictor  

# Datasets
Data was taken from PDB (Protein Data Bank, https://www.rcsb.org/).
The updated list of all instances in the PDB can be downloaded from https://data.rcsb.org/rest/v1/holdings/current/entry_ids

# Requirements
- Biopython
- requests

# USAGE

1. fetch the current Entries and Polymer Entities from PDB and download them:
`python download.py -f -fp -de -dpe`
data will be saved in data/pdb_data.
2. Parse the relevant data from Entries (experiment data) and Polymer Entities (sequence of each chain in each structure) into a csv:
`python parse.py `
