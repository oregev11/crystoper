# crystoper - Crystallization Conditions Predictor  
crystoper is a tool for predicting diffracting crystalization conditions for proteins based on their sequence.
Current version (0.1) only implements data fetching and processing.

# Datasets
Data was taken from PDB (Protein Data Bank, https://www.rcsb.org/).
The updated list of all instances in the PDB can be downloaded from https://data.rcsb.org/rest/v1/holdings/current/entry_ids

# USAGE

1. Fetch the updated list of Entries and Polymer Entities from PDB and download them (this takes a few days) using `$ python download.py -f -fp -de -dpe`.
data will be saved in data/pdb_data.
2. Parse the relevant data from Entries (experiment data) and Polymer Entities (sequence of each chain in each structure) into a csv using `$ python parse.py`.
data will be saved to `data/pdb_data/parsed_data.csv`.
3. Preprocess data and pack it as pickle using `$ python process.py`. 
data will be saved to `data/pdb_data/processed_data.pkl`.
4. Extract sequences & pdbx_details vectors using `$ python vectorize.py -s -d`.
data will be added to `data/pdb_data/processed_data.pkl`.




