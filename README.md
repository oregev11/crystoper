# crystoper - Crystallization Conditions Predictor  
crystoper is a tool for predicting diffracting crystallization conditions for proteins based on their sequence.
Current version (0.1) only implements:
1. downloading entries and poly-entities data as *.json files from the PDB.
2. Parsing and processing the data into csv file.
3. Training an ESM and BART based model to predict crystallization conditions based on the PDB data.
4. Using our trained model (called ESMCcomplex) to predict crystallization conditions for sequences.

NOTE: Our model, ESMCcomplex Was trained on ~113K instances for 20 epochs and did not converge. A better architecture of the model  may very likely result in better performance. We hope to provide such a model in the near future. Nevertheless, this repo is a great basis for (1) acquiring the PDB data and (2) training a better model. ENJOY!

# Background
Crystallography is the golden standard for accurately determining a protein 3D structure.
The process has 4 main steps:
1. purification - purifying the protein in high quantity.
2. crystallization - producing crystals of the protein.
3. X-RAY diffraction - omitting x-ray radiation on the crystals and monitor the diffraction.
4. solving the proteins structure based onf the diffraction pattern.

This work focuses on step 2 - * finding the proper crystallization conditions*

# Datasets
Data was taken from PDB (Protein Data Bank, https://www.rcsb.org/).
The updated list of all instances in the PDB can be downloaded from https://data.rcsb.org/rest/v1/holdings/current/entry_ids

# USAGE

1.(Optional) Fetch the updated list of Entries and Polymer Entities from PDB and download them (this takes a few days) using `$ python download.py -f -fp -de -dpe`.
data will be saved in data/pdb_data.
2. Parse the relevant data from Entries (experiment data) and Polymer Entities (sequence of each chain in each structure) into a csv using `$ python parse.py`.
data will be saved to `data/pdb_data/parsed_data.csv`.
3. Preprocess data and pack it a csv files using `$ python process.py`. (full data will be saved to data/pdb_data/processed_data.csv. The train, test, val & toy data will be saved in `data/*.csv`)
4. Create a BART-coded representation of the crystallization conditions ('pdbx_details' feature in the PDB data) of each instance using `$ python vectorize.py -d`. Due to the large vectors size, the pickled data will be saved as shard files under `vectors/details/**/bart_vectors_*.pkl`.
5. Load a fresh ESMCcomplex model and train it on the data. 

7. (Optional) - If you wish to train your own model on the sequences embedded space by extracting ESM last hidden layer you can use  `$ python vectorize.py -s`.






