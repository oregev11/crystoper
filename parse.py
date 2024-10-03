"""
parse.py
Parse the PDB downloaded data into a csv file.
"""
import argparse
from crystoper import config
from crystoper.pdb_db.data_downloader import download_pdbs_data



def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-e', '--entries-folder', type=str, default=config.pdb_entries_path,
                        help='Path to parent folder of all PDB entries json files.')
    parser.add_argument('-pe', '--poly-entities-folder', type=str, default=config.pdb_polymer_entities_path,
                        help='Path to parent folder of all PDB polymer entities json files.')
    
    parser.add_argument('-f', '--features', nargs='+', type=str,
                        help='List of features to extract from the PDB entries and polymer entities.')
    
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    parser(**vars(args))

if __name__ == "__main__":
    main()

