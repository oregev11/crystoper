"""
parse.py
Parse the PDB downloaded data into a csv file.
"""
import argparse
from crystoper import config
from crystoper.pdb_db.parser import pdb_json_parser

def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-e', '--entries-folder', type=str, default=config.pdb_entries_path,
                        help='Path to parent folder of all PDB entries json files.')
    parser.add_argument('-pe', '--poly-entities-folder', type=str, default=config.pdb_polymer_entities_path,
                        help='Path to parent folder of all PDB polymer entities json files.')
        
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    pdb_json_parser(**vars(args))
    
if __name__ == "__main__":
    main()

