"""
preprocess.py
Pre-process the pdb data
"""
import argparse
from crystoper import config
from crystoper.preprocessor import preprocess_pdb_data



def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-i', '--input-path', type=str, default=config.parsed_pdbs_path,
                        help='Path to the parsed data csv file')
    parser.add_argument('-o', '--output-path', type=str, default=config.processed_data_path,
                        help='output path (csv)')
    parser.add_argument('-fnp', '--filter-non-proteins', type=str, default=True,
                        help='filter out entries that include poly entities (chains) that are not proteins. (All the entry will be removed)')
    parser.add_argument('-c', '--chains_per_entry', type=str, default=[1], nargs='+',
                        help='maximal number of chains (polymer entities) per PDB id. PDB ids (entries) with larger number of chains will be removed')

        
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    preprocess_pdb_data(**vars(args))

if __name__ == "__main__":
    main()
