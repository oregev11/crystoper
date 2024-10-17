"""
preprocess.py
Pre-process the pdb data
"""
import argparse
from crystoper import config
from crystoper.preprocessor import process_pdb_data



def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-i', '--input-path', type=str, default=config.parsed_pdbs_path,
                        help='Path to the parsed data csv file')
    parser.add_argument('-o', '--output-path', type=str, default=config.processed_data_path,
                        help='output path (csv)')
    parser.add_argument('-fnp', '--filter-non-proteins', default=True,
                        help='filter out entries that include poly entities (chains) that are not proteins. (All the entry will be removed)')
    parser.add_argument('-c', '--chains-per-entry', type=int, default=[1], nargs='+',
                        help='maximal number of chains (polymer entities) per PDB id. PDB ids (entries)\
                            with larger number of chains will be removed. default is [1]. if [0] no filtration will be done')
    parser.add_argument('-fed', '--filter-empty-details',  default=True,
                        help='filter out entries with empty "pdbx_details" feature')
    
    parser.add_argument('-pph', '--parse-ph',  default=True,
                        help='Parse missing pH values from "pdbx_details" string')
    parser.add_argument('-pt', '--parse-temperature',  default=True,
                        help='Parse missing temperature values from "pdbx_details" string')
    
    

        
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    process_pdb_data(**vars(args))

if __name__ == "__main__":
    main()
