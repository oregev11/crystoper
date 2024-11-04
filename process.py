"""
preprocess.py
Pre-process the pdb data
"""
import argparse
from crystoper import config
from crystoper.processor import preprocess_pdb_data
from crystoper.utils.general import vprint

from crystoper.trainer import train_test_val_toy_split


def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-i', '--input-path', type=str, default=config.parsed_data_path,
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
    parser.add_argument('-mnl', '--minimum-details-length', type=int, default=5,
                        help='minimum length of pdbx details (instances with longer pdbx_details will be filtered out)')
    parser.add_argument('-mxl', '--maximum-details-length', type=int,  default=250,
                        help='filter out entries with empty "pdbx_details" feature')
    
    parser.add_argument('-pph', '--parse-ph',  default=True,
                        help='Parse missing pH values from "pdbx_details" string')
    parser.add_argument('-pt', '--parse-temperature',  default=True,
                        help='Parse missing temperature values from "pdbx_details" string')
    
    parser.add_argument('-ts', '--test-size',  default=0.1,
                        help='train size (should be in the range (0,1))')
    parser.add_argument('-vs', '--validation-size',  default=0.045,
                        help='validation size (should be in the range (0,1))')
    
    

        
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    df = preprocess_pdb_data(**vars(args))
    
    df.to_csv(config.processed_data_path, index=False)
    vprint(f'Processed data was saved to {config.processed_data_path}')
    
    train_df, test_df, val_df, toy_df = train_test_val_toy_split(df, args.test_size, args.validation_size)
    
    train_df.to_csv(config.train_path, index=False)
    vprint(f'Train data saved to {config.train_path}')
    
    test_df.to_csv(config.test_path, index=False)
    vprint(f'Test data saved to {config.test_path}')
    
    val_df.to_csv(config.val_path, index=False)
    vprint(f'Val data saved to {config.val_path}')
    
    toy_df.to_csv(config.toy_path, index=False)
    vprint(f'Toy data saved to {config.train_path}')
    

if __name__ == "__main__":
    main()
