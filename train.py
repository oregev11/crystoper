"""
train.py
train model
"""
import argparse
from crystoper import config
from crystoper.processor import filter_by_pdbx_details_length, filter_for_single_entities
from crystoper.utils.general import vprint

import pandas as pd



def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-i', '--data-path', type=str, default=config.processed_data_path,
                        help='output path (csv)')
   
   
    parser.add_argument('-d', '--dont-save-data',  action='store_true',
                        help='dont save the splitted data (train, test, val, toy)')
    
    
    
    
    
    

        
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    df = pd.read_csv(args.data_path)
    
    df = filter_for_single_entities(df)
    
    df = filter_by_pdbx_details_length(df, args.minimum_details_length, args.maximum_details_length)
    
    df.pdbx_details = df.pdbx_details.str.replace('\n', ' ')
    
   
        
    
if __name__ == "__main__":
    main()
