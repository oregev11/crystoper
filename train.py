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
    
    parser.add_argument('-m', '--model', type=str, default='esmc-complex',
                        help='model to use (if checkpoint is passed - it will be loaded instead)')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Checkpoint for loading a pre-trained model')
    
    
    
        
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    df = pd.read_csv(args.data_path)
 
    
   
        
    
if __name__ == "__main__":
    main()
