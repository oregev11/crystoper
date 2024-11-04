"""
train.py
train model
"""
import argparse
from crystoper import config
from crystoper.processor import filter_by_pdbx_details_length, filter_for_single_entities
from crystoper.trainer import train_test_val_toy_split
from crystoper.utils.general import vprint



def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-i', '--data-path', type=str, default=config.processed_data_path,
                        help='output path (csv)')
    parser.add_argument('-so', '--single-chain-only', default=True,
                        help='train only on single chain proteins')
    parser.add_argument('-mnl', '--minimum-details-length', type=int, default=5,
                        help='minimum length of pdbx details (instances with longer pdbx_details will be filtered out)')
    parser.add_argument('-mxl', '--maximum-details-length',  default=True,
                        help='filter out entries with empty "pdbx_details" feature')
    
    parser.add_argument('-ts', '--test-size',  default=0.1,
                        help='train size (should be in the range (0,1))')
    parser.add_argument('-vs', '--validation-size',  default=0.045,
                        help='validation size (should be in the range (0,1))')
    
    parser.add_argument('-sd', '--save-split-data',  action='store_true',
                        help='save the splitted data (train, test, val, toy)')
    
    
    
    
    

        
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    df = pd.read_csv(args.data_path)
    
    df = filter_for_single_entities(df)
    
    df = filter_by_pdbx_details_length(df, args.minimum_details_length, args.maximum_details_length)
    
    df.pdbx_details = df.pdbx_details.str.replace('\n', ' ')
    
    train_df, test_df, val_df, toy_df = train_test_val_toy_split(df, args.test_size, args.val_size)
    
    if args.save_split_data:
        train_df.to_csv(join(DATA_ROOT, 'train_test', 'train.csv'), index=False)
        test_df.to_csv(join(DATA_ROOT, 'train_test', 'test.csv'), index=False)
        val_df.to_csv(join(DATA_ROOT, 'train_test', 'validation.csv'), index=False)
        toy_df.to_csv(join(DATA_ROOT, 'train_test', 'toy.csv'), index=False)
        
    
    vprint(f'Processed data was saved to {config.processed_data_path}')

if __name__ == "__main__":
    main()
