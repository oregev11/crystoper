import argparse
from crystoper import config



def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse get_data.py arguments")
    
    parser.add_argument('-i', '--ids-path', type=str, default=config.pdb_ids_path, help='Path to json with pdb ids list')
    parser.add_argument('-o', '--output', type=str, default=config.downloaded_data_path, help='Path to save downloaded data in.')
    parser.add_argument('-s', '--seq', type=str, action='store_true', help='Boolean flag for downloading the sequence of each id.')
    parser.add_argument('-c', '--conditions', type=str, action='store_true', help='Boolean flag for downloading the the crystallization conditions\
        for each id.')
    
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    

if __name__ == "__main__":
    main()
