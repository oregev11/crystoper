import argparse
from crystoper import config
from crystoper.pdb_db.data_downloader import download_pdbs_data



def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse get_data.py arguments")
    
    parser.add_argument('-i', '--ids-path', type=str, default=config.pdb_ids_path, help='Path to json with pdb ids list')
    parser.add_argument('-o', '--output-path', type=str, default=config.downloaded_data_path, help='Path to save downloaded data in.')
    parser.add_argument('-r', '--reset', action='store_true', help='reset output file. if not passed data will be appended to the output file.')
    parser.add_argument('-f', '--fetch-ids', action='store_true', help='Fetch and overwrite the pdbs ids list json file using PDB search query')
    parser.add_argument('-fp', '--fetch-pe', action='store_true', help='Fetch and overwrite the polymer-entities list json file of the using PDB search query')
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    download_pdbs_data(**vars(args))

if __name__ == "__main__":
    main()

