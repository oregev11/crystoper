import argparse
from crystoper import config

def main():
    parser = argparse.ArgumentParser(description="A script template using argparse")
    
    parser.add_argument('-i', '--ids-path', type=str, default=, help='path to json with pdb ids list')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    
    args = parser.parse_args()

if __name__ == "__main__":
    main()
