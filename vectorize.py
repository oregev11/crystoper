"""
vectorize.py
Convert sequences to vectors and pack data in a *.pkl file.
"""
import argparse
from crystoper import config
from crystoper.vectorizer import SequencesVectorizer



def parse_args():
    
    parser = argparse.ArgumentParser(description="Covert sequences into vectors")
    
    parser.add_argument('-i', '--data-path', type=str, default=config.pdb_entries_path,
                        help='Path to csv with a "sequence""')
    parser.add_argument('-s', '--sequence-col', type=str, default='sequence',
                        help='Sequence column in the csv')
    
    
    
    
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    

if __name__ == "__main__":
    main()

