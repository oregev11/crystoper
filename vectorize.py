"""
vectorize.py
Convert sequences to vectors and pack data in a *.pkl file.
"""
import argparse
import torch
from crystoper import config
from crystoper.utils.general import vprint
from crystoper.vectorizer import SequencesVectorizer
from crystoper.utils.data import Data



def parse_args():
    
    parser = argparse.ArgumentParser(description="Covert sequences into vectors")
    
    parser.add_argument('-i', '--data-path', type=str, default=config.pdb_entries_path,
                        help='Path to csv with a "sequence""')
    #sequence-related args
    parser.add_argument('-s', '--extract-sequences-vectors', action='store_true',
                        help='flag for extracting the sequences embedded vectors')
    parser.add_argument('-sm', '--sequences-model', type=str, default='facebook/esm2_t33_650M_UR50D',
                        help='checkpoint to use for extracting the sequences embedded vectors')
    parser.add_argument('-sb', '--sequences-batch-size', type=int, default=16,
                        help='batch size for extracting the sequences embedded vectors')
    parser.add_argument('-sp', '--sequences-pooling', type=str, default='average',
                        help='pooling method for extracting the sequences embedded vectors')
    
    #pdbx_details-related args
    parser.add_argument('-d', '--extract-details-vectors', action='store_true',
                        help='flag for extracting the pdbx_details embedded vectors')
    parser.add_argument('-dm', '--details-model', type=str, default='gpt',
                        help='checkpoint to use for extracting the pdbx details embedded vectors')
    parser.add_argument('-db', '--details-batch-size', type=int, default=16,
                        help='batch size for extracting the pdbx details embedded vectors')
    parser.add_argument('-dp', '--details-pooling', type=str, default='average',
                        help='pooling method for extracting the pdbx details embedded vectors')
    
    parser.add_argument('--cpu', action='store_true',
                        help='Force cpu usage')
    
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    if args.extract_sequences_vectors:

        data = Data(args.data_path)
        df = torch.load(args.data_path)
        
        vec = SequencesVectorizer(model=args.sequences_model,
                                  batch_size = args.sequences_batch_size,
                                  pooling=args.sequences_pooling)

        data = vec(data)
        
        vprint(f'Sequnces embbeded vectors extraciotn using {args.sequences_model} is done!')
        vprint('Going over to pdbx_details vectors extraction...')
    
    if args.extract_details_vectors:

        data = torch.load(args.data_path)
        
        vec = SequencesVectorizer(model=args.sequences_model,
                                  batch_size = args.sequences_batch_size,
                                  pooling=args.sequences_pooling)

        data = vec(data)
        
        vprint(f'Sequnces embbeded vectors extraciotn using {args.sequences_model} is done!')
        vprint('Going over to pdbx_details vectors extraction...')

    
    torch.save(data, args.data_path)


    

if __name__ == "__main__":
    main()

