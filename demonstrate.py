"""
train.py
train model
"""
import argparse
from os.path import join
from pathlib import Path
import re 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import EsmTokenizer, BartTokenizer, BartModel, BartForConditionalGeneration


from crystoper import config
from crystoper.processor import filter_by_pdbx_details_length, filter_for_single_entities
from crystoper.utils.general import vprint, make_parent_dirs
from crystoper.esmc_models import ESMCcomplex
from crystoper.trainer import ESMCTrainer, seq2sent

def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-m', '--model', type=str, default='esmc-complex',
                        help='model to use (if checkpoint is passed - it will be loaded instead)')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Checkpoint for loading a pre-trained model')
    parser.add_argument('-d', '--data_path', type=str, default=config.toy_path,
                        help='csv to use for data input. default is using the toy data')
    parser.add_argument('-x', '--x-column', type=str, default='sequence',
                        help='Column to use for input (sequences)')
    parser.add_argument('-y', '--y-column', type=str, default='pdbx_details',
                        help='Column to use as true labels (used for display only)')
    parser.add_argument('--device', default='cpu',
                        help='device to use')
    
    
    
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    device = args.device
    
    if args.checkpoint:
        esm_model = torch.load(args.checkpoint)
        vprint(f"loaded previous model from checkpoint {args.checkpoint}")
       
    elif args.model == 'esmc-complex':
        esm_model = ESMCcomplex()
        vprint(f"A fresh {args.model} has been created!")
    else:
        raise ValueError('Model cannot be resolved')
    
    esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")    
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')     
    

    data = pd.read_csv(args.data_path)
    X = data[args.x_column]
    Y = data[args.y_column]
    
    bart_model.to(device)
    esm_model.to(device)

        
    for x, y_true in zip(X,Y):
        pred = seq2sent(x, esm_model, esm_tokenizer, bart_model, bart_tokenizer, ac=True)
        
        print(f'True sentence: {y_true}')
        print(f'Pred sentence: {pred}')
        print('\n\n')
        
        
    
if __name__ == "__main__":
    main()