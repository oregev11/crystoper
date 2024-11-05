"""
train.py
train model
"""
import argparse
from os.path import join
from pathlib import Path

import torch.nn as nn
import torch.optim as optim


from crystoper import config
from crystoper.processor import filter_by_pdbx_details_length, filter_for_single_entities
from crystoper.utils.general import vprint
from crystoper.esmc_models import ESMCcomplex
from crystoper.trainer import ESMCTrainer


def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse PDB 'entries' and 'polymer entities' json files.")
    
    parser.add_argument('-s', '--session-name', type=str, default='myESMCsession',
                        help='name to give the training session (and the output checkpoint)')
    parser.add_argument('-m', '--model', type=str, default='esmc-complex',
                        help='model to use (if checkpoint is passed - it will be loaded instead)')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Checkpoint for loading a pre-trained model')
    parser.add_argument('-n', '--n-epochs', type=int, default=1,
                        help='Checkpoint for loading a pre-trained model')
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help='batch size for training')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='learning rate'),
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle instances in train')
    parser.add_argument('--cpu', action='store_true',
                        help='Force cpu usage')
    parser.add_argument('-st', '--start-from-shard', type=int, default=0,
                        help='index of shard train file to start from .default is 0. if run was crushed during epoch, you can start from the middle of epoch\
                            by starting from the last loaded train shard file.')
    
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    if args.model and args.checkpoint:
        raise Error('--model and --checkpoint can not be passed together. choose checkpoint to start training from, or load a fresh model')
    if args.model == 'esmc-complex':
        esm_model = ESMCcomplex()
    else:
        esm_model = torch.load(args.checkpoint)
            
    next_epoch = 1
    
    #if checkpoint exist, parse the epoch number from the name (name should look like: 'esmccomplex_singles_113K_e3' meaning the model was traind for 3 epochs)
    if args.checkpoint:
        prev_epoch = int(args.checkpoint.split('_')[-1][1:])
        next_epoch = prev_epoch + 1
    
    if args.checkpoint:
            base_session_name = get_session_name_from_checkpoint(args.checkpoint)
    else:
        base_session_name = args.session_name
    
    for epoch in range(next_epoch, next_epoch + args.n_epochs):
        
        trainer = ESMCTrainer(session_name=base_session_name + f'_e{epoch}',
                              esm_model=esm_model,
                              train_folder=join(config.details_vectors_path, 'train'),
                              val_pkl_path=join(config.details_vectors_path, 'validation', 'bart_vectors_0.pkl'),
                              batch_size=args.batch_size,
                              loss_fn = nn.MSELoss(),
                              optimizer=optim.Adam(esm_model.parameters(), lr=args.learning_rate),
                              shuffle=args.shuffle,
                              cpu=args.cpu,
                              start_from_shard=args.start_from_shard)
        
        trainer.single_epoch_train()
            

if __name__ == "__main__":
    main()

def get_session_name_from_checkpoint(checkpoint):
    """parse the session name from the checkpoint name
    Checkpoint name format should be "<SESSION-NAME>_e<NUM>.pkl" where e<NUM> represent the epoch number.
    
    """
    return '_'.join(Path(checkpoint).stem.split('_')[:-1])