"""
train.py
train model
"""
import argparse
from os.path import join
from os import makedirs
from pathlib import Path
import re 
import torch
import torch.nn as nn
import torch.optim as optim


from crystoper import config
from crystoper.processor import filter_by_pdbx_details_length, filter_for_single_entities
from crystoper.utils.general import vprint, make_parent_dirs
from crystoper.utils.data import dump_json
from crystoper.esmc_models import ESMCcomplex
from crystoper.trainer import ESMCTrainer, load_train_and_val_loss_from_logs_folder


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
    parser.add_argument('-bu', '--backup-mid-epoch', action='store_true',
                        help='backup the model in the middle of the epoch after each train shard file')
    parser.add_argument('--toy-train', action='store_true',
                        help='use toy data for training for debug purposes')
    parser.add_argument('--save-last-only', action='store_true',
                        help='save only the last checkpoint after the last epoch. (if not using toy train it is best to save after each epoch to avoid losing the trained model)')
    
    
    
    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()
    
    if args.model == 'esmc-complex':
        esm_model = ESMCcomplex()
        vprint(f"A fresh {args.model} has been created!")
    else:
        raise ValueError('Model cannot be resolved')
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(esm_model.parameters(), lr=args.learning_rate)
    epoch = 1
    
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        esm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_fn = checkpoint['loss_fn']
                
        vprint(f"loaded previous model from checkpoint {args.checkpoint}")
        
        #update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        
    else:
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(esm_model.parameters(), lr=args.learning_rate)
           
    
    start_from_shard = args.start_from_shard
    
    output_folder = join(config.checkpoints_path, args.session_name)
    params_path = join(output_folder, 'params.json')
    make_parent_dirs(params_path)
    dump_json(vars(args), params_path)
    
    #If checkpoint was passed, parse the epoch number from the name (name should look like: 'esmccomplex_singles_113K_e3.pkl' meaning the model was traind for 3 epochs)
    #If the passed checkpoint was from incomplete epoch, it should look like 'esmccomplex_singles_113K_e3_trainfile2.pkl' meaning the model is from epoch 3 trained on trainfiles 0, 1 and 2. 
    #training will continue the session from on the rest of the train files
    if args.checkpoint:
        #if last epoch was incomplete
        if 'trainfile' in args.checkpoint:
            last_train_file = get_last_train_file_from_checkpoint(args.checkpoint)
            print(f'Last train file for checkpoint was: {last_train_file}')
            
            start_from_shard = last_train_file + 1
            print(f'Starting current train from train file: {start_from_shard}')
            
            base_session_name = get_session_name_from_checkpoint(args.checkpoint, is_trainfile=True)
            
            #if this is a trainfile epoch (mid-epoch checkpoint) we start from the same epoch
            prev_epoch = epoch
            next_epoch = epoch
            print(f'Train will continue the previous epoch: epoch {next_epoch}')
            
        else:
            prev_epoch = epoch
            next_epoch = prev_epoch + 1
            base_session_name = get_session_name_from_checkpoint(args.checkpoint)
            
    else:
        base_session_name = args.session_name
        next_epoch = 1
    
    for epoch in range(next_epoch, next_epoch + args.n_epochs):
        
        vprint(f'\n\n*************************\nStarting Epoch {epoch} (out of {next_epoch + args.n_epochs})!\n***********************\n\n')
        
        trainer = ESMCTrainer(epoch=epoch,
                              session_name=base_session_name + f'_e{epoch}',
                              esm_model=esm_model,
                              train_folder=join(config.details_vectors_path, 'toy' if args.toy_train else 'train'),
                              val_folder=join(config.details_vectors_path, 'toy' if args.toy_train else 'val'),
                              batch_size=args.batch_size,
                              loss_fn = loss_fn,
                              optimizer=optimizer,
                              shuffle=args.shuffle,
                              cpu=args.cpu,
                              start_from_shard=start_from_shard,
                              backup_mid_epoch=args.backup_mid_epoch,
                              backup_end_epoch=(not args.save_last_only))
        
        trainer.single_epoch_train()
        
        #
        _ = load_train_and_val_loss_from_logs_folder(output_folder, prefix=base_session_name)
        
    #dump the model after the end of all epochs
    if args.save_last_only:
        checkpoint_path = join(output_folder, args.session_name + f'_e{epoch}.pth')
        make_parent_dirs(checkpoint_path)
                
        print(f'Dumping model to {checkpoint_path}...')
        torch.save({
                'model_state_dict': esm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss_fn': loss_fn,
            }, checkpoint_path)
        print(f'Saved model to {checkpoint_path}')
        


def get_session_name_from_checkpoint(checkpoint, is_trainfile=False):
    """parse the session name from the checkpoint name
    Checkpoint name format should be "<SESSION-NAME>_e<NUM>.pkl" where e<NUM> represent the epoch number.
    if is_trainfile True - assume the term 'is_trainfile' in the checkpoint name and parse accordingly.
    
    """
    if is_trainfile:
        return '_'.join(Path(checkpoint).stem.split('_')[:-2])
    else:
        return '_'.join(Path(checkpoint).stem.split('_')[:-1])

def get_last_train_file_from_checkpoint(checkpoint):
    
    match = re.search(r'trainfile(\d+)', checkpoint)
    if match:
        return int(match.group(1))
    return None  


if __name__ == "__main__":
    main()