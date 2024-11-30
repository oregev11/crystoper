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
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use. "cuda" or "cpu"')
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
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available. please pass device "cpu"')
    
    if args.model == 'esmc-complex':
        esm_model = ESMCcomplex().to(args.device)
        vprint(f"A fresh {args.model} has been created!")
    else:
        raise ValueError('Model cannot be resolved')
        
    
        
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        esm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_fn = checkpoint['loss_fn']
        prev_epoch = checkpoint['epoch']
        session_name = checkpoint['session_name']
        
        epoch = prev_epoch + 1
                
        vprint(f"loaded previous model from checkpoint {args.checkpoint}")
        
        #update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        
    else:
        optimizer = optim.Adam(esm_model.parameters(), lr=args.learning_rate)
        loss_fn = nn.MSELoss()
        epoch = 1
        session_name = args.session_name
        
            
    last_epoch = epoch + args.n_epochs - 1 #the last epoch to be in current run     
    
    for epoch in range(epoch, last_epoch + 1):
        
        output_folder = join(config.checkpoints_path, session_name + f'_e{epoch}')
        params_path = join(output_folder, 'params.json')
        make_parent_dirs(params_path)
        dump_json(vars(args), params_path)
        
        vprint(f'\n\n*************************\nStarting Epoch {epoch} (out of {last_epoch})!\n***********************\n\n')
        
        trainer = ESMCTrainer(epoch=epoch,
                              session_name=session_name,
                              esm_model=esm_model,
                              train_folder=join(config.details_vectors_path, 'toy' if args.toy_train else 'train'),
                              val_folder=join(config.details_vectors_path, 'toy' if args.toy_train else 'val'),
                              batch_size=args.batch_size,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              shuffle=args.shuffle,
                              device=args.device,
                              start_from_shard=0,
                              backup_mid_epoch=args.backup_mid_epoch,
                              backup_end_epoch=(not args.save_last_only))
        
        trainer.single_epoch_train()
        
        #
        _, _, summary = load_train_and_val_loss_from_logs_folder(output_folder, prefix=session_name)
        summary.to_csv(join(output_folder, 'summary.csv'), index=False)
        
        
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
                'session_name': session_name,
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