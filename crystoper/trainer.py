from os.path import join
from os import makedirs
from glob import glob
from pathlib import Path
from tqdm import tqdm 
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from transformers import EsmTokenizer, BartTokenizer, BartModel, BartForConditionalGeneration


from . import config
from .bart import bart_decode
from .esmc_models import esm_encode, load_example
from .dataset import  Sequence2BartDataset


def train_test_val_toy_split(df, test_size, val_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df =  train_test_split(train_df, test_size=val_size, random_state=42)
    toy_df = train_df.iloc[:10]
    
    return train_df, test_df, val_df, toy_df

class ESMCTrainer():
    def __init__(self, session_name, esm_model, train_folder, val_folder, batch_size, loss_fn, optimizer, shuffle, cpu, start_from_shard=0, backup_mid_epoch=False):

        self.session_name = session_name
        self.esm_model = esm_model
        self.esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.shuffle=shuffle
        self.device = 'cpu' if cpu \
                    else 'cuda' if torch.cuda.is_available()\
                        else 'cpu'
        self.start_from_shard = start_from_shard
        self.backup_mid_epoch = backup_mid_epoch
        
        self.output_folder = join(config.checkpoints_path, session_name)
        makedirs(self.output_folder, exist_ok=True)
        
        self.log_path = join(self.output_folder, self.session_name + '.txt')
        self.logger = SimpleLogger(self.log_path, overwrite=self.start_from_shard == 0)
        
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        #Bart is only used for a single instance sanitiy inference in each epoch so cpu can be used to avoid GPU overload
        self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base') 
        
    def single_epoch_train(self):
        
        print(f'Loading model to {self.device}...')
        self.esm_model.to(self.device)
                
        #init batch over shard files (if it is not a fresh model load last batch form log)
        global_batch_idx = 0 if self.start_from_shard == 0 else\
                            load_log(self.log_path)['train_loss']['batch'].max()
        
        example = load_example()
                            
        #itterate the shard train files and train on each one of them
        for shard_file_index, (data_train_shard, path) in enumerate(load_shard_vectors(self.train_folder), ):

            #skip shard files they are below starting point
            if shard_file_index >= self.start_from_shard:

                #create dataset and loader for this piece of data
                train_dataset = Sequence2BartDataset(data_train_shard['sequences'], data_train_shard['det_vecs'], self.esm_tokenizer, device=self.device)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=train_dataset.collate)


                global_batch_idx, last_loss = train_model(self.esm_model, train_loader, self.loss_fn,
                                                        self.optimizer, global_batch_idx, self.batch_size,
                                                        self.logger, self.device)
                self.bart_model.to(self.device)
                pred = seq2sent(example["sequence"], self.esm_model, self.esm_tokenizer, self.bart_model, self.bart_tokenizer, ac=True)
                self.bart_model.to('cpu')

                print(f'Finished file {shard_file_index} from train data. predicted: {pred}')

                self.logger.info(LogLine(batch=global_batch_idx,
                                    i = global_batch_idx * self.batch_size,
                                    pred_sent = pred))

                print('Evaluating val loss...')

                #perform validation (done after each shard file of the training)
                val_loss = evaluate_model(self.esm_model, self.esm_tokenizer, self.val_folder, self.batch_size, self.loss_fn, self.device, self.shuffle)
                

                self.logger.info(LogLine(batch=global_batch_idx,
                                    i = global_batch_idx * self.batch_size,
                                    val_loss=val_loss))

                print(f"i: {self.batch_size*global_batch_idx},  val loss: {val_loss}")

                torch.cuda.empty_cache()

                if self.backup_mid_epoch:
                    #dump model (dump after each shard file as a backup incase of connection loss)
                    model_path = join(self.output_folder, self.session_name + f'_trainfile{shard_file_index}.pkl')
                    print(f'Dumping model to {model_path}...')
                    torch.save(self.esm_model, model_path)
                    print(f'Saved model to {model_path}')
                    self.logger.dump()

        #dump final model
        model_path = join(self.output_folder, self.session_name + '.pkl')
        print(f'Dumping model to {model_path}...')
        torch.save(self.esm_model, model_path)
        print(f'Saved model to {model_path}')
        self.logger.dump()


        print('DONE!')

# load all BART decoded shared files from a folder (by order)
def load_shard_vectors(path, start_from=0):
    paths =  glob(join(path, 'bart_vectors_*.pkl'))
    paths = sorted(paths, key=lambda path: int(Path(path).stem.split('_')[-1])) #sort by number at end of name
    
    for path in paths:
        n_file = int(Path(path).stem.split('_')[-1])
        print(f'parsed n: {n_file}')
        if n_file >= start_from:
            print(f'Loading {path}...')
            yield torch.load(path, weights_only=True), path
        else:
            print(f'skipping {path}...')

def train_model(model, train_loader,
                loss_fn, optimizer,
                global_batch_idx,
                batch_size,
                logger,
                device, verbose=True):
    
    model.to(device)

    model.train()  # Set model to training mode
    running_train_loss = 0.0

    # Training loop
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        output_matrices = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = loss_fn(output_matrices, batch['target_matrices'])

        # Backward pass
        loss.backward()
        optimizer.step()


        result = LogLine(epoch='',
                        batch=global_batch_idx + batch_idx,
                         i= (global_batch_idx + batch_idx) * batch_size,
                        train_loss= loss.item())

        logger.info(result)

        if batch_idx % 100 == 0:
            logger.dump()
            print(f'i: {(global_batch_idx + batch_idx) * batch_size}, train_loss: {loss.item()}')

        if batch_idx % 500 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()

        logger.dump()


    return batch_idx + global_batch_idx, loss.item()

#before first epoch training we will create an evalutaion method
def evaluate_model(model,tokenizer, val_folder,  batch_size, loss_fn, device, shuffle):

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # Disable gradient calculations
        #iterate the val shard files, load them and evaluate
        for _, (data_val_shard, _) in enumerate(load_shard_vectors(val_folder), ):
            #create validation datasets and loader
            val_dataset = Sequence2BartDataset(data_val_shard['sequences'], data_val_shard['det_vecs'], tokenizer, device=device)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate, shuffle=shuffle)
            
            for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):

                output_matrices = model(batch['input_ids'], attention_mask=batch['attention_mask'])

                loss = loss_fn(output_matrices, batch['target_matrices'])
                total_loss += loss.item()  # Accumulate the batch loss
                num_batches += 1


    avg_loss = total_loss / num_batches  # Calculate average loss over all batches
    return avg_loss

def seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer, ac=False):
    if ac:
        with autocast():
            return _seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer)
    else:
        return _seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer)


def _seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer, bart_device='cpu'):
    esm_encoded_matrix = esm_encode(seq, esm_model, esm_tokenizer)
    decoded_sentence = bart_decode(esm_encoded_matrix, bart_model, bart_tokenizer)
    return decoded_sentence



def load_data_toy():
    #TOY DATA WITH A SINGLE INSTANCE
    #we will prepere a toy data for the overfitting

    example = load_example()

    data_toy = {'det_vecs': example['det_vec'].unsqueeze(0), #unsqeeze reshape it to 1 x n x 768 to resamble batch
                'pdb_ids': [example['pdb_id']],
                'sequences': [example['sequence']],
                'details': [example['pdbx_details_true']]}

    return data_toy

####### TRAINING LOGGER

class SimpleLogger():

    def __init__(self, path, overwrite=False):
        self.path = path
        self.text = ''
        self.last = ''

        if overwrite:
            with open(self.path, 'w'): pass

    def info(self, text):
        self.text += str(text)
        self.text += '\n'
        self.last = text

    def dump(self, flush=True):

        with open(self.path, 'a') as f:
            f.write(self.text)

        self.text = ''

def read_log_file(file_path):
    dict_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # Use ast.literal_eval to safely parse the line as a dictionary
            parsed_dict = ast.literal_eval(line.strip())
            dict_list.append(parsed_dict)
    return dict_list

def load_log(path):
    df = pd.DataFrame(read_log_file(path))
    if len(df)>0:
        return {'train_loss': df[df.train_loss.notna()],
                'val_loss' : df[df.val_loss.notna()],
                'pred_sent' :  df[df['pred_sent'].notna()]}


class LogLine():
    #define a line in log

    def __init__(self, **kwargs):
        self.d = {'epoch': None,
                'batch': None,
                'i': None,
                'train_loss': None,
                "val_loss" : None,
                'true_sent': None,
                'pred_sent': None}
        for k,v in kwargs.items():
            if k in self.d.keys():
                self.d[k] = v
            else:
                raise ValueError('Key not define in LogLine!')

    def __str__(self) -> str:
        return str(self.d)

    def __setitem__(self, index, value):
        if index in self.d.keys():
            self.d[index] = value
        else:
            raise ValueError('Key not define in LogLine!')
    def keys(self):
        return self.d.keys()

    def __repr__(self):
        return str(self.d)



