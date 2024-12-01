import os
import json 
import numpy as np
import pandas as pd
from os.path import isfile, dirname, join
from pathlib import Path
import csv
from tqdm import tqdm
from .. import config

ENTITY_SPLIT_CHAR = '_'
CSV_BATCH_SIZE = 1000
VEC_SUFFIX = '.npy'

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def dump_json(obj, path):
    
    with open(path, 'w') as f:
        return json.dump(obj, f)

# class Data():
#     """Class for handling crystoper data
    
#     ARGS:
#         df_path(str): path to dataframe with processed pdb data
#         seq_vec_path(str): path to vectors numpy file with 'sequences' embedded vectors
#         det_vec_path(str): path to vectors numpy file with 'pdbx_details' embedded vectors
#     """
#     def __init__(self,
#                  df_path=None,
#                  seq_model=None,
#                  det_model=None):
                
#         self.df_path = df_path
#         self.seq_vec_path = seq_vec_path
#         self.det_vec_path = det_vec_path
        
#         self.seq_model = seq_model
#         self.det_model = det_model
        
#         self.df = None
#         self.seq_vec = None
#         self.det_vec = None
        
        
    
#     def load(self):
#         if self.df_path:
#             self.df = pd.read_csv(self.df_path)
#         if self.seq_model:
#             path = get_vectors_path(self.seq_model, 'sequences')
#             self.seq_vec = load_vectors()
#         if self.det_model:
#             path = get_vectors_path(self.det_model, 'details')
#             self.det_vec = load_vectors()
            
            

# def pack_data(df):
#     return {'df': df}

def load_vectors(model_name, model_type):
    path = get_vectors_path(model_type, model_name)
    path = os.join(path, model_name)  + VEC_SUFFIX
    
    return np.load(path)

def dump_vectors(vectors, model_name, model_type):
    path = get_vectors_path(model_name, model_type)
    path = join(path, model_name)  + VEC_SUFFIX
    
    make_dir(str(Path(path).parent))
    
    np.save(path, vectors)
    
    return path

def get_vectors_path(model_name, model_type):
    if model_type == 'sequences':
        path = config.sequences_vectors_path
    elif model_type == 'details':
        path = config.details_vectors_path
    else:
        raise ValueError('Model type path is undefined!')
    
    return path



def write_to_csv_in_batches(data_generator, headers, output_file, batch_size=CSV_BATCH_SIZE, verbose=True, tqdm_total=None):
    
    buffer = []
    
    with open(output_file, mode='w', newline='') as file:
        
        writer = csv.writer(file)
        
        writer.writerow(headers)
        
        if verbose:
            for row in tqdm(data_generator, total=tqdm_total):
                buffer.append(row)
                
                if len(buffer) >= batch_size:
                    writer.writerows(buffer)
                    buffer.clear()  
        
        if buffer:
            writer.writerows(buffer)


def filter_pdb_polymer_ids(polymer_ids, relevant_ids):
    """
    Use pandas for fast filtering of large data.

    Args:
        polymer_ids (list): lister of polymer ids like ['1AON-1', '1AON-2', '5CQ2-1',....]
        relevant_ids (list): relevant ids to filter by. like ['1AON1', '5CQ2',...]
    
    """
    
    df = pd.DataFrame([id.split(ENTITY_SPLIT_CHAR) for id in polymer_ids], columns=['id', 'entity'])
    filter_df = pd.DataFrame({'id': relevant_ids})
    
    df =  df.merge(filter_df, how='inner')
    
    return list(df['id'] + '-' + df['entity'])
    
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
    return {'train_loss': df[df.train_loss.notna()],
            'val_loss' : df[df.val_loss.notna()],
            'pred_sent' :  df[df['pred_sent'].notna()]}
    
def load_train_and_val_loss_from_logs_folder(path, prefix='esmccomplex_singles_113K_'):
    train_loss = pd.DataFrame()
    val_loss = pd.DataFrame()

    for i in tqdm(range(5,22)):
        name = f'{prefix}e{i}'
        log_path = join(path, name + '.txt')
        log = load_log(log_path)

        temp_train_loss = pd.DataFrame(log['train_loss'])
        temp_val_loss = pd.DataFrame(log['val_loss'])

        temp_train_loss['epoch'] = int(i)
        temp_val_loss['epoch'] = int(i)


        train_loss = pd.concat([train_loss, temp_train_loss])
        val_loss = pd.concat([val_loss, temp_val_loss])
        
    df1 = train_loss.groupby('epoch')['train_loss'].mean()
    df2 = val_loss.groupby('epoch')['val_loss'].mean()
    summary = pd.concat([df1, df2], axis=1).reset_index()

    print(f"train loss over epochs was: \n{summary.to_string()}")
    
    return train_loss, val_loss
    