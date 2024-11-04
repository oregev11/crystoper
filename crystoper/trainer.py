
from sklearn.model_selection import train_test_split

def train_test_val_toy_split(df, test_size, val_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df =  train_test_split(train_df, test_size=val_size, random_state=42)
    toy_df = train_df.iloc[:10]
    
    return train_df, test_df, val_df, toy_df


# load all BART decoded shared files from a folder (by order)
def load_shard_vectors(path, start_from=0):
    for path in glob(join(path, 'bart_vectors_*.pkl')):
        n_file = int(Path(path).stem.split('_')[-1])
        print(f'parsed n: {n_file}')
        if n_file >= start_from:
            print(f'Loading {path}...')
            yield torch.load(path, weights_only=True), path
        else:
            print(f'skipping {path}...')

def train_model(model, train_loader, loss_fn, optimizer, epoch, global_batch_idx, batch_size,
                                bart_model, bart_tokenizer, logger, device, verbose=True):
    """
    Train the model with validation loss tracking.

    Args:
    - model: The PyTorch model to be trained.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - loss_fn: The loss function to use.
    - optimizer: The optimizer for updating model parameters.
    - num_epochs: Number of training epochs.
    - device: Device to use ('cpu' or 'cuda').

    Returns:
    - nested dict - loging of performacnce during train
    """
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


        result = LogLine(epoch=epoch,
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
def evaluate_model(model, val_loader, batch_size, loss_fn, device):

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculations
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            output_matrices = model(batch['input_ids'], attention_mask=batch['attention_mask'])

            loss = loss_fn(output_matrices, batch['target_matrices'])
            total_loss += loss.item()  # Accumulate the batch loss



    avg_loss = total_loss / idx  # Calculate average loss over all batches
    return avg_loss

def seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer, device, ac=False):
    if ac:
        with autocast():
            return _seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer, device)
    else:
        return _seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer, device)


def _seq2sent(seq, esm_model, esm_tokenizer, bart_model, bart_tokenizer, device):
    esm_encoded_matrix = esm_encode(seq, esm_model, esm_tokenizer, device)
    decoded_sentence = bart_decode(esm_encoded_matrix, bart_model, bart_tokenizer, device)
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
    df = pd.DataFrame(read_log_file(log_path))
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



