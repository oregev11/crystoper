import numpy as np
import pandas as pd
from tqdm import tqdm
from crystoper.dataset import ProteinSequences

from transformers import EsmTokenizer, EsmForMaskedLM
import torch
from torch.utils.data import DataLoader

SEQ_COL = 'sequence'
SEQUENCES_VEC = 'seq_vec'
SEQUENCES_MODEL = 'seq_model'

DETAILS_COL = 'pdbx_details'
DETAILS_VEC = 'det_vec'
DETAILS_MODEL = 'det_model'



VERBOSE = True

class Sequence2Vector():
    def __init__(self, model, pooling='average', hidden_fn=None):
        self.model = model
        self.pooling = pooling
        self.hidden_fn = hidden_fn

    def __call__(self, data_iter):
        results = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_iter):
                lengths = batch.pop('length')
                if self.hidden_fn is None:
                    hiddens = self.model(**batch).hidden_states[-1]
                else:
                    hiddens = self.hidden_fn(self.model, batch)

                #the hidden layer is in the shape of <1280xPROTEIN_LENGTH>.
                # By pooling we create a 1-D vector in the size 1280.
                # This means all extracted vectors are same length.
                if self.pooling == 'average':
                    result = hiddens.sum(1).div(lengths.unsqueeze(1))
                elif self.pooling == 'max':
                    result = hiddens.max(1)
                #if pooling not passed - the full hidden matrix will be exported
                else:
                    result = hiddens[:,-1,:]

                results.append(result.to('cpu'))
                del hiddens
                torch.cuda.empty_cache()

            results = torch.cat(results)
        return results

class SequencesVectorizer():
    def __init__(self, model, batch_size, data_constructor=ProteinSequences, pooling=None, hidden_fn=None, cpu=False):
        self.device = 'cpu' if cpu \
                        else 'cuda' if torch.cuda.is_available() \
                            else 'cpu'
        self.model_name = model
        self.batch_size = batch_size
        self.data_constructor = data_constructor
        self.pooling = pooling
        self.hidden_fn = hidden_fn


    def get_model(self):

        model = EsmForMaskedLM.from_pretrained(self.model_name, output_hidden_states=True)
        tokenizer = EsmTokenizer.from_pretrained(self.model_name)

        return model, tokenizer

    def get_vectors(self, sequences):

        model, tokenizer = self.get_model()
        data = self.data_constructor(sequences, tokenizer, device=self.device)
        data_iter = DataLoader(data, batch_size=self.batch_size, collate_fn=data.collate)

        if VERBOSE:
            print(f'Starting protein sequence extraction! {len(sequences)} sequences in {len(sequences) // self.batch_size} batches...')
        seq2vec = Sequence2Vector(model.to(self.device), pooling=self.pooling, hidden_fn=self.hidden_fn)
        vectors = seq2vec(data_iter)

        return vectors

    def __call__(self, data):
        
        data[SEQUENCES_MODEL] = self.model_name
        data[SEQUENCES_VEC] = self.get_vectors(data['df'][SEQ_COL].values)
        
        return data



