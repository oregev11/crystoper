import numpy as np
import pandas as pd
from tqdm import tqdm
from os import makedirs
from os.path import join

from transformers import EsmTokenizer, EsmForMaskedLM, GPT2Tokenizer, GPT2Model
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import torch
from torch.utils.data import DataLoader

from crystoper.dataset import ProteinSequencesDataset , BartDetailsDataset
from crystoper.bart import bart_encode, bart_decode



SEQ_COL = 'sequence'
SEQUENCES_VEC = 'seq_vec'
SEQUENCES_MODEL = 'seq_model'

DETAILS_COL = 'pdbx_details'
DETAILS_VEC = 'det_vec'
DETAILS_MODEL = 'det_model'

CHECKPOINTS = {'esm2': 'facebook/esm2_t33_650M_UR50D',
               'bart': 'facebook/bart-base'}


VERBOSE = True

TEST_SENTENCE = '0.1M Hepes-NaOH (pH7.0), 50%(v/v) MPD'

class Sequence2Vectors():
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
    def __init__(self, model, batch_size, data_constructor=ProteinSequencesDataset, pooling=None, hidden_fn=None, cpu=False):
        self.device = 'cpu' if cpu \
                        else 'cuda' if torch.cuda.is_available() \
                            else 'cpu'
        
        #get model name from a list of supported models
        if CHECKPOINTS.get(model):
            self.model_name = CHECKPOINTS[model]
        else:
            raise ValueError(f'The selected model {model} is not supported!')
            
        self.batch_size = batch_size
        self.data_constructor = data_constructor
        self.pooling = pooling
        self.hidden_fn = hidden_fn


    def get_model(self):
        if 'esm2' in self.model_name:
            model = EsmForMaskedLM.from_pretrained(self.model_name, output_hidden_states=True)
            tokenizer = EsmTokenizer.from_pretrained(self.model_name)

        return model, tokenizer

    def get_vectors(self, sequences):

        model, tokenizer = self.get_model()
        model.to(self.device)
        
        data = self.data_constructor(sequences, tokenizer, device=self.device)
        data_iter = DataLoader(data, batch_size=self.batch_size, collate_fn=data.collate)

        if VERBOSE:
            print(f'Starting protein sequence extraction! {len(sequences)} sequences in {len(sequences) // self.batch_size} batches...')
            print(f'Using {self.device}')
        seq2vec = Sequence2Vectors(model, pooling=self.pooling, hidden_fn=self.hidden_fn)
        vectors = seq2vec(data_iter)

        return vectors

    def __call__(self, sequences):
        
        return self.get_vectors(sequences.values)



#####################################################



    
class DetailsVectorizer():
    def __init__(self, model, batch_size, dump_batch_size=1000, pooling=None, hidden_fn=None, cpu=False):
        self.device = 'cpu' if cpu \
                        else 'cuda' if torch.cuda.is_available() \
                            else 'cpu'

        print(f"Using {self.device} fro details vectorization!" )
        
        #get model name from a list of supported models
        if CHECKPOINTS.get(model):
            self.model_name = CHECKPOINTS[model]
        else:
            raise ValueError(f'The selected model {model} is not supported!')
        
        self.batch_size = batch_size
        self.dump_batch_size = dump_batch_size
        self.pooling = pooling


    def get_model(self):
        if 'bart' in self.model_name:
            model =  BartForConditionalGeneration.from_pretrained(self.model_name)
            tokenizer = BartTokenizer.from_pretrained(self.model_name)
            
            model.to(self.device)
            
            #test model
            encoder_hidden_states = bart_encode(TEST_SENTENCE, model, tokenizer, self.device)
            reconstructed_sentence = bart_decode(encoder_hidden_states, model, tokenizer, 'cpu')
            print('SANITY CHECK: encoding and decoding a sentence with BART. the decode sentence should be the same as the original....')
            print(f"Original Sentence:\t{TEST_SENTENCE}")
            print(f"Reconstructed Sentence:\t{reconstructed_sentence}")
            print(f'The hidden state shape: {encoder_hidden_states.shape}\n')
        
        return model, tokenizer

    def get_vectors_and_dump(self, data, output_folder, n_batch_to_dump=1000):
        
        if 'bart' in self.model_name:
            
            makedirs(output_folder, exist_ok=True)

            model, tokenizer = self.get_model()
            print(f'Loading model to {self.device}')            
            model.to(self.device)
            
            dataset = BartDetailsDataset(data.pdb_id, data.sequence, data.pdbx_details, tokenizer, device=self.device)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.collate)

            if VERBOSE:
                print(f'Starting sentences embedding extraction using {self.model_name}! {len(data)} instances in {len(data) // self.batch_size} batches...')
            
            det_vecs = []
            pdb_ids = []
            sequences = []
            details = []

            torch.cuda.empty_cache()

            save_batch_index = 0

                        
            for i, batch in tqdm(enumerate(data_loader), total = len(data_loader)):

                hidden = model.model.encoder(**batch['input_ids']).last_hidden_state

                det_vecs.append(hidden.detach().cpu())
                pdb_ids += batch['pdb_ids']
                sequences += batch['sequences']
                details += batch['details']
                torch.cuda.empty_cache()

                del hidden

                if i % self.dump_batch_size == 0 and i > 0:
                    det_vecs = torch.cat(det_vecs, dim=0)

                    dump_dict = {'det_vecs': det_vecs,
                                'pdb_ids': pdb_ids,
                                'sequences': sequences,
                                'details': details}

                    output_path = join(output_folder, f'bart_vectors_{save_batch_index}.pkl')

                    print(f'Dumping vectors to {output_path}...')
                    torch.save(dump_dict, output_path)

                    del dump_dict

                    print(f'Dumped vectors to {output_path}')

                    det_vecs = []
                    pdb_ids = []
                    sequences = []
                    details = []
                    save_batch_index += 1


            det_vecs = torch.cat(det_vecs, dim=0)

            dump_dict = {'det_vecs': det_vecs,
                        'pdb_ids': pdb_ids,
                        'sequences': sequences,
                        'details': details}

            

            output_path = join(output_folder, f'bart_vectors_{save_batch_index}.pkl')

            print(f'Dumping vectors to {output_path}...')
            torch.save(dump_dict, output_path)
            del dump_dict

            print(f'Dumped vectors to {output_path}')

    def __call__(self, data, output_folder):
        return self.get_vectors_and_dump(data, output_folder)



