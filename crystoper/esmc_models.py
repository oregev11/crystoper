#esmc_models.py
#ESM-crystoper models

from . import config
import torch 
from transformers import EsmForMaskedLM
N_WORDS_IN_DETAILS = 250 
#encode a single sequence with ESM
def esm_encode(sequence, model, tokenizer):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(sequence, return_tensors="pt")
        
        device = next(model.parameters()).device
        input_ids = {key: value.to(device) for key, value in input_ids.items()}

        return model(**input_ids)

def load_example(path=config.example_path):
    #load a simple example of sequence, BART matrix and true pdbx_details
    return torch.load(path)

class ESMCcomplex(torch.nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        
        # Load the pre-trained ESM model
        self.esm = EsmForMaskedLM.from_pretrained(model_name, output_hidden_states=True)

        # Freeze the layers if you don't want to fine-tune the whole model
        for param in self.esm.parameters():
            param.requires_grad = False
            
            

        # Define a linear layer to output the desired matrix size (500x768)
        # Assuming the hidden size of ESM is 1280 and we want a 500x768 output
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear1 = torch.nn.Linear(1280, 1024)  # Intermediate layer
        self.linear2 = torch.nn.Linear(1024, 768 * N_WORDS_IN_DETAILS)
        self.activation = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(1024)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1][:, :, :].mean(dim=1)

        #added complexity to the model - residual connection + normalization
        hidden_states = self.dropout(hidden_states)
        transformed = self.activation(self.linear1(hidden_states))
        transformed = self.layer_norm(transformed + self.linear1(hidden_states))  # Residual connection
        matrix_output = self.linear2(transformed).view(-1, N_WORDS_IN_DETAILS, 768)

        return matrix_output
    
    
class ESMCavg(torch.nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        # Load the pre-trained ESM model
        self.esm = EsmForMaskedLM.from_pretrained(model_name, output_hidden_states=True)

        # Freeze the layers if you don't want to fine-tune the whole model
        for param in self.esm.parameters():
            param.requires_grad = False

        # Define a linear layer to output the desired matrix size (500x768)
        # Assuming the hidden size of ESM is 1280 and we want a 500x768 output
        self.linear = torch.nn.Linear(1280, 768 * self.n_words)

    def forward(self, input_ids, attention_mask=None):
        # Get the hidden states from the ESM model
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the [CLS] token or aggregate outputs
        hidden_states = outputs.hidden_states[-1][:, :, :].mean(dim=1)  # shape: (n_tokens x 1280) ->  (1 x 1280)

        # Pass through the custom linear layer to get the desired shape
        matrix_output = self.linear(hidden_states)
        matrix_output = matrix_output.view(-1, self.n_words, 768)  # Reshape to (batch_size, N_WORDS_IN_DETAILS, 768)

        return matrix_output
    
class ESMC(torch.nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        # Load the pre-trained ESM model
        self.esm = EsmForMaskedLM.from_pretrained(model_name, output_hidden_states=True)

        # Freeze the layers if you don't want to fine-tune the whole model
        for param in self.esm.parameters():
            param.requires_grad = False

        # Define a linear layer to output the desired matrix size (500x768)
        # Assuming the hidden size of ESM is 1280 and we want a 500x768 output
        self.linear = torch.nn.Linear(1280, 768 * self.n_words)

    def forward(self, input_ids, attention_mask=None):
        # Get the hidden states from the ESM model
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the [CLS] token or aggregate outputs
        hidden_states = outputs.hidden_states[-1][:, 0, :]  # (batch_size, hidden_size)

        # Pass through the custom linear layer to get the desired shape
        matrix_output = self.linear(hidden_states)
        matrix_output = matrix_output.view(-1, self.n_words, 768)  # Reshape to (batch_size, N_WORDS_IN_DETAILS, 768)

        return matrix_output