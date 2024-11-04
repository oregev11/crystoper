
import torch
from transformers.modeling_outputs import BaseModelOutput

N_WORDS_IN_DETAILS = 250

def bart_encode(sentence, model, tokenizer, max_len=N_WORDS_IN_DETAILS, device='cpu'):

    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True).to(device)

    # Get the embeddings from the encoder
    with torch.no_grad():
        hidden_states = model.model.encoder(**inputs).last_hidden_state #indclude only 'last_hidden_state'

    if hidden_states.size(1) < max_len:
        padding = torch.zeros((hidden_states.size(0), max_len - hidden_states.size(1), hidden_states.size(2)), device=device)
        hidden_states = torch.cat((hidden_states, padding), dim=1)

    else:
        hidden_states = hidden_states[:, :max_len, :]

    return hidden_states

def bart_decode(encoder_hidden_states, model, tokenizer, device='cpu'):

    wrapped_hiddens = BaseModelOutput(last_hidden_state=encoder_hidden_states)

    generated_ids = model.generate(encoder_outputs=wrapped_hiddens, min_length=0, max_length=N_WORDS_IN_DETAILS,
                                   length_penalty=1, repetition_penalty=1.2, early_stopping=True)
    reconstructed_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return reconstructed_sentence
