"""
Set up masked language model task
See the effect of certain internal representation on the behavior.
"""
import torch
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM
from transformers import pipeline
# configuration = BertConfig()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Initializing a model from the bert-base-uncased style configuration
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.requires_grad_(False)
model.eval()
#%%
text = "The quick brown fox [MASK] over the lazy dog."
tokens = tokenizer.encode(text, return_tensors='pt')
with torch.no_grad():
    outputs = model(tokens, output_hidden_states=True)
#%%
#%%
def topk_decode(logits, tokenizer, k=10):
    val, ids = torch.topk(logits, k, dim=-1)
    toks = []
    if len(ids.shape) == 1:
        ids = ids.unsqueeze(0)

    for i in range(len(ids)):
        toks.append(tokenizer.convert_ids_to_tokens(ids[i]))

    return toks, val.numpy()

# last_hidden = model.bert(tokens)[0]
# logits = model.cls(last_hidden)  # outputs.hidden_states[-1]
# topk_decode(logits[0, :], tokenizer)
#%%
logits = outputs.logits
for i in range(logits.size(1)):
    toks, vals = topk_decode(logits[0, i, :], tokenizer)
    print(toks, "\n", vals)
#%%
logits = model.cls(outputs.hidden_states[2])  #
for i in range(logits.size(1)):
    toks, vals = topk_decode(logits[0, i, :], tokenizer)
    print(toks, )  # "\n", vals)
#%%
toks, vals = topk_decode(logits[0, 5, :], tokenizer)
print(toks, )
#%%
# Evolution through layers - Concentration / entropy decrease
#     Dynamics lead to the entropy decrease.
# Learning Dynamics through training. A ha moment.
# Entropy and center of mass.
# Evolution of prediction through layer.
#%%
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
#%%
unmasker("The quick brown [MASK] jumped over the sleeping dog.")
#%%
#%%
text = "The quick brown fox [MASK] over the lazy dog."
tokens = tokenizer.encode(text, return_tensors='pt')
outputs = model.forward(input_ids=tokens, token_type_ids=None)  # token_type_ids
mask_emb = outputs.last_hidden_state[:, 4, :]
#%%
# masktok_dist = torch.softmax(outputs.last_hidden_state @ model.embeddings.word_embeddings.weight.T, dim=2)
masktok_dist = outputs.last_hidden_state @ model.embeddings.word_embeddings.weight.T
#%%
val, ids = torch.topk(masktok_dist[0, 5], 10)
tokenizer.convert_ids_to_tokens(ids)

#%%
val, ids = torch.topk(masktok_dist[0,], 10, dim=-1)
for i in range(len(ids)):
    print(tokenizer.convert_ids_to_tokens(ids[i]),'\n', val[i].numpy())
#%%
model.embeddings(tokenizer.encode("The quick brown fox [MASK]", return_tensors='pt'))