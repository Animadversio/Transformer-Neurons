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
import torch.nn as nn # import ModuleList
from core.interp_utils import topk_decode
from copy import deepcopy
#%%
original = deepcopy(model.bert.encoder.layer)

#%% Recover the layer
model.bert.encoder.layer = original# nn.ModuleList([original[i] for i in layer_shfl])

#%% Shuffle the layers
layer_shfl = torch.randperm(12)
model.bert.encoder.layer = nn.ModuleList([original[i] for i in layer_shfl])
#%% Repeat certain layers
# layer_order = 5 * torch.ones(12, dtype=int)
layer_order = torch.arange(12, dtype=torch.int)
# layer_order[8] = 6
# layer_order[7] = 7
# layer_order[6] = 8
# layer_order[3] = 1
# layer_order[2] = 2
# layer_order[1] = 3
# layer_order[2:9] = 4
model.bert.encoder.layer = nn.ModuleList([original[i] for i in layer_order])
#%%
# text = "The quick brown fox [MASK] over the lazy dog."
text = "Vatican is located in the city of [MASK]."
# text = "The forbidden palace is located in the city of [MASK]."
# text = "New York is the [MASK] of the US."
# text = "Vatican is located on the [MASK] part of Italy."
token_ids = tokenizer.encode(text, return_tensors='pt')
with torch.no_grad():
    outputs = model(token_ids, output_hidden_states=True)

#% Layer output across the layers
k = 8
print("Layer order: ", layer_order.tolist())
print(f"Decoded top {k} tokens for the masked word across the layers")
# mask_loc = token_ids[0].tolist().index(tokenizer.mask_token_id)
mask_loc = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]
read_loc = mask_loc.item()
for layeri in range(len(outputs.hidden_states)):
    logits = model.cls(outputs.hidden_states[layeri])  #
    toks, vals = topk_decode(logits[0, read_loc, :], tokenizer, k)
    print(toks[0], )
#%%
logits = outputs.logits
for i in range(logits.size(1)):
    toks, vals = topk_decode(logits[0, i, :], tokenizer, 5)
    print(toks, )#"\n", vals)
#%%
logits = model.cls(outputs.hidden_states[0])  #
for i in range(logits.size(1)):
    toks, vals = topk_decode(logits[0, i, :], tokenizer, 5)
    print(toks, )  # "\n", vals)
