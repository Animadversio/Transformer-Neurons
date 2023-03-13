import torch
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from core.layer_hook_utils import featureFetcher_module
#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Initializing a model from the configuration
# model = GPT2Model(configuration) #Bug: this is not a pretrained model
model = GPT2Model.from_pretrained("gpt2")

# Accessing the model configuration
configuration = model.config
#%%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model_ver = "gpt2-large"  # "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_ver)
model = GPT2LMHeadModel.from_pretrained(model_ver)
#%%
text = "The vatican is in the city of"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
loss = outputs.loss
logits = outputs.logits
all_hidden = (outputs.hidden_states)
#%%
for li in range(len(all_hidden)):
    logits_lyr = model.lm_head(all_hidden[li])
    topvals, topidxs = torch.topk(logits_lyr[0, -1, ], 8)
    toptokens = tokenizer.convert_ids_to_tokens(topidxs)
    print(li, [tok.replace("Ġ","") for tok in toptokens])
    # print(li, [tok for tok in toptokens])
#%%
maxlogit, maxidx = logits.max(dim=-1)
#%%
tokenizer.decode(maxidx[0])
#%%
topvals, topidxs = torch.topk(logits[0,-1,], 10)
tokenizer.decode(topidxs.T)

# tokenizer.convert_ids_to_tokens(topidxs)

