"""
Loading and basic analysis of the wikitext corpus.
"""
from datasets import load_dataset
# dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
#%%
dataset_len = dataset.filter(lambda x: len(x['text']) > 20)
#%%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
# Accessing the model configuration
configuration = model.config
model.requires_grad_(False)
model.eval()
#%%
import torch
from core.layer_hook_utils import featureFetcher_module
def model_forward_for_text(text, model, tokenizer, ):
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens)

    return tokens, outputs


def record_activations_for_text(text, model, tokenizer, layer_num=11):
    fetcher = featureFetcher_module()
    if type(layer_num) not in [tuple, list, range]:
        layer_num = [layer_num]
    for layer_i in layer_num:
        fetcher.record_module(model.transformer.h[layer_i].attn, f"GPT2_B{layer_i}_attn")
        fetcher.record_module(model.transformer.h[layer_i].mlp.act, f"GPT2_B{layer_i}_act")
        fetcher.record_module(model.transformer.h[layer_i].mlp.c_fc, f"GPT2_B{layer_i}_fc")
        fetcher.record_module(model.transformer.h[layer_i].mlp.c_proj, f"GPT2_B{layer_i}_proj")
        fetcher.record_module(model.transformer.h[layer_i], f"GPT2_B{layer_i}")
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens)
    fetcher.cleanup()
    for k, v in fetcher.activations.items():
        print(k, v.shape)
    return tokens, outputs, fetcher
#%%
from tqdm import tqdm
act_storage = {}
#%% Record the GPT response
for i in tqdm(range(100, 200)):
    text = dataset_len["train"][i]["text"]
    tokens, _, fetcher = record_activations_for_text(text, model, tokenizer, layer_num=range(48))
    for k in fetcher.activations:
        if k in act_storage:
            act_storage[k].append(fetcher[k]) #= torch.cat((act_storage[k], fetcher[k]), dim=1)
        else:
            act_storage[k] = [fetcher[k]]

#%%
for k in act_storage:
    act_storage[k] = torch.cat(act_storage[k], dim=1)
#%%
from os.path import join
datadir = r"F:\insilico_exps\GPT-XL_grad_trace"
torch.save(act_storage, join(datadir, "activation_storage.pt"))
#%%
