#%%
import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from core.plot_utils import saveallforms, join
from core.causal_tracing_lib import causal_trace_pipeline, visualize_causal_trace
from core.text_token_utils import find_substr_loc
from transformers import GPT2Tokenizer, GPT2LMHeadModel
figroot = r"F:\insilico_exps\causal_traces"
os.makedirs(figroot, exist_ok=True)
device = "cuda"
#%%
probe_set = [
    ("The Space Needle is located in downtown", "The Space Needle"),
    ("Vatican is located in the city of", "Vatican"),
    ("The Eiffel Tower is located in", "The Eiffel Tower"),
    ("The headquarter of Zillow is in downtown", "The headquarter of Zillow"),
    ("Otto von Bismarck worked in the city of", "Otto von Bismarck"),
    ("Claude d'Annebault is a citizen of", "Claude d'Annebault"),
    ("Iron Man is affiliated with", "Iron Man"),
]
#%%
"""GPT2 type models"""
#%%
for model_str in ["gpt2-medium", ]: # "gpt2", "gpt2-large", "gpt2-xl",
    tokenizer = GPT2Tokenizer.from_pretrained(model_str)
    model = GPT2LMHeadModel.from_pretrained(model_str)
    model.requires_grad_(False)
    model.eval().to(device)
    noise_std = 3 * model.transformer.wte.weight.std().item()
    print(model_str, "Noise scale=", noise_std)
    figname = ""
    for text, subj_str in probe_set:
        figdir = join(figroot, model_str, text.replace(" ", "_")[0:25])
        os.makedirs(figdir, exist_ok=True)
        subj_span = find_substr_loc(tokenizer, text, subj_str)
        subject_loc = [*range(subj_span[0], subj_span[1])]
        df, degrade_map, clamp_map, degrprob_map, clamprob_map = \
            causal_trace_pipeline(text, model, tokenizer, subject_loc,
                                  noise_batch=10, noise_std=noise_std, device=device)
        df.to_csv(join(figdir, "causal_trace_data.csv"))
        visualize_causal_trace(text, tokenizer, df, figname, figdir=figdir, subject_loc=subject_loc)

#%%
import json
k1000_data = json.load(open("dataset\\known_1000.json", 'r'))
for model_str in ["gpt2-xl", ]:  # "gpt2", "gpt2-medium", "gpt2-large",
    tokenizer = GPT2Tokenizer.from_pretrained(model_str)
    model = GPT2LMHeadModel.from_pretrained(model_str)
    model.requires_grad_(False)
    model.eval().to(device)
    noise_std = 3 * model.transformer.wte.weight.std().item()
    print(model_str, "Noise scale=", noise_std)
    figname = ""
    for i in tqdm(range(0, len(k1000_data))):
        text = k1000_data[i]['prompt']
        subj_str = k1000_data[i]['subject']
        figdir = join(figroot, model_str, "fact%04d-"%(i)+text.replace(" ", "_")[0:25])
        os.makedirs(figdir, exist_ok=True)
        subj_span = find_substr_loc(tokenizer, text, subj_str)
        subject_loc = [*range(subj_span[0], subj_span[1])]
        df, degrade_map, clamp_map, degrprob_map, clamprob_map = \
            causal_trace_pipeline(text, model, tokenizer, subject_loc,
                                  noise_batch=10, noise_std=noise_std, device=device)
        df.to_csv(join(figdir, "causal_trace_data.csv"))
        visualize_causal_trace(text, tokenizer, df, figname, figdir=figdir, subject_loc=subject_loc)

#%%
"""BERT type models"""
from core.text_token_utils import find_substr_loc
from transformers import BertTokenizer, BertForMaskedLM
from core.causal_tracing_lib import causal_trace_pipeline_bert, embed_hidden_clean_bert
#%%
device = "cuda"
#%%
for model_str in ["bert-base", "bert-large" ]:
    tokenizer = BertTokenizer.from_pretrained(model_str+'-uncased')
    model = BertForMaskedLM.from_pretrained(model_str+'-uncased')
    model.requires_grad_(False)
    model.eval().to(device)
    noise_std = 3 * model.bert.embeddings.word_embeddings.weight.std().item()
    print(model_str, "Noise scale=", noise_std)
    figname = ""
    for text, subj_str in probe_set:
        text = text + " [MASK]."
        figdir = join(figroot, model_str, text.replace(" ", "_")[0:25])
        os.makedirs(figdir, exist_ok=True)
        subj_span = find_substr_loc(tokenizer, text, subj_str, uncase=True)
        subject_loc = [*range(subj_span[0], subj_span[1])]
        df, degrade_map, clamp_map, degrprob_map, clamprob_map = \
            causal_trace_pipeline_bert(text, model, tokenizer, subject_loc,
                                       noise_batch=10, noise_std=noise_std, device=device,)
        df.to_csv(join(figdir, "causal_trace_data.csv"))
        visualize_causal_trace(text, tokenizer, df, figname, figdir=figdir, subject_loc=subject_loc)


#%% Dev zone
#%%
model_str = "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
#%%
model.requires_grad_(False)
model.eval().to(device)
#%%
# text, subj_str = "Vatican is located in the city of", "Vatican"
text, subj_str = "The Space Needle is located in downtown", "The Space Needle"
figdir = join(figroot, model_str, text.replace(" ", "_")[0:25])
os.makedirs(figdir, exist_ok=True)
subj_span = find_substr_loc(tokenizer, text, subj_str)
subject_loc = [*range(subj_span[0], subj_span[1])]
df, degrade_map, clamp_map, degrprob_map, clamprob_map = \
    causal_trace_pipeline(text, model, tokenizer, subject_loc,
                          noise_batch=10, noise_std=0.15, device=device)
df.to_csv(join(figdir, "causal_trace_data.csv"))
visualize_causal_trace(text, tokenizer, df, "", figdir=figdir, subject_loc=subject_loc)
#%%

model_str = "BERT-base"
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model_str = "BERT-large"
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained("bert-large-uncased")
configuration = model.config
model.requires_grad_(False)
model.eval().to(device)
#%%
noise_std = 3 * model.bert.embeddings.word_embeddings.weight.std().item()
print(model_str, "Noise scale=", noise_std)
#%%
text, subj_str = "The Space Needle is located in downtown [MASK].", "The Space Needle"
# text, subj_str = "Vatican is located in the city of [MASK].", "Vatican"
figdir = join(figroot, model_str, text.replace(" ", "_")[0:25])
os.makedirs(figdir, exist_ok=True)
figname = ""

subj_span = find_substr_loc(tokenizer, text, subj_str, uncase=True)
subject_loc = [*range(subj_span[0], subj_span[1])]
df, degrade_map, clamp_map, degrprob_map, clamprob_map = \
    causal_trace_pipeline_bert(text, model, tokenizer, subject_loc,
                               noise_batch=10, noise_std=noise_std, device=device,)
df.to_csv(join(figdir, "causal_trace_data.csv"))
visualize_causal_trace(text, tokenizer, df, figname, figdir=figdir, subject_loc=subject_loc)