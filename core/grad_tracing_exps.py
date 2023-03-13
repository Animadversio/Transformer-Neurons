
import torch
from core.grad_tracing_lib import grad_tracing, visualize_grad_trace
from core.plot_utils import saveallforms, join
from core.causal_tracing_lib import causal_trace_pipeline, visualize_causal_trace
from core.text_token_utils import find_substr_loc
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

figroot = r"F:\insilico_exps\grad_traces"
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
for model_str in ["gpt2-medium", ]: # "gpt2", "gpt2-large", "gpt2-xl",
    tokenizer = GPT2Tokenizer.from_pretrained(model_str)
    model = GPT2LMHeadModel.from_pretrained(model_str)
    model.requires_grad_(False)
    model.eval().to(device)
    for text, subj_str in probe_set:
        figdir = join(figroot, model_str, text.replace(" ", "_")[0:25])
        os.makedirs(figdir, exist_ok=True)
        gradsavedict, actsavedict, gradmapsavedict, actmapsavedict = \
            grad_tracing(text, tokenizer, model, device=device, )
        visualize_grad_trace(text, tokenizer, gradsavedict, actsavedict, expdir=figdir, )
        torch.save({"grad": gradsavedict, "act": actsavedict, },
                   join(figdir, f"grad_act_save.pt"))

#%%