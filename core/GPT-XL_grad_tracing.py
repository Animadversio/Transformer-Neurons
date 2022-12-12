"""
Understand BERT and GPT via gradient tricks
"""
import os
import warnings
from os.path import join
import torch
import torch.nn as nn  # import ModuleList
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
# from transformers.models.bert.modeling_bert import BertModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertModel, BertForMaskedLM, BertConfig, BertTokenizer
from core.layer_hook_utils import featureFetcher_module
from core.interp_utils import top_tokens_based_on_activation, topk_decode
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
# Accessing the model configuration
configuration = model.config
model.requires_grad_(False)
model.eval()
#%%
def get_causal_grad_hook(key, gradvar_store):
    """ add req grad variable to trace gradient. """
    def causal_grad(module, input, output):
        """Substitute the output of certain layer as a fixed representation
        Note: Assume the first output variable is used in downstream computing.
        """
        gradvar = torch.zeros_like(output[0].data, requires_grad=True)
        # output[0] + gradvar
        output[0].add_(gradvar)
        # output[0] = torch.autograd.Variable(output[0].data, requires_grad=True)
        gradvar_store[key] = gradvar
    return causal_grad


def get_record_causal_grad_hook(key, gradvar_store, act_store):
    """Both record activation and add req grad variable to trace gradient. """
    def causal_grad(module, input, output):
        """Substitute the output of certain layer as a fixed representation
        Note: Assume the first output variable is used in downstream computing.
        """
        gradvar = torch.zeros_like(output[0].data, requires_grad=True)
        # output[0] + gradvar
        output[0].add_(gradvar)
        # output[0] = torch.autograd.Variable(output[0].data, requires_grad=True)
        gradvar_store[key] = gradvar
        act_store[key] = output[0].detach().data.cpu()
    return causal_grad


def forward_gradhook_vars(model, token_ids, module_str="block",
                          module_fun=None):
    if module_str == "block":
        module_fun = lambda model, layeri: model.transformer.h[layeri]
    elif module_str == "attn":
        module_fun = lambda model, layeri: model.transformer.h[layeri].attn
    elif module_str == "mlp":
        module_fun = lambda model, layeri: model.transformer.h[layeri].mlp
    elif module_str == "ln_2":
        module_fun = lambda model, layeri: model.transformer.h[layeri].ln_2
    else:
        if module_fun is None:
            raise NotImplementedError
        else:
            module_fun = module_fun
            warnings.warn("not recognized module_str, use the custom module function")
    gradvar_store = {}  # store the gradient variable, layer num as key
    hook_hs = []
    for layeri in range(len(model.transformer.h)):
        target_module = module_fun(model, layeri)  # model.transformer.h[layeri]
        hook_fun = get_causal_grad_hook(layeri, gradvar_store)  # get_clamp_hook(fixed_hidden_states, fix_mask)
        hook_h = target_module.register_forward_hook(hook_fun)
        hook_hs.append(hook_h)

    # with torch.no_grad():
    try:
        outputs = model(token_ids, output_hidden_states=True)
    except:
        # Need to clean up (remove hooks) or it will keep raising error.
        for hook_h in hook_hs:
            hook_h.remove()
        raise
    else:
        for hook_h in hook_hs:
            hook_h.remove()
        return outputs, gradvar_store


from easydict import EasyDict as edict
def forward_record_gradhook_vars_all(model, token_ids, ):
    module_funs = {
        "block": lambda model, layeri: model.transformer.h[layeri],
        "attn":  lambda model, layeri: model.transformer.h[layeri].attn,
        "mlp":   lambda model, layeri: model.transformer.h[layeri].mlp,
        "ln_2":  lambda model, layeri: model.transformer.h[layeri].ln_2,
    }
    gradvar_stores = {}  # store the gradient variable, layer num as key
    act_stores = {}
    for k in module_funs:
        gradvar_stores[k] = {}
        act_stores[k] = {}
    hook_hs = []
    for layeri in range(len(model.transformer.h)):
        for module_str, module_fun in module_funs.items():
            target_module = module_fun(model, layeri)  #model.transformer.h[layeri]  # model.transformer.h[layeri]
            hook_fun = get_record_causal_grad_hook(layeri, gradvar_stores[module_str], act_stores[module_str])  # get_clamp_hook(fixed_hidden_states, fix_mask)
            hook_h = target_module.register_forward_hook(hook_fun)
            hook_hs.append(hook_h)

    # with torch.no_grad():
    try:
        outputs = model(token_ids, output_hidden_states=True)
    except:
        # Need to remove hooks or it will keep raising error.
        for hook_h in hook_hs:
            hook_h.remove()
        raise
    else:
        for hook_h in hook_hs:
            hook_h.remove()
        return outputs, gradvar_stores, act_stores

savedir = r"F:\insilico_exps\GPT-XL_grad_trace"
#%%

text = "The official residence of the Pope, Vatican is located in the city of"
token_ids = tokenizer.encode(text, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
tokens = [t.replace("Ġ", "") for t in tokens]
text_label = "-".join(text.split(" ")[:10])
expdir = join(savedir, text_label)
os.makedirs(expdir, exist_ok=True)

outputs, gradvar_stores, act_stores = forward_record_gradhook_vars_all(model, token_ids, )
max_ids = torch.argmax(outputs.logits[0, -1, :])
maxlogit = outputs.logits[0, -1, max_ids]
print(text)
top_token = tokenizer.convert_ids_to_tokens(max_ids.item()).replace('\u0120','')
print(f"Top token is '{top_token}', with logit {maxlogit.item():.3f}")
for sub_module in ["block", "attn", "mlp", "ln_2"]: #
    act_store = [*act_stores[sub_module].values()]
    grad_maps = torch.autograd.grad(maxlogit,
                                    [*gradvar_stores[sub_module].values()], retain_graph=True, )  # create_graph=True
    if grad_maps[0].ndim == 3:  # "block", "attn",
        grad_map_tsr = torch.cat(grad_maps, dim=0)
        act_map_tsr = torch.cat(act_store, dim=0)
    elif grad_maps[0].ndim == 2:  # "mlp", "ln_2", the Batch dim is suppressed
        grad_map_tsr = torch.stack(grad_maps, dim=0)
        act_map_tsr = torch.stack(act_store, dim=0)
    else:
        raise ValueError
    grad_map_norm_layers = grad_map_tsr.norm(dim=-1)
    act_map_norm_layers = act_map_tsr.norm(dim=-1)

    sns.heatmap(grad_map_norm_layers.T)
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.xlabel("Layer")
    plt.title(f"Gradient L2 norm heat map ({sub_module})")
    plt.tight_layout()
    plt.savefig(join(expdir, f'grad_norm_map_layers_{sub_module}.png'))
    plt.savefig(join(expdir, f'grad_norm_map_layers_{sub_module}.pdf'))
    plt.show()

    sns.heatmap(act_map_norm_layers.T)
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.xlabel("Layer")
    plt.title(f"Activation L2 norm heat map ({sub_module})")
    plt.tight_layout()
    plt.savefig(join(expdir, f'act_norm_map_layers_{sub_module}.png'))
    plt.savefig(join(expdir, f'act_norm_map_layers_{sub_module}.pdf'))
    plt.show()

    sns.heatmap(grad_map_norm_layers.T / act_map_norm_layers.T)
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.xlabel("Layer")
    plt.title(f"Grad norm / Activation norm raio heat map ({sub_module})")
    plt.tight_layout()
    plt.savefig(join(expdir, f'grad_act_ratio_map_layers_{sub_module}.png'))
    plt.savefig(join(expdir, f'grad_act_ratio_map_layers_{sub_module}.pdf'))
    plt.show()
#%%
savedir = r"F:\insilico_exps\GPT-XL_grad_trace"

text = "The Space Needle is located in downtown"
text = "On Wednesday, Peter just learned from class that, the Space Needle is located in downtown"
# text = "Vatican is located in the city of"
text = "The official residence of the Pope, Vatican is located in the city of"
# text = "The Apostolic Palace, the official residence of the Pope, is located in the city of"
# text = "Michael Jackson, one of the most significant cultural figures, was born in"
# text = "Brian De Palma works in the area of"
# text = "The headquarter of Zillow is in downtown"
# text = "Lawrence Taylor professionally plays the sport of"
# text = "Germaine Greer's domain of work is"
token_ids = tokenizer.encode(text, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
tokens = [t.replace("Ġ", "") for t in tokens]
text_label = "-".join(text.split(" ")[:3])
expdir = join(savedir, text_label)
os.makedirs(expdir, exist_ok=True)
print("Experiment dir: ", expdir)
for sub_module in ["block", "attn", "mlp", "ln_2"]:
    outputs, gradvar_store = forward_gradhook_vars(model, token_ids, module_str=sub_module)
    #% Get the top token and its logit
    max_ids = torch.argmax(outputs.logits[0, -1, :])
    maxlogit = outputs.logits[0, -1, max_ids]
    print(text)
    top_token = tokenizer.convert_ids_to_tokens(max_ids.item()).replace('\u0120','')
    print(f"Top token is '{top_token}', with logit {maxlogit.item():.3f}")
    # torch.save(outputs, join(expdir, "model_outputs_hiddens.pt"))
    #%% Compute first order gradient
    grad_maps = torch.autograd.grad(maxlogit,
                    [*gradvar_store.values()], retain_graph=True,)  #  create_graph=True
    if grad_maps[0].ndim == 3:
        grad_map_tsr = torch.cat(grad_maps, dim=0)
    elif grad_maps[0].ndim == 2:
        grad_map_tsr = torch.stack(grad_maps, dim=0)
    else:
        raise ValueError
    grad_map_norm_layers = grad_map_tsr.norm(dim=-1)  # torch.cat(grad_map_layers, dim=0)
    torch.save(grad_map_tsr, join(expdir, f"grad_map_tsr_{sub_module}.pt"))
    # torch.save(grad_map_norm_layers, join(expdir, f"grad_norm_map_tsr_{sub_module}.pt"))
    #% Visualize the first order gradient magnitude
    sns.heatmap(grad_map_norm_layers.T)
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.xlabel("Layer")
    plt.title(f"Gradient L2 norm heat map ({sub_module})")
    plt.tight_layout()
    plt.savefig(join(expdir, f'grad_norm_map_layers_{sub_module}.png'))
    plt.savefig(join(expdir, f'grad_norm_map_layers_{sub_module}.pdf'))
    plt.show()
#%% Compute 1st order gradient with 2nd order graph set up.
# grad_maps_grad = torch.autograd.grad(maxlogit,
#                 [*gradvar_store.values()], retain_graph=True, create_graph=True)  #  create_graph=True
#%%
import json
k1000_data = json.load(open("dataset\\known_1000.json", 'r'))
#%%
import matplotlib
matplotlib.use("Agg")
#%% 990
for i in range(990,991):#range(991, len(k1000_data)):
    text = k1000_data[i]['prompt']
    token_ids = tokenizer.encode(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
    tokens = [t.replace("Ġ", "") for t in tokens]
    text_label = "-".join(text.split(" ")[:4])
    expdir = join(savedir, "fact%04d-" % (i) + text_label)
    os.makedirs(expdir, exist_ok=True)

    outputs, gradvar_stores, act_stores = forward_record_gradhook_vars_all(model, token_ids, )
    max_ids = torch.argmax(outputs.logits[0, -1, :])
    maxlogit = outputs.logits[0, -1, max_ids]
    print(text)
    top_token = tokenizer.convert_ids_to_tokens(max_ids.item()).replace('\u0120','')
    print(f"Top token is '{top_token}', with logit {maxlogit.item():.3f}")
    gradsavedict = {}
    actsavedict = {}
    gradmapsavedict = {}
    actmapsavedict = {}
    for sub_module in ["block", "attn", "mlp", "ln_2"]:
        act_store = [ * act_stores[sub_module].values()]
        grad_maps = torch.autograd.grad(maxlogit,
                                        [*gradvar_stores[sub_module].values()], retain_graph=True, )  # create_graph=True
        if grad_maps[0].ndim == 3:
            grad_map_tsr = torch.cat(grad_maps, dim=0)
            act_map_tsr = torch.cat(act_store, dim=0)
        elif grad_maps[0].ndim == 2:
            grad_map_tsr = torch.stack(grad_maps, dim=0)
            act_map_tsr = torch.stack(act_store, dim=0)
        else:
            raise ValueError
        gradsavedict[sub_module] = grad_map_tsr
        actsavedict[sub_module] = act_map_tsr
        grad_map_norm_layers = grad_map_tsr.norm(dim=-1)
        act_map_norm_layers = act_map_tsr.norm(dim=-1)
        gradmapsavedict[sub_module] = grad_map_norm_layers
        actmapsavedict[sub_module]  = act_map_norm_layers
        plt.figure()
        sns.heatmap(grad_map_norm_layers.T)
        plt.gca().set_yticks(0.5 + np.arange(len(tokens)))
        plt.gca().set_yticklabels(tokens, rotation=0)
        plt.xlabel("Layer")
        plt.title(f"Gradient L2 norm heat map ({sub_module})")
        plt.tight_layout()
        plt.savefig(join(expdir, f'grad_norm_map_layers_{sub_module}.png'))
        plt.savefig(join(expdir, f'grad_norm_map_layers_{sub_module}.pdf'))
        plt.show()

        plt.figure()
        sns.heatmap(act_map_norm_layers.T)
        plt.gca().set_yticks(0.5 + np.arange(len(tokens)))
        plt.gca().set_yticklabels(tokens, rotation=0)
        plt.xlabel("Layer")
        plt.title(f"Activation L2 norm heat map ({sub_module})")
        plt.tight_layout()
        plt.savefig(join(expdir, f'act_norm_map_layers_{sub_module}.png'))
        plt.savefig(join(expdir, f'act_norm_map_layers_{sub_module}.pdf'))
        plt.show()

        plt.figure()
        sns.heatmap(grad_map_norm_layers.T / act_map_norm_layers.T)
        plt.gca().set_yticks(0.5 + np.arange(len(tokens)))
        plt.gca().set_yticklabels(tokens, rotation=0)
        plt.xlabel("Layer")
        plt.title(f"Grad norm / Activation norm raio heat map ({sub_module})")
        plt.tight_layout()
        plt.savefig(join(expdir, f'grad_act_ratio_map_layers_{sub_module}.png'))
        plt.savefig(join(expdir, f'grad_act_ratio_map_layers_{sub_module}.pdf'))
        plt.show()

    # torch.save(savedict, join(expdir, f"grad_act_tsr_save.pt"))
    # torch.save(mapsavedict, join(expdir, f"grad_act_map_save.pt"))
    torch.save({"grad" : gradsavedict,
                "act"  : actsavedict,},
               join(expdir, f"grad_act_save.pt"))
    torch.save({"gradmap": gradmapsavedict,
                "actmap": actmapsavedict},
               join(expdir, f"grad_act_map_save.pt"))
    json.dump(k1000_data[i], open(join(expdir,f"knowledge_entry.json"), "w"))
