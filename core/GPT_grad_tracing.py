"""
Understand BERT and GPT via gradient tricks
"""
import os
from os.path import join
import torch
import torch.nn as nn  # import ModuleList
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from core.interp_utils import topk_decode
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertModel, BertForMaskedLM, BertConfig, BertTokenizer
from core.layer_hook_utils import featureFetcher_module
from core.interp_utils import top_tokens_based_on_activation
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Initializing a model from the bert-base-uncased style configuration
# model = BertModel(configuration).from_pretrained('bert-base-uncased')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.requires_grad_(False)
model.eval()
#%%
savedir = r"F:\insilico_exps\GPT_hessian"
def get_causal_grad_hook(key, gradvar_store):
    def causal_grad(module, input, output):
        """Substitute the output of certain layer as a fixed representation"""
        gradvar = torch.zeros_like(output[0].data, requires_grad=True)
        # output[0] + gradvar
        output[0].add_(gradvar)
        # output[0] = torch.autograd.Variable(output[0].data, requires_grad=True)
        gradvar_store[key] = gradvar
    return causal_grad


def forward_gradhook_vars(model, token_ids):
    gradvar_store = {}  # store the gradient variable, layer num as key
    hook_hs = []
    for layeri in range(12):
        target_module = model.transformer.h[layeri]
        hook_fun = get_causal_grad_hook(layeri, gradvar_store)  # get_clamp_hook(fixed_hidden_states, fix_mask)
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
        return outputs, gradvar_store


text = "The Space Needle is located in downtown"
# text = "Vatican is located in the city of"
token_ids = tokenizer.encode(text, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
tokens = [t.replace("Ġ", "") for t in tokens]
text_label = "-".join(text.split(" ")[:3])
expdir = join(savedir, text_label)
os.makedirs(expdir, exist_ok=True)
print("Experiment dir: ", expdir)
outputs, gradvar_store = forward_gradhook_vars(model, token_ids)
#% Get the top token and its logit
max_ids = torch.argmax(outputs.logits[0, -1, :])
maxlogit = outputs.logits[0, -1, max_ids]
print(text)
top_token = tokenizer.convert_ids_to_tokens(max_ids.item()).replace('\u0120','')
print(f"Top token is '{top_token}', with logit {maxlogit.item():.3f}")
torch.save(outputs, join(expdir, "model_outputs_hiddens.pt"))
#%% Compute first order gradient
grad_maps = torch.autograd.grad(maxlogit,
                [*gradvar_store.values()], retain_graph=True,)  #  create_graph=True
grad_map_tsr = torch.cat(grad_maps, dim=0)
grad_map_norm_layers = grad_map_tsr.norm(dim=-1)  # torch.cat(grad_map_layers, dim=0)
torch.save(grad_map_tsr, join(expdir, "grad_map_tsr.pt"))
#% Visualize the first order gradient magnitude
sns.heatmap(grad_map_norm_layers.T)
plt.gca().set_yticklabels(tokens, rotation=0)
plt.xlabel("Layer")
plt.title("Gradient L2 norm heat map")
plt.tight_layout()
plt.savefig(join(expdir, f'grad_norm_map_layers.png'))
plt.savefig(join(expdir, f'grad_norm_map_layers.pdf'))
plt.show()
#%% Compute 1st order gradient with 2nd order graph set up.
grad_maps_grad = torch.autograd.grad(maxlogit,
                [*gradvar_store.values()], retain_graph=True, create_graph=True)  #  create_graph=True
#%% Compute the 2nd order gradient to the hidden states.
Hlayer = 0
for Hlayer in range(12):
    hiddim = grad_maps_grad[Hlayer].size(-1)
    seqdim = grad_maps_grad[Hlayer].size(1)
    grad_maps_hess_all = []
    for i in tqdm(range(seqdim)):
        # inspired by
        # https://github.com/pytorch/pytorch/issues/7786#issuecomment-391612473
        grad_maps_hess_B = torch.autograd.grad(grad_maps_grad[Hlayer][0, i, :],
                    gradvar_store[Hlayer], grad_outputs=torch.eye(hiddim),  # this is a trick to avoid sum over outputs
                    is_grads_batched=True, retain_graph=True, create_graph=False, )  #
        print(grad_maps_hess_B[0].shape)
        grad_maps_hess_all.append(grad_maps_hess_B[0])
    grad_maps_hess_all = torch.stack(grad_maps_hess_all, dim=0)
    print(grad_maps_hess_all.shape)
    torch.save(grad_maps_hess_all, join(expdir, f"hessian_map_tsr_layer{Hlayer}.pt"))
    #%%
    """ Note on matrix norm order 
    2: matrix max singlur value norm
    "nuc": matrix sum singlur value / nuclear norm
    "fro": matrix with L2 vector norm / frobenius norm
    """
    matnorm_ord = 2
    for matnorm_ord in [2, "nuc", "fro"]:
        grad_hess_norm_map = torch.linalg.norm(
                grad_maps_hess_all[:, :, 0, :, :],
                dim=(1, 3), ord=matnorm_ord)
        sns.heatmap(grad_hess_norm_map)
        plt.axis('image')
        plt.gca().set_xticklabels(tokens, rotation=90)
        plt.gca().set_yticklabels(tokens, rotation=0)
        plt.title(f'Matrix {matnorm_ord} norm of Hessian of the logit of the top token')
        plt.tight_layout()
        plt.savefig(join(expdir, f'grad_hess_{matnorm_ord}_norm_map_layer{Hlayer}.png'))
        plt.savefig(join(expdir, f'grad_hess_{matnorm_ord}_norm_map_layer{Hlayer}.pdf'))
        plt.show()
#%%
sns.heatmap(torch.linalg.norm(
            grad_maps_hess_all[:, :, 0, :, :],
            dim=(1, 3), ord="fro"))
plt.gca().set_xticklabels(tokens, rotation=90)
plt.gca().set_yticklabels(tokens, rotation=0)
plt.axis('image')
plt.show()
#%%
sns.heatmap(grad_map_norm_layers[10][:, None] @
            grad_map_norm_layers[10][None, :])
plt.gca().set_xticklabels(tokens, rotation=90)
plt.gca().set_yticklabels(tokens, rotation=0)
plt.axis('image')
plt.show()
#%%
print(grad_maps_hess)
sns.heatmap(grad_maps_hess.detach().numpy()[:,0,1,:])
plt.show()
#%%
U, S, V = torch.svd(grad_maps_hess[:, 0, 1, :], )
print(torch.cumsum(S[:50], 0) / S.sum())
#%%
U, S, V = torch.svd(grad_maps_hess[:, 0, 2, :], )
print(torch.cumsum(S[:50], 0) / S.sum())

#%%

