"""
Understand BERT and GPT via gradient tricks
"""
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
# Initializing a model from the bert-base-uncased style configuration
# model = BertModel(configuration).from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# Accessing the model configuration
configuration = model.config
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.requires_grad_(False)
model.eval()
#%%
def get_causal_grad_hook(key):
    def causal_grad(module, input, output):
        """Substitute the output of certain layer as a fixed representation"""
        gradvar = torch.zeros_like(output[0].data, requires_grad=True)
        # output[0] + gradvar
        output[0].add_(gradvar)
        # output[0] = torch.autograd.Variable(output[0].data, requires_grad=True)
        gradvar_store[key] = gradvar
    return causal_grad


text = "Vatican is located in the city of [MASK]."
# text = "The Space Needle is located in downtown [MASK]."
gradvar_store = {}  # store the gradient variable, layer num as key
hook_hs = []
for layeri in range(12):
    target_module = model.bert.encoder.layer[layeri]
    hook_fun = get_causal_grad_hook(layeri)  # get_clamp_hook(fixed_hidden_states, fix_mask)
    hook_h = target_module.register_forward_hook(hook_fun)
    hook_hs.append(hook_h)

token_ids = tokenizer.encode(text, return_tensors='pt')
# with torch.no_grad():
outputs_clamp = model(token_ids, output_hidden_states=True)

for hook_h in hook_hs:
    hook_h.remove()

#%% Get the top token and its logit
mask_loc = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]
max_ids = torch.argmax(outputs_clamp.logits[0, mask_loc, :])
maxlogit = outputs_clamp.logits[0, mask_loc, max_ids]
print(f"Top token is '{tokenizer.convert_ids_to_tokens(max_ids.item())}', with logit {maxlogit.item():.3f}")
#%% Compute first order gradient
grad_maps = torch.autograd.grad(maxlogit,
                [*gradvar_store.values()], retain_graph=True,)  #  create_graph=True
grad_map_tsr = torch.cat(grad_maps, dim=0)
grad_map_layers = grad_map_tsr.norm(dim=-1) #torch.cat(grad_map_layers, dim=0)
#%% Compute 1st order gradient with 2nd order graph set up.
grad_maps_grad = torch.autograd.grad(maxlogit,
                [*gradvar_store.values()], retain_graph=True, create_graph=True)  #  create_graph=True
#%% Compute the 2nd order gradient to the hidden states.
hiddim = grad_maps_grad[0].size(-1)
seqdim = grad_maps_grad[0].size(1)
grad_maps_hess_all = []
for i in tqdm(range(seqdim)):
    # inspired by https://github.com/pytorch/pytorch/issues/7786#issuecomment-391612473
    grad_maps_hess_B = torch.autograd.grad(grad_maps_grad[0][0, i, :],
                gradvar_store[0], grad_outputs=torch.eye(768),  # this is a trick to avoid sum over outputs
                is_grads_batched=True, retain_graph=True, create_graph=False, )  #
    print(grad_maps_hess_B[0].shape)
    grad_maps_hess_all.append(grad_maps_hess_B[0])
grad_maps_hess_all = torch.stack(grad_maps_hess_all, dim=0)
print(grad_maps_hess_all.shape)
#  Obsolete, not efficient
# grad_maps_hess = []
# for i in tqdm(range(hiddim)):
#     grad_maps_part = torch.autograd.grad(grad_maps_grad[0][0, 1, i],
#                 gradvar_store[0], retain_graph=True, create_graph=False)  #
#     grad_maps_hess.append(grad_maps_part[0])
# grad_maps_hess = torch.stack(grad_maps_hess, dim=0)
#%% Visualize the first order gradient magnitude
tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
sns.heatmap(grad_map_layers.detach().numpy().T)
plt.gca().set_yticklabels(tokens, rotation=0)
plt.xlabel("Layer")
plt.title("Gradient L2 norm heat map")
plt.tight_layout()
plt.show()
#%%
""" Note on matrix norm order 
2: matrix max singlur value norm
"nuc": matrix sum singlur value / nuclear norm
"fro": matrix with L2 vector norm / frobenius norm
"""
matnorm_ord = 2
grad_hess_norm_map = torch.linalg.norm(
        grad_maps_hess_all[:, :, 0, :, :],
        dim=(1, 3), ord=matnorm_ord)
sns.heatmap(grad_hess_norm_map)
plt.axis('image')
plt.gca().set_xticklabels(tokens, rotation=90)
plt.gca().set_yticklabels(tokens, rotation=0)
plt.title(f'Matrix {matnorm_ord} norm Hessian of the logit of the top token')
plt.tight_layout()
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

