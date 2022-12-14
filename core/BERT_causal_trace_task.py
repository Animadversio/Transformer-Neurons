"""What if we clamp the hidden output of one or more tokens to the initial ones?
How will that affects the mask prediction results
"""

import torch
import matplotlib.pyplot as plt
from transformers import BertModel, BertForMaskedLM, BertConfig, BertTokenizer
from core.layer_hook_utils import featureFetcher_module
from core.interp_utils import top_tokens_based_on_activation
# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()
# Initializing a model from the bert-base-uncased style configuration
# model = BertModel(configuration) # BUG not trained
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# model = BertModel(configuration).from_pretrained('bert-base-uncased')
# Accessing the model configuration
configuration = model.config
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.requires_grad_(False)
model.eval()
#%%
from copy import deepcopy
import torch.nn as nn  # import ModuleList
from core.interp_utils import topk_decode
from transformers.models.bert.modeling_bert import BertModel
def top_k_decode_layers(model, tokenizer, hidden_states, read_loc, k=8):
    for layeri in range(len(hidden_states)):
        logits = model.cls(hidden_states[layeri])  #
        toks, vals = topk_decode(logits[0, read_loc, :], tokenizer, k)
        print(f"Layer {layeri:02d}", toks[0], )

#%%
text = "Vatican is located in the city of [MASK]."
# text = "The forbidden palace is located in the city of [MASK]."
# text = "New York is the [MASK] of the US."
# text = "Vatican is located on the [MASK] part of Italy."
text = "The [MASK] is located in downtown Seattle."
text = "The [MASK] is located in the city of Rome."
token_ids = tokenizer.encode(text, return_tensors='pt')
mask_loc = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]
with torch.no_grad():
    outputs = model(token_ids, output_hidden_states=True)

#% Layer output across the layers
# print("Layer order: ", layer_order.tolist())
k = 8
print(f"Decoded top {k} tokens for the masked word across the layers")
# mask_loc = token_ids[0].tolist().index(tokenizer.mask_token_id)
top_k_decode_layers(model, tokenizer, outputs.hidden_states, mask_loc, k=8)
#%%
#%% Similarity of the hidden states at layer i with the input .
cossim_mat = []
for layeri in range(13):
    cossim = torch.cosine_similarity(outputs.hidden_states[0],
                            outputs.hidden_states[layeri],
                            dim=2)
    cossim_mat.append(cossim)
cossim_mat = torch.cat(cossim_mat, dim=0)
print(cossim_mat)
#%%
cossim_mat = []
for layeri in range(13):
    cossim = torch.cosine_similarity(outputs.hidden_states[-1],
                            outputs.hidden_states[layeri],
                            dim=2)
    cossim_mat.append(cossim)
cossim_mat = torch.cat(cossim_mat, dim=0)
print(cossim_mat)
#%%
outputs.hidden_states[11].norm(dim=2)
#%%

def probe_hook(module, input, output):
    print(type(output))
    print(len(output))
    print(output[0].data.shape)


def subst_bertlayer_hook(module, input, output):
    """Substitute the output of certain layer as a fixed representation"""
    # print(type(output))
    # print(len(output))
    # print(output[0].data.shape)
    output[0].data[fix_mask] = fixed_hidden_states[fix_mask]


text = "Vatican is located in the city of [MASK]."
text = "The forbidden palace is located in the city of [MASK]."
text = "Vatican is located on the [MASK] part of Italy."
token_ids = tokenizer.encode(text, return_tensors='pt')
with torch.no_grad():
    outputs_free = model(token_ids, output_hidden_states=True)

mask_tok_loc = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]
free_loc = mask_tok_loc.item()

fixed_hidden_states = outputs_free.hidden_states[0]
fix_mask = torch.ones(fixed_hidden_states.shape, dtype=torch.bool)
fix_mask[:, mask_tok_loc.item(), :] = 0
# target_module = model.bert.encoder.layer[0]
hook_hs = []
for target_module in model.bert.encoder.layer:
    hook_fun = subst_bertlayer_hook
    hook_h = target_module.register_forward_hook(hook_fun)
    hook_hs.append(hook_h)

with torch.no_grad():
    outputs_clamp = model(token_ids, output_hidden_states=True)

for hook_h in hook_hs:
    hook_h.remove()

assert not torch.allclose(outputs_clamp.hidden_states[11][0, mask_tok_loc, :],
                      outputs_clamp.hidden_states[0][0, mask_tok_loc, :], )
assert torch.allclose(outputs_clamp.hidden_states[11][0, 5, :],
                      outputs_clamp.hidden_states[0][0, 5, :], )
#%%
print("BERT without clamp")
top_k_decode_layers(model, tokenizer, outputs_free.hidden_states, mask_tok_loc, k=8)
print("BERT with hidden states clamp")
top_k_decode_layers(model, tokenizer, outputs_clamp.hidden_states, mask_tok_loc, k=8)
#%%
from torch.nn.functional import softmax
def get_clamp_hook(fixed_hidden_states, fix_mask):
    def clamp_hidden_hook(module, input, output):
        """Substitute the output of certain layer as a fixed representation"""
        output[0].data[fix_mask] = fixed_hidden_states[fix_mask]
    return clamp_hidden_hook


def bert_degenerate_cmp(model, tokenizer, text,
                        clamp_layer, clamp_token_loc,
                        noise_loc, noise_std=0.01, noise_batch=1):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    mask_tok_loc = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]
    # token_ids,
    inputs_embeds = model.bert.embeddings.word_embeddings(token_ids)
    with torch.no_grad():
        outputs_free = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    """
    Add noise to the input word embeddings at the noise_loc
    """
    inputs_embeds_degrade = inputs_embeds.repeat(noise_batch, 1, 1)
    inputs_embeds_degrade[:, noise_loc, :] += torch.randn_like(inputs_embeds_degrade[:, noise_loc, :]) * noise_std
    with torch.no_grad():
        outputs_degrade = model(inputs_embeds=inputs_embeds_degrade, output_hidden_states=True)
    """
    Clamp the hidden states at the clamp_layer at the clamp_token_loc to be the 
        same as the hidden states of outputs_free. 
    """
    hook_hs = []
    for layeri in clamp_layer:
        target_module = model.bert.encoder.layer[layeri]
        fixed_hidden_states = outputs_free.hidden_states[layeri+1].expand(noise_batch, -1, -1)
        fix_mask = torch.zeros(fixed_hidden_states.shape, dtype=torch.bool)
        fix_mask[:, clamp_token_loc, :] = 1
        hook_fun = get_clamp_hook(fixed_hidden_states, fix_mask)
        hook_h = target_module.register_forward_hook(hook_fun)
        hook_hs.append(hook_h)

    with torch.no_grad():
        outputs_clamp = model(inputs_embeds=inputs_embeds_degrade, output_hidden_states=True)

    for hook_h in hook_hs:
        hook_h.remove()

    logits_free = model.cls(outputs_free.hidden_states[-1][:, mask_tok_loc, :])  #
    maxval, maxids = torch.topk(logits_free, 1, dim=-1)
    maxids = maxids.item()
    # print(maxids, maxval)
    # print(logits_free.shape)
    print("Free", tokenizer.decode([maxids]), f"({maxids}) logprob {maxval[0].item():.3f}", )
    logits_degrade = model.cls(outputs_degrade.hidden_states[-1][:, mask_tok_loc, :])  #
    print(f"Degrade {logits_degrade[:, 0, maxids].mean().item():.3f}", )
    logits_clamp = model.cls(outputs_clamp.hidden_states[-1][:, mask_tok_loc, :])  #
    print(f"Clamp {logits_clamp[:, 0, maxids].mean().item():.3f}", )
    # print("BERT without clamp")
    # top_k_decode_layers(model, tokenizer, outputs_free.hidden_states, mask_tok_loc, k=8)
    # print(f"BERT with input noise at {noise_loc} and std {noise_std}")
    # top_k_decode_layers(model, tokenizer, outputs_degrade.hidden_states, mask_tok_loc, k=8)
    # print(f"BERT with hidden states clamp at {clamp_layer} layer, {clamp_token_loc} loc")
    # top_k_decode_layers(model, tokenizer, outputs_clamp.hidden_states, mask_tok_loc, k=8)
    return outputs_free, outputs_degrade, outputs_clamp, \
           logits_free, logits_degrade, logits_clamp

# model.cpu()
# text = "Vatican is located in the city of [MASK]."
text = "The Space Needle is located in downtown [MASK]."
token_ids = tokenizer.encode(text)
normal_locs = [loc for loc, idx in enumerate(token_ids)
                if not idx in tokenizer.all_special_ids]
data_dict = {}
for clamp_layer in range(12):
    for clamp_token_loc in normal_locs:
        outputs_clean, outputs_noise, outputs_clamp, \
        logits_clean, logits_degrade, logits_clamp = bert_degenerate_cmp(model, tokenizer, text,
                            clamp_layer=[clamp_layer], clamp_token_loc=[clamp_token_loc, ],
                            noise_loc=[1, 2, 3], noise_std=0.2, noise_batch=10)
        maxidx = logits_clean.argmax()
        prob_clean = softmax(logits_clean, dim=-1)
        prob_degrade = softmax(logits_degrade, dim=-1)
        prob_clamp = softmax(logits_clamp, dim=-1)
        data_dict[(clamp_layer, clamp_token_loc)] = {"degrade": logits_degrade[:, 0, maxidx].mean().item(),
                                                    "clamp": logits_clamp[:, 0, maxidx].mean().item(),
                                                    "clean": logits_clean[:, 0, maxidx].mean().item(),
                                                     "prob_clean": prob_clean[:, 0, maxidx].mean().item(),
                                                     "prob_degrade": prob_degrade[:, 0, maxidx].mean().item(),
                                                     "prob_clamp": prob_clamp[:, 0, maxidx].mean().item(),}

#%%
import numpy as np
import seaborn as sns
import pandas as pd
df = pd.DataFrame(data_dict).T #.plot()
df.reset_index(inplace=True)
df.rename(columns={"level_0": "layer", "level_1": "position"}, inplace=True)
degrade_map = df.pivot(index="layer", columns="position", values="degrade") # .plot()
clamp_map = df.pivot(index="layer", columns="position", values="clamp")
degrprob_map = df.pivot(index="layer", columns="position", values="prob_degrade") # .plot()
clamprob_map = df.pivot(index="layer", columns="position", values="prob_clamp")

#%%
normal_tokens = tokenizer.convert_ids_to_tokens(
    [token_ids[loc] for loc in normal_locs])
# sns.heatmap(degrade_map, annot=True, fmt=".1f")
# sns.heatmap(clamp_map, annot=True, fmt=".1f")
plt.figure(figsize=(7, 4.5))
sns.heatmap((clamp_map - degrade_map).T, annot=True, fmt=".1f")
plt.axis("image")
plt.yticks(ticks=0.5 + np.arange(len(normal_tokens)),
           labels=normal_tokens, )
plt.tight_layout()
plt.title("logits difference: clamp - degrade")
plt.savefig(f"figs/clamp_degrade_diff_logit_{text.replace(' ','-')}.pdf")
plt.savefig(f"figs/clamp_degrade_diff_logit_{text.replace(' ','-')}.png")
plt.show()

plt.figure(figsize=(7, 4.5))
sns.heatmap((clamprob_map - degrprob_map).T, annot=True, fmt=".1f")
plt.axis("image")
plt.yticks(ticks=0.5 + np.arange(len(normal_tokens)),
           labels=normal_tokens, )
plt.tight_layout()
plt.title("prob difference: clamp - degrade")
plt.savefig(f"figs/clamp_degrade_diff_prob_{text.replace(' ','-')}.pdf")
plt.savefig(f"figs/clamp_degrade_diff_prob_{text.replace(' ','-')}.png")
plt.show()

# sns.heatmap((clamprob_map - degrprob_map).T, annot=True, fmt=".1f")
# sns.heatmap(clamp_map, annot=True, fmt=".1f")
# degrade_map.plot()

#%%
# def bert_forward(model, hidden_states):
#     extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_shape, device)
#     embedding_output = model.embeddings(
#         input_ids=input_ids,
#         position_ids=position_ids,
#         token_type_ids=token_type_ids,
#         inputs_embeds=inputs_embeds,
#         past_key_values_length=past_key_values_length,
#     )
#     head_mask = model.get_head_mask(head_mask, model.config.num_hidden_layers)
#     all_hidden_states = ()
#     for i, layer_module in enumerate(model.bert.encoder.layer):
#         layer_outputs = layer_module(
#             hidden_states,
#             # attention_mask,
#             # layer_head_mask,
#             # encoder_hidden_states,
#             # encoder_attention_mask,
#             # past_key_value,
#             # output_attentions,
#         )
#         hidden_states = layer_outputs[0]
#         # all_hidden_states = all_hidden_states + (hidden_states,)

