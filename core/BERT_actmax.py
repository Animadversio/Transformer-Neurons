"""
Find sequence of tokens / embedding that maximizes the activation of a given layer unit.

"""
import torch
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig, BertTokenizer
from core.layer_hook_utils import featureFetcher_module
from transformers import pipeline
#%%
configuration = BertConfig()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Initializing a model from the bert-base-uncased style configuration
# model = BertModel(configuration) # Bug, not properly initialized
model = BertModel.from_pretrained("bert-base-uncased")
#%%
model.requires_grad_(False)
model.eval()
#%%
input_words, maxprob, max_wordid = argmax_decode(masktok_dist[0], tokenizer)
#%%
inputs_embeds = torch.randn(1, 10, 768)
model.forward(inputs_embeds=inputs_embeds, token_type_ids=None)  # token_type_ids
#%%
layer_num = 1
fetcher = featureFetcher_module()
fetcher.record_module(model.encoder.layer[layer_num].intermediate.intermediate_act_fn, f"BERT_B{layer_num}_act", ingraph=True)
layerkey = f"BERT_B{layer_num}_act"
# unitact = fetcher['BERT_B11_act'][0, 1, unitid]
#%%
norm_bound = model.embeddings.word_embeddings.weight.norm(dim=1).mean()
Umat, Smat, Vmat = torch.svd(model.embeddings.word_embeddings.weight)
#%%
unitid = 10
inputs_embeds = torch.randn(5, 10, 768)
inputs_embeds.requires_grad_(True)
optim = torch.optim.Adam([inputs_embeds], lr=0.05)
for i in range(25):
    model.forward(inputs_embeds=inputs_embeds, token_type_ids=None)
    # unitact = fetcher[layerkey][:, :, unitid].mean()
    unitact, _ = fetcher[layerkey][:, :, unitid].max(dim=1)
    loss = -unitact.mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    # exceedmask =  inputs_embeds.data.norm(dim=-1, ) > norm_bound
    # inputs_embeds.data[exceedmask] = inputs_embeds.data[exceedmask] / inputs_embeds.data[exceedmask].norm(dim=-1, keepdim=True) * norm_bound
    if inputs_embeds.data.norm(dim=-1, ).mean() > norm_bound:
        inputs_embeds.data = inputs_embeds.data / inputs_embeds.data.norm(dim=-1, keepdim=True) * norm_bound
    # inputs_embeds.data = inputs_embeds.data + torch.randn_like(inputs_embeds.data) * 0.001
    print(f"Loss: {loss.item():.3f}")

"""Still very hard to optimize and hard to interpret, not very stable evolution"""
#%%
fetcher.cleanup()
#%%
def inputembed2dist(model, input_embeds, ):
    embW = model.embeddings.word_embeddings.weight / norm_bound
    input_word_dist = torch.einsum('BTH,WH->BTW', input_embeds
                               / input_embeds.norm(dim=-1, keepdim=True), embW)
    input_word_dist = torch.softmax(input_word_dist, dim=-1)
    return input_word_dist


def argmax_decode(input_dist, tokenizer):
    # input_wordid = torch.argmax(input_dist, dim=-1)
    maxprob, max_wordid = input_dist.max(dim=-1)
    input_words = tokenizer.convert_ids_to_tokens(max_wordid)
    return input_words, maxprob, max_wordid
#%%

input_word_dist = inputembed2dist(model, inputs_embeds.detach())
maxprobs, max_wordids = input_word_dist.max(dim=-1)
for i in range(input_word_dist.size(0)):
    input_words, maxprob, max_wordid = argmax_decode(input_word_dist[i], tokenizer)
    print(" ".join(input_words))

#%%
model.forward(input_ids=max_wordids, token_type_ids=None) # max_wordid[None, :]
unitact_val = fetcher.activations[layerkey][:, :, unitid]
print(unitact_val.mean(), unitact_val.std(), unitact_val)
#%% This will yield the exact same result as the above. Equivalent on code level.
model.forward(inputs_embeds=model.embeddings.word_embeddings(max_wordid[None, :]), token_type_ids=None)
unitact_valemb = fetcher.activations[layerkey][0, :, unitid]
#%%
model.forward(inputs_embeds=inputs_embeds.detach(), token_type_ids=None)
unitact_emb = fetcher.activations[layerkey][0, :, unitid]
#%%
embW = model.embeddings.word_embeddings.weight
input_word_dist = torch.einsum('BTH,WH->BTW', inputs_embeds.detach(), embW)
#%%
embnorm = embW.norm(dim=1,keepdim=True)
distmat = embnorm**2 + embnorm.T**2 - 2 * (embW @ embW.T)
torch.topk(distmat, 10, dim=-1)
torch.