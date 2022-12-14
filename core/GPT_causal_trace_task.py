import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.functional import softmax
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.requires_grad_(False)
model.eval()
#%%
wte_std = model.transformer.wte.weight.std()
print(f"std of Token embedding vectors {wte_std:.4f}", )
print("Empirical std of Token embedding vectors?", )#model.transformer.wte.weight.std())

#%%
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.plot_utils import saveallforms, join


def get_clamp_hook(fixed_hidden_states, fix_mask):
    def clamp_hidden_hook(module, input, output):
        """Substitute part of the output of certain layer by a fixed representation"""
        output[0].data[fix_mask] = fixed_hidden_states[fix_mask]
    return clamp_hidden_hook


def embed_hidden_clean(model, tokenizer, text, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt")
    inputs_embeds = model.transformer.wte(inputs["input_ids"].to(device))
    inputs_attn_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs_clean = model(  # input_ids=inputs["input_ids"],
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_attn_mask,
            output_hidden_states=True, )

    logits_clean = outputs_clean.logits[0, -1, :]
    prob_clean = softmax(logits_clean, dim=-1)
    return outputs_clean, inputs_embeds, inputs_attn_mask, logits_clean, prob_clean


def degradation_recovery(model, inputs_embeds, inputs_attn_mask, outputs_clean,
                         clamp_layer, clamp_token_loc,
                         noise_loc, noise_std=0.2, noise_batch=5):

    inputs_embeds_degrade = inputs_embeds.repeat(noise_batch, 1, 1)
    inputs_embeds_degrade[:, noise_loc, :] += torch.randn_like(inputs_embeds_degrade[:, noise_loc, :]) * noise_std
    with torch.no_grad():
        outputs_noise = model(  # input_ids=inputs["input_ids"],
            inputs_embeds=inputs_embeds_degrade,
            attention_mask=inputs_attn_mask.expand(noise_batch, -1),
            output_hidden_states=True, )

    hook_hs = []
    for layeri in clamp_layer:
        target_module = model.transformer.h[layeri]
        fixed_hidden_states = outputs_clean.hidden_states[layeri + 1].expand(noise_batch, -1, -1)
        fix_mask = torch.zeros(fixed_hidden_states.shape, dtype=torch.bool)
        fix_mask[:, clamp_token_loc, :] = 1
        hook_fun = get_clamp_hook(fixed_hidden_states, fix_mask)
        hook_h = target_module.register_forward_hook(hook_fun)
        hook_hs.append(hook_h)

    with torch.no_grad():
        outputs_clamp = model(inputs_embeds=inputs_embeds_degrade,
            attention_mask=inputs_attn_mask.expand(noise_batch, -1),
            output_hidden_states=True, )

    for hook_h in hook_hs:
        hook_h.remove()

    logits_noise = outputs_noise.logits[:, -1, :]
    prob_noise = softmax(logits_noise, dim=-1)
    logits_clamp = outputs_clamp.logits[:, -1, :]
    prob_clamp = softmax(logits_clamp, dim=-1)
    return outputs_noise, outputs_clamp, logits_noise, logits_clamp, prob_noise, prob_clamp


def causal_trace_pipeline(text, model, tokenizer, subject_loc, noise_batch=10, noise_std=.5, device="cpu"):
    """Causal trace computation pipeline"""
    outputs_clean, inputs_embeds, inputs_attn_mask, logits_clean, _ = \
        embed_hidden_clean(model, tokenizer, text, device=device)
    maxidx = torch.argmax(logits_clean)
    besttoken = tokenizer.decode(maxidx)
    print(f"{besttoken} [{maxidx.item()}] logit={logits_clean[maxidx].item():.2f}")
    data_dict = {}
    for clamp_layer in tqdm(range(len(model.transformer.h))):
        for clamp_loc in tqdm(range(inputs_embeds.size(1))):
            # Clamp one token at one layer at a time.
            outputs_noise, outputs_clamp, logits_noise, logits_clamp, prob_noise, prob_clamp = \
                degradation_recovery(model, inputs_embeds, inputs_attn_mask, outputs_clean,
                     clamp_layer=[clamp_layer], clamp_token_loc=[clamp_loc],
                     noise_loc=subject_loc, noise_std=noise_std, noise_batch=noise_batch)
            data_dict[(clamp_layer, clamp_loc)] = {"logit_degrade": logits_noise[:, maxidx].mean().item(),
                                                   "logit_clamp": logits_clamp[:, maxidx].mean().item(),
                                                   "prob_degrade": prob_noise[:, maxidx].mean().item(),
                                                   "prob_clamp": prob_clamp[:, maxidx].mean().item(), }

    df = pd.DataFrame(data_dict).T  # .plot()
    df.reset_index(inplace=True)
    df.rename(columns={"level_0": "layer", "level_1": "position"}, inplace=True)
    degrade_map = df.pivot(index="layer", columns="position", values="logit_degrade")  # .plot()
    clamp_map = df.pivot(index="layer", columns="position", values="logit_clamp")
    degrprob_map = df.pivot(index="layer", columns="position", values="prob_degrade")  # .plot()
    clamprob_map = df.pivot(index="layer", columns="position", values="prob_clamp")
    return df, degrade_map, clamp_map, degrprob_map, clamprob_map


def visualize_causal_trace(text, tokenizer, df, figname, figdir="", subject_loc=[],):
    degrade_map = df.pivot(index="layer", columns="position", values="logit_degrade")  # .plot()
    clamp_map = df.pivot(index="layer", columns="position", values="logit_clamp")
    degrprob_map = df.pivot(index="layer", columns="position", values="prob_degrade")  # .plot()
    clamprob_map = df.pivot(index="layer", columns="position", values="prob_clamp")

    tok_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(tok_ids)
    tokens = [t.replace("Ġ", " ") for t in tokens]
    for loc in subject_loc:
        tokens[loc] = tokens[loc] + "*"

    plt.figure(figsize=(7, 4.5))
    sns.heatmap((clamp_map - degrade_map).T, annot=False, fmt=".1e")
    # plt.axis("image")
    # plt.yticks(ticks=0.5 + np.arange(len(tokens)),
    #            labels=tokens, )
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.ylabel(None)
    plt.title("logits difference: clamp - degrade")
    plt.tight_layout()
    saveallforms(figdir, f"causal_trace_{figname}_logit",)
    plt.show()

    plt.figure(figsize=(7, 4.5))
    sns.heatmap((clamprob_map - degrprob_map).T, annot=False, fmt=".1e")
    # plt.axis("image")
    # plt.yticks(ticks=0.5 + np.arange(len(tokens)),
    #            labels=tokens, )
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.ylabel(None)
    plt.title("prob difference: clamp - degrade")
    plt.tight_layout()
    saveallforms(figdir, f"causal_trace_{figname}_prob",)
    plt.show()

#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.requires_grad_(False)
model.eval()
device = "cuda"
figroot = r"F:\insilico_exps\GPT-XL_causal_trace"
# text = "Vatican is located in the city of"
text = "The Space Needle is located in downtown"
subject_loc = [0, 1, 2, 3]
figdir = join(figroot, text.replace(" ", "_")[0:25])
os.makedirs(figdir, exist_ok=True)
figname = ""
df, degrade_map, clamp_map, degrprob_map, clamprob_map = \
    causal_trace_pipeline(text, model, tokenizer, subject_loc, noise_batch=10, noise_std=0.15, device=device)
df.to_csv(join(figdir, "causal_trace_data.csv"))
visualize_causal_trace(text, tokenizer, df, figname, figdir=figdir, subject_loc=subject_loc)
#%%
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
# model = GPT2LMHeadModel.from_pretrained("gpt2-large")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.requires_grad_(False)
model.eval()
#%%
device = "cuda"
model.to(device)
figroot = r"F:\insilico_exps\GPT-large_causal_trace"
figroot = r"F:\insilico_exps\GPT_causal_trace"
# text = "The Space Needle is located in downtown"
# subject_loc = [0, 1, 2, 3]
text = "Vatican is located in the city of"
subj_tokens = [0, 1]
figdir = join(figroot, text.replace(" ", "_")[0:25])
os.makedirs(figdir, exist_ok=True)
figname = ""
df, degrade_map, clamp_map, degrprob_map, clamprob_map = \
    causal_trace_pipeline(text, model, tokenizer, subject_loc, noise_batch=10, noise_std=0.45, device=device)
df.to_csv(join(figdir, "causal_trace_data.csv"))
visualize_causal_trace(text, tokenizer, df, figname, figdir=figdir, subject_loc=subject_loc)

#%%
# text = "The Space Needle is located in downtown"
# subj_tokens = [0, 1, 2, 3]
text = "Vatican is located in the city of"
subj_tokens = [0, 1]
outputs_clean, inputs_embeds, logits_clean, _ = \
    embed_hidden_clean(model, tokenizer, text, )
maxidx = torch.argmax(logits_clean)
besttoken = tokenizer.decode(maxidx)
data_dict = {}
for clamp_layer in range(12):
    for clamp_loc in range(inputs_embeds.size(1)):
        outputs_noise, outputs_clamp, logits_noise, logits_clamp, prob_noise, prob_clamp = \
            degradation_recovery(model, inputs_embeds, outputs_clean,
                 clamp_layer=[clamp_layer], clamp_token_loc=[clamp_loc],
                 noise_loc=subj_tokens, noise_std=0.5, noise_batch=10)
        data_dict[(clamp_layer, clamp_loc)] = {"logit_degrade": logits_noise[:, maxidx].mean().item(),
                                             "logit_clamp": logits_clamp[:, maxidx].mean().item(),
                                             "prob_degrade": prob_noise[:, maxidx].mean().item(),
                                             "prob_clamp": prob_clamp[:, maxidx].mean().item(), }
#%%
df = pd.DataFrame(data_dict).T #.plot()
df.reset_index(inplace=True)
df.rename(columns={"level_0": "layer", "level_1": "position"}, inplace=True)
degrade_map = df.pivot(index="layer", columns="position", values="degrade") # .plot()
clamp_map = df.pivot(index="layer", columns="position", values="clamp")
degrprob_map = df.pivot(index="layer", columns="position", values="prob_degrade") # .plot()
clamprob_map = df.pivot(index="layer", columns="position", values="prob_clamp")

#%%
tok_ids = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(tok_ids)
tokens = [t.replace("Ġ", " ") for t in tokens]
# normal_tokens = tokenizer.convert_ids_to_tokens(
#     [token_ids[loc] for loc in normal_locs])
# sns.heatmap(degrade_map, annot=True, fmt=".1f")
# sns.heatmap(clamp_map, annot=True, fmt=".1f")
plt.figure(figsize=(7, 4.5))
sns.heatmap((clamp_map - degrade_map).T, annot=False, fmt=".1e")
plt.axis("image")
plt.yticks(ticks=0.5 + np.arange(len(tokens)),
           labels=tokens, )
plt.tight_layout()
plt.title("logits difference: clamp - degrade")
plt.savefig(f"figs/GPT2_clamp_degrade_diff_logit_{text.replace(' ','-')}.pdf")
plt.savefig(f"figs/GPT2_clamp_degrade_diff_logit_{text.replace(' ','-')}.png")
plt.show()

plt.figure(figsize=(7, 4.5))
sns.heatmap((clamprob_map - degrprob_map).T, annot=False, fmt=".1e")
plt.axis("image")
plt.yticks(ticks=0.5 + np.arange(len(tokens)),
           labels=tokens, )
plt.tight_layout()
plt.title("prob difference: clamp - degrade")
plt.savefig(f"figs/GPT2_clamp_degrade_diff_prob_{text.replace(' ','-')}.pdf")
plt.savefig(f"figs/GPT2_clamp_degrade_diff_prob_{text.replace(' ','-')}.png")
plt.show()

#%%

print(f"{besttoken} ({maxidx.item()})")
# print(f"Noise position {}")
print("Noise Logit %.2f"%logits_noise[:, maxidx].mean().item())
print("Clamp recover Logit %.2f"%logits_clamp[:, maxidx].mean().item())
print("diff = %.2f"%((logits_clamp[:, maxidx] - logits_noise[:, maxidx])
                     .mean().item()))
#%%
all_hidden = (outputs.hidden_states)
for li in range(len(all_hidden)):
    logits_lyr = model.lm_head(all_hidden[li])
    topvals, topidxs = torch.topk(logits_lyr[0, -1, ], 10)
    print(li, tokenizer.decode(topidxs.T))
#%%

#%%
logits_lyr = model.lm_head(outputs.hidden_states[-1])

#%%
# text = "The Vatican is in the city of"
text = "The Space Needle is located in downtown"
inputs = tokenizer(text, return_tensors="pt")
input_embeds = model.base_model.wte(inputs["input_ids"])
with torch.no_grad():
    outputs = model(#input_ids=inputs["input_ids"],
                    inputs_embeds=input_embeds,
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,)
    # labels=inputs["input_ids"],
loss = outputs.loss
logits = outputs.logits
logits_final = logits[0, -1, :]
prob_final = softmax(logits_final, dim=-1)
topvals, topidxs = torch.topk(logits_final, 10)
print(tokenizer.decode(topidxs))
