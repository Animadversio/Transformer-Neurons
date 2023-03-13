import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.plot_utils import saveallforms, join
import torch
from torch.nn.functional import softmax


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
            output_hidden_states=False, )

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
            output_hidden_states=False, )

    for hook_h in hook_hs:
        hook_h.remove()

    logits_noise = outputs_noise.logits[:, -1, :]
    prob_noise = softmax(logits_noise, dim=-1)
    logits_clamp = outputs_clamp.logits[:, -1, :]
    prob_clamp = softmax(logits_clamp, dim=-1)
    return outputs_noise, outputs_clamp, logits_noise, logits_clamp, prob_noise, prob_clamp


def causal_trace_pipeline(text, model, tokenizer, subject_loc, noise_batch=10, noise_std=.5, device="cpu", ):
    """Causal trace computation pipeline"""
    outputs_clean, inputs_embeds, inputs_attn_mask, logits_clean, _ = \
            embed_hidden_clean(model, tokenizer, text, device=device)
    maxidx = torch.argmax(logits_clean)
    besttoken = tokenizer.decode(maxidx)
    print(f"{besttoken} [{maxidx.item()}] logit={logits_clean[maxidx].item():.2f}")
    layer_num = len(model.transformer.h)
    data_dict = {}
    for clamp_layer in tqdm(range(layer_num)):
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


def embed_hidden_clean_bert(model, tokenizer, text, device="cpu"):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    mask_tok_loc = torch.where(token_ids[0] == tokenizer.mask_token_id)[0].item()
    # token_ids,
    inputs_embeds = model.bert.embeddings.word_embeddings(token_ids.to(device))
    with torch.no_grad():
        outputs_clean = model(inputs_embeds=inputs_embeds, output_hidden_states=True)

    logits_clean = outputs_clean.logits[0, mask_tok_loc, :]
    prob_clean = softmax(logits_clean, dim=-1)
    return outputs_clean, inputs_embeds, mask_tok_loc, logits_clean, prob_clean


def degradation_recovery_bert(model, inputs_embeds, mask_tok_loc, outputs_clean,
                        clamp_layer, clamp_token_loc,
                        noise_loc, noise_std=0.01, noise_batch=5):
    """
    Add noise to the input word embeddings at the noise_loc
    """
    inputs_embeds_degrade = inputs_embeds.repeat(noise_batch, 1, 1)
    inputs_embeds_degrade[:, noise_loc, :] += torch.randn_like(inputs_embeds_degrade[:, noise_loc, :]) * noise_std
    with torch.no_grad():
        outputs_noise = model(inputs_embeds=inputs_embeds_degrade, output_hidden_states=False)
    """
    Clamp the hidden states at the clamp_layer at the clamp_token_loc to be the 
        same as the hidden states of outputs_free. 
    """
    hook_hs = []
    for layeri in clamp_layer:
        target_module = model.bert.encoder.layer[layeri]
        fixed_hidden_states = outputs_clean.hidden_states[layeri+1].expand(noise_batch, -1, -1)
        fix_mask = torch.zeros(fixed_hidden_states.shape, dtype=torch.bool)
        fix_mask[:, clamp_token_loc, :] = 1
        hook_fun = get_clamp_hook(fixed_hidden_states, fix_mask)
        hook_h = target_module.register_forward_hook(hook_fun)
        hook_hs.append(hook_h)

    with torch.no_grad():
        outputs_clamp = model(inputs_embeds=inputs_embeds_degrade, output_hidden_states=False)

    for hook_h in hook_hs:
        hook_h.remove()

    logits_noise = outputs_noise.logits[:, mask_tok_loc, :]
    prob_noise = softmax(logits_noise, dim=-1)
    logits_clamp = outputs_clamp.logits[:, mask_tok_loc, :]
    prob_clamp = softmax(logits_clamp, dim=-1)
    return outputs_noise, outputs_clamp, logits_noise, logits_clamp, prob_noise, prob_clamp


def causal_trace_pipeline_bert(text, model, tokenizer, subject_loc, noise_batch=10, noise_std=.5, device="cpu", ):
    """Causal trace computation pipeline"""
    outputs_clean, inputs_embeds, mask_tok_loc, logits_clean, _ = \
        embed_hidden_clean_bert(model, tokenizer, text, device=device)
    maxidx = torch.argmax(logits_clean)
    besttoken = tokenizer.decode(maxidx)
    print(f"{besttoken} [{maxidx.item()}] logit={logits_clean[maxidx].item():.2f}")
    layer_num = len(model.bert.encoder.layer)
    data_dict = {}
    for clamp_layer in tqdm(range(layer_num)):
        for clamp_loc in tqdm(range(inputs_embeds.size(1))):
            # Clamp one token at one layer at a time.
            outputs_noise, outputs_clamp, logits_noise, logits_clamp, prob_noise, prob_clamp = \
                degradation_recovery_bert(model, inputs_embeds, mask_tok_loc, outputs_clean,
                 clamp_layer=[clamp_layer], clamp_token_loc=[clamp_loc],
                 noise_loc=subject_loc, noise_std=noise_std, noise_batch=noise_batch)
            data_dict[(clamp_layer, clamp_loc)] = {"logit_degrade": logits_noise[:, maxidx].mean().item(),
                                                   "logit_clamp": logits_clamp[:, maxidx].mean().item(),
                                                   "prob_degrade": prob_noise[:, maxidx].mean().item(),
                                                   "prob_clamp": prob_clamp[:, maxidx].mean().item(), }

    df = pd.DataFrame(data_dict).T
    df.reset_index(inplace=True)
    df.rename(columns={"level_0": "layer", "level_1": "position"}, inplace=True)
    degrade_map = df.pivot(index="layer", columns="position", values="logit_degrade")  # .plot()
    clamp_map = df.pivot(index="layer", columns="position", values="logit_clamp")
    degrprob_map = df.pivot(index="layer", columns="position", values="prob_degrade")  # .plot()
    clamprob_map = df.pivot(index="layer", columns="position", values="prob_clamp")
    return df, degrade_map, clamp_map, degrprob_map, clamprob_map


def visualize_causal_trace(text, tokenizer, df, figname, figdir="", subject_loc=(),):
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
    plt.gca().set_yticks(0.5 + torch.arange(len(tokens)))
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.ylabel(None)
    plt.title("logits difference: clamp - degrade")
    plt.tight_layout()
    saveallforms(figdir, f"causal_trace_{figname}_logit",)
    plt.show()

    plt.figure(figsize=(7, 4.5))
    sns.heatmap((clamprob_map - degrprob_map).T, annot=False, fmt=".1e")
    plt.gca().set_yticks(0.5 + torch.arange(len(tokens)))
    plt.gca().set_yticklabels(tokens, rotation=0)
    plt.ylabel(None)
    plt.title("prob difference: clamp - degrade")
    plt.tight_layout()
    saveallforms(figdir, f"causal_trace_{figname}_prob",)
    plt.show()