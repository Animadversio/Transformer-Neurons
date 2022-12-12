import os
import warnings
from os.path import join
import torch
import torch.nn as nn  # import ModuleList
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
# model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
# # Accessing the model configuration
# configuration = model.config
# model.requires_grad_(False)
# model.eval()
savedir = r"F:\insilico_exps\GPT-XL_grad_trace"
#%%
def match_subject_decode(tokenizer, token_ids, subject_str):
    for subj_strt in range(len(token_ids)):
        for subj_end in range(subj_strt + 1, len(token_ids)):
            subj_prefix_raw = tokenizer.decode(token_ids[subj_strt:subj_end])
            subj_prefix = subj_prefix_raw.strip()
            if subj_prefix == subject_str[:len(subj_prefix)]:
                if subj_prefix == subject_str:
                    return subj_strt, subj_end
            elif subj_prefix == (subject_str + "'"):
                return subj_strt, subj_end
            else:
                break
    raise ValueError


def match_subject(tokens, subject_str):
    for subj_strt in range(len(tokens)):
        subj_residue = subject_str
        for subj_end in range(subj_strt + 1, len(tokens)):
            cur_token = tokens[subj_end - 1]
            if cur_token == subj_residue[:len(cur_token)]:
                subj_residue = subj_residue[len(cur_token):].strip()
                if len(subj_residue) == 0:
                    return subj_strt, subj_end
            else:
                break
    raise ValueError


import json
sumdir = join(savedir, "summary")
os.makedirs(sumdir, exist_ok=True)

k1000_data = json.load(open("dataset\\known_1000.json", 'r'))
mod_strs = ['block', 'attn', 'mlp', 'ln_2']
gradmap_summary = {mod_str: [] for mod_str in mod_strs}
actmap_summary = {mod_str: [] for mod_str in mod_strs}
gradactdotmap_summary = {mod_str: [] for mod_str in mod_strs}
gradflow_summary = {mod_str: [] for mod_str in ["pre", "post", "delta"]}
counter = 0
for i in tqdm(range(len(k1000_data))):
    text = k1000_data[i]['prompt']
    subject_str = k1000_data[i]["subject"]
    token_ids = tokenizer.encode(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
    tokens = [t.replace("Ġ", "") for t in tokens]
    text_label = "-".join(text.split(" ")[:4])
    try:
        # subj_tok_ids = tokenizer.encode(k1000_data[i]["subject"])
        # tok_ids = tokenizer.encode(text)
        # assert tok_ids[:len(subj_tok_ids)] == subj_tok_ids
        start_loc, end_loc = match_subject_decode(tokenizer, token_ids[0], subject_str)
    except ValueError:
        print(i, text, "Subject", subject_str)
        continue
    expdir = join(savedir, "fact%04d-" % (i) + text_label)
    grad_act_dict = torch.load(join(expdir, f"grad_act_save.pt"))
    gradsavedict = grad_act_dict["grad"]
    actsavedict = grad_act_dict["act"]
    grad_act_map_dict = torch.load(join(expdir, f"grad_act_map_save.pt"))
    gradmapsavedict = grad_act_map_dict["gradmap"]
    actmapsavedict = grad_act_map_dict["actmap"]
    rec_loc = [0, start_loc, max(start_loc, end_loc - 2), end_loc - 1, end_loc, -2, -1]
    for mod_str in mod_strs:
        recgradmap = gradmapsavedict[mod_str][:, rec_loc]
        gradmap_summary[mod_str].append(recgradmap)

        recactmap = actmapsavedict[mod_str][:, rec_loc]
        actmap_summary[mod_str].append(recactmap)

        gradvectsr = gradsavedict[mod_str][:, rec_loc, :]
        actvectsr = actsavedict[mod_str][:, rec_loc, :]
        gradactdotmap = torch.einsum("LTH,LTH->LT", gradvectsr, actvectsr)
        gradactdotmap_summary[mod_str].append(gradactdotmap)

    mod_str = "block"
    gradvectsr = gradsavedict[mod_str][:, rec_loc, :]
    actvectsr = actsavedict[mod_str][:, rec_loc, :]
    actdeltatsr = actvectsr[1:, :, :] - actvectsr[:-1, :, :]
    gradflow_summary["delta"].append(actdeltatsr.norm(dim=-1))
    gradflowdotmap = torch.einsum("LTH,LTH->LT", gradvectsr[:-1,:,:], actdeltatsr)
    gradflow_summary["pre"].append(gradflowdotmap)
    gradflowdotmap = torch.einsum("LTH,LTH->LT", gradvectsr[1:,:,:], actdeltatsr)
    gradflow_summary["post"].append(gradflowdotmap)
    counter += 1

#%% Stack and summary
for mod_str in mod_strs:
    gradmap_tsr = torch.stack(gradmap_summary[mod_str])
    gradmap_summary[mod_str] = gradmap_tsr

    actmap_tsr = torch.stack(actmap_summary[mod_str])
    actmap_summary[mod_str] = actmap_tsr

    gradactdot_tsr = torch.stack(gradactdotmap_summary[mod_str])
    gradactdotmap_summary[mod_str] = gradactdot_tsr

for k in gradflow_summary:
    gradflow_tsr = torch.stack(gradflow_summary[k])
    gradflow_summary[k] = gradflow_tsr
#%%
torch.save(gradmap_summary, join(sumdir, "gradmap_summary.pt"))
torch.save(actmap_summary, join(sumdir, "actmap_summary.pt"))
torch.save(gradactdotmap_summary, join(sumdir, "gradactdotmap_summary.pt"))
torch.save(gradflow_summary, join(sumdir, "gradflow_summary.pt"))
#%%
gradmap_tsr.mean(dim=0)


#%% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from core.plot_utils import saveallforms
rec_labels = np.array(["1st token", "Subject 1st", "Subject 2nd last", "Subject last",
              "1st Subsequent", "2nd last token", "last token"])
lab_ord = [1, 2, 3, 6, 4, 5]
#%%
def line_CI_plot(val_tsr, rec_labels, figsize=(8, 4), titlestr=""):
    sqrtN = np.sqrt(val_tsr.shape[0])
    layerN = val_tsr.shape[1]
    mapmean = val_tsr.mean(dim=0)
    mapstd = val_tsr.std(dim=0)
    mapsem = mapstd / sqrtN
    zval = 1.96
    map_UB = torch.quantile(val_tsr, 0.975, dim=0)
    map_LB = torch.quantile(val_tsr, 0.025, dim=0)
    assert mapmean.size(1) == len(rec_labels)
    figh = plt.figure(figsize=figsize)
    for iCol in range(mapmean.size(1)):
        plt.plot(mapmean[:, iCol], label=None)
        plt.fill_between(range(layerN), mapmean[:, iCol] - zval * mapsem[:, iCol],
                         mapmean[:, iCol] + zval * mapsem[:, iCol], label=rec_labels[iCol], alpha=0.5)
        # plt.fill_between(range(48), map_LB[:, iCol], map_UB[:, iCol],
        #                  label=rec_labels[iCol], alpha=0.5)
    plt.legend()
    plt.xlabel("layer depth")
    # plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(titlestr)
    plt.tight_layout()
    return figh


#%% Gradient ~ state difference
for key in gradflow_summary:
    plt.figure(figsize=[8, 4])
    sns.heatmap(gradflow_summary[key].mean(dim=0).T)
    plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(f"Mean hidden flow proj on {key} grad")
    plt.tight_layout()
    saveallforms(sumdir, f"gradflow_{key}_average")
    plt.show()
    figh = line_CI_plot(gradflow_summary[key], rec_labels,
                titlestr=f"Mean hidden flow proj on {key} grad")
    saveallforms(sumdir, f"gradflow_{key}_line_CI", figh)
    plt.show()
    figh2 = line_CI_plot(gradflow_summary[key][:, :, lab_ord], rec_labels[lab_ord],
                        titlestr=f"Mean hidden flow proj on {key} grad")
    saveallforms(sumdir, f"gradflow_{key}_line_CI_clrmat", figh2)
    plt.show()

#%% Gradient norm of outputs
for mod_str in mod_strs:
    plt.figure(figsize=[8,4])
    sns.heatmap(gradmap_summary[mod_str].mean(dim=0).T)
    plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(f"Mean gradient norm map for [{mod_str}] output")
    plt.tight_layout()
    saveallforms(sumdir, f"grad_map_average_{mod_str}")
    plt.show()

    figh = line_CI_plot(gradmap_summary[mod_str], rec_labels,
                        titlestr=f"Mean gradient norm for [{mod_str}] output")
    saveallforms(sumdir, f"grad_line_CI_{mod_str}", figh)
    plt.show()

    figh2 = line_CI_plot(gradmap_summary[mod_str][:, :, lab_ord], rec_labels[lab_ord],
                        titlestr=f"Mean gradient norm for [{mod_str}] output")
    saveallforms(sumdir, f"grad_line_CI_{mod_str}_clrmat", figh2)
    plt.show()

#%% Gradient norm of attention layer output (excluding layer 0 )
mod_str = "attn"
plt.figure(figsize=[8, 4])
sns.heatmap(gradmap_summary[mod_str].mean(dim=0).T[:, 1:])
plt.gca().set_yticklabels(rec_labels, rotation=0)
plt.title(f"Mean gradient norm map for [{mod_str}] output")
plt.tight_layout()
saveallforms(sumdir, f"grad_map_average_{mod_str}_excl0", )
plt.show()
figh = line_CI_plot(gradmap_summary[mod_str][:, 1:, :], rec_labels,
                    titlestr=f"Mean gradient norm for [{mod_str}] output")
saveallforms(sumdir, f"grad_line_CI_{mod_str}_excl0", figh)
plt.show()
figh2 = line_CI_plot(gradmap_summary[mod_str][:, 1:, lab_ord], rec_labels[lab_ord],
                    titlestr=f"Mean gradient norm for [{mod_str}] output")
saveallforms(sumdir, f"grad_line_CI_{mod_str}_excl0_clrmat", figh2)
plt.show()
# %% Activation norm
for mod_str in mod_strs:
    plt.figure(figsize=[8,4])
    sns.heatmap(actmap_summary[mod_str].mean(dim=0).T)
    plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(f"Mean activation norm map for [{mod_str}] output")
    plt.tight_layout()
    saveallforms(sumdir, f"act_map_average_{mod_str}", )
    plt.show()
    figh = line_CI_plot(actmap_summary[mod_str], rec_labels,
                        titlestr=f"Mean activation norm for [{mod_str}] output")
    saveallforms(sumdir, f"act_line_CI_{mod_str}", )
    plt.show()
    figh2 = line_CI_plot(actmap_summary[mod_str][:,:,lab_ord], rec_labels[lab_ord],
                        titlestr=f"Mean activation norm for [{mod_str}] output")
    saveallforms(sumdir, f"act_line_CI_{mod_str}_clrmat", figh2)
    plt.show()

# %% Gradient activation Dot Map of a single layer
for mod_str in mod_strs:
    plt.figure(figsize=[8, 4])
    sns.heatmap(gradactdotmap_summary[mod_str].mean(dim=0).T)
    plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(f"Mean grad-act dot map for [{mod_str}] output")
    plt.tight_layout()
    saveallforms(sumdir, f"grad_act_dot_map_average_{mod_str}", )
    plt.show()

    figh = line_CI_plot(gradactdotmap_summary[mod_str], rec_labels,
                        titlestr=f"Mean grad act dot for [{mod_str}] output")
    saveallforms(sumdir, f"grad_act_dot_line_CI_{mod_str}", figh)
    plt.show()
    figh2 = line_CI_plot(gradactdotmap_summary[mod_str][:,:,lab_ord], rec_labels[lab_ord],
                        titlestr=f"Mean grad act dot for [{mod_str}] output")
    saveallforms(sumdir, f"grad_act_dot_line_CI_{mod_str}_clrmat", figh2)
    plt.show()
#%% Grad block == Grad MLP, Grad act dot of block - MLP == Grad of block dot Attn + prev block
gradactdotmap_res = gradactdotmap_summary["block"] - gradactdotmap_summary["mlp"]
plt.figure(figsize=[8, 4])
sns.heatmap(gradactdotmap_res.mean(dim=0).T)
plt.gca().set_yticklabels(rec_labels, rotation=0)
plt.title(f"Mean grad-act dot map for [res = block - mlp] output")
plt.tight_layout()
saveallforms(sumdir, f"grad_act_dot_map_average_res", )
plt.show()
#%%
# 10 layer moving average of grad act dot product.
from scipy.ndimage import uniform_filter
for mod_str in mod_strs:
    # mavg_map = uniform_filter(gradactdotmap_summary[mod_str].mean(dim=0),
    #                           size=[10, 1], mode="constant", cval=0.0)
    mavg_tsr = torch.tensor(uniform_filter(gradactdotmap_summary[mod_str],
                              size=[1, 10, 1], mode="constant", cval=0.0))
    plt.figure(figsize=[8, 4])
    sns.heatmap(mavg_tsr.mean(dim=0).T)
    plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(f"Mean grad-act dot map for [{mod_str}] output")
    plt.tight_layout()
    saveallforms(sumdir, f"grad_act_dot_map_average_10Lavg_{mod_str}", )
    plt.show()

    figh = line_CI_plot(mavg_tsr, rec_labels,
                        titlestr=f"Mean grad act dot for [{mod_str}] output")
    saveallforms(sumdir, f"grad_act_dot_line_CI_10Lavg_{mod_str}", figh)
    plt.show()

    figh2 = line_CI_plot(mavg_tsr[:,:,lab_ord], rec_labels[lab_ord],
                        titlestr=f"Mean grad act dot for [{mod_str}] output")
    saveallforms(sumdir, f"grad_act_dot_line_CI_10Lavg_{mod_str}_clrmat", figh2)
    plt.show()
# %% Cosine between gradient and activation.
for mod_str in mod_strs:
    gradact_cosine = gradactdotmap_summary[mod_str] / actmap_summary[mod_str] / gradmap_summary[mod_str]
    plt.figure(figsize=[8, 4])
    sns.heatmap(gradact_cosine.mean(dim=0).T)
    plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(f"Mean grad-act cosine map for [{mod_str}] output")
    plt.tight_layout()
    saveallforms(sumdir, f"grad_act_cosine_map_average_{mod_str}", )
    plt.show()
    figh = line_CI_plot(gradact_cosine, rec_labels,
                        titlestr=f"Mean grad act cosine for [{mod_str}] output")
    saveallforms(sumdir, f"grad_act_cosine_line_CI_{mod_str}", figh)
    plt.show()

    figh2 = line_CI_plot(gradact_cosine[:, :, lab_ord], rec_labels[lab_ord],
                         titlestr=f"Mean grad act cosine for [{mod_str}] output")
    saveallforms(sumdir, f"grad_act_cosine_line_CI_{mod_str}_clrmat", figh2)
    plt.show()
#%% Ratio between gradient norm and activation norm
mod_str = "block"
for mod_str in mod_strs:
    plt.figure(figsize=[8,4])
    ratiomap_tsr = (gradmap_summary[mod_str] / actmap_summary[mod_str])
    ratiomap = torch.nanmean(ratiomap_tsr, dim=0)
    sns.heatmap(ratiomap.T)
    plt.gca().set_yticklabels(rec_labels, rotation=0)
    plt.title(f"Mean grad / act norm ratio map for [{mod_str}] output")
    plt.tight_layout()
    saveallforms(sumdir, f"grad_act_ratio_map_average_{mod_str}", )
    plt.show()
#%%
#%%
match_subject(tokens, k1000_data[i]["subject"])

text = "Co-operative Commonwealth Federation (Ontario Section)'s headquarters are in"
subject_str = "Co-operative Commonwealth Federation (Ontario Section)"
token_ids = tokenizer.encode(text)
subj_dec_raw = tokenizer.decode(token_ids[0:10]).strip()
subj_dec_raw == subject_str + "'"