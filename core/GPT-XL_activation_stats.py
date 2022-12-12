"""
Analysis of the activation statistics of GPT-XL

"""
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from core.plot_utils import saveallforms
figdir = r"F:\insilico_exps\GPT-XL_grad_trace\stat_summary"
datadir = r"F:\insilico_exps\GPT-XL_grad_trace"
act_storage = torch.load(join(datadir, "activation_storage.pt"))
sval_storage = torch.load(join(figdir, "spectrum_all_layer.pt"))
# torch.save(act_storage, join(datadir, "activation_storage.pt"))
#%% Compute singular value spectrum for each activation tensor
sval_storage = {}
for k in tqdm(act_storage):
    svals = torch.linalg.svdvals(act_storage[k][0].cuda()).cpu()
    sval_storage[k] = svals
#%%
torch.save(sval_storage, join(figdir, "spectrum_all_layer.pt"))
#%%
SVnum = 100
norm_mode = "sqsum1"  # "max"  # "sqsum1" "sum1" "none"
mod_sfx = "_act"  # "_proj" "_fc", "", "_attn"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)])
if norm_mode == "max":
    norm_spect_tsr = spect_tsr / spect_tsr[:, 0:1]
elif norm_mode == "none":
    norm_spect_tsr = spect_tsr
elif norm_mode == "sum1":
    norm_spect_tsr = spect_tsr / spect_tsr.sum(dim=1, keepdim=True)
elif norm_mode == "sqsum1":
    norm_spect_tsr = (spect_tsr**2) / (spect_tsr**2).sum(dim=1, keepdim=True)
else:
    raise NotImplementedError

plt.figure()
sns.heatmap(torch.log10(norm_spect_tsr[:, :SVnum]))  # / spect_tsr.sum(dim=1, keepdim=True)
plt.xlabel("singular value #")
plt.ylabel("layer #")
plt.tight_layout()
plt.show()
#%%
mod_sfx = ""  # "_proj" "_fc", "", "_attn"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)])
plt.figure()
plt.plot(torch.log10(spect_tsr[:, 0]))  # / spect_tsr.sum(dim=1, keepdim=True)
plt.ylabel("log10(max singular value)")
plt.xlabel("layer #")
plt.title(f"SVD for GPT2_B#{mod_sfx} activation tensor")
plt.tight_layout()
plt.show()
#%%
mod_sfx = ""  # "_proj" "_fc", "", "_attn"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)])
plt.figure()
plt.plot(torch.log10(spect_tsr[:, :].sum(dim=1)))  # / spect_tsr.sum(dim=1, keepdim=True)
plt.ylabel("log10(sum(singular value)) i.e. log(nucl norm)")
plt.xlabel("layer #")
plt.title(f"SVD for GPT2_B#{mod_sfx} activation tensor")
plt.tight_layout()
plt.show()

#%%
mod_sfx = ""  # "_proj" "_fc", "", "_attn"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)])
plt.figure()
plt.plot(torch.log10((spect_tsr[:, :]**2).sum(dim=1)))  # / spect_tsr.sum(dim=1, keepdim=True)
plt.ylabel("log10(sum(squared singular value)) i.e. log(Frob norm)")
plt.xlabel("layer #")
plt.title(f"SVD for GPT2_B#{mod_sfx} activation tensor")
plt.tight_layout()
plt.show()
#%%
mod_sfx = ""  # "_proj" "_fc", "", "_attn"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)])
plt.figure()
plt.plot(((spect_tsr[:, :100]**2).sum(dim=1)/(spect_tsr[:, :]**2).sum(dim=1)))  # / spect_tsr.sum(dim=1, keepdim=True)
plt.ylabel("sum(100 top singular value squared) / sum(all squared)")
plt.xlabel("layer #")
plt.title(f"SVD for GPT2_B#{mod_sfx} activation tensor")
plt.tight_layout()
plt.show()

#%%
""" Vector L2 norm across layers """
mod_sfx = ""
for mod_sfx in ["", "_proj", "_fc", "_act", "_attn"]:
    norm_vectors = [act_storage[f"GPT2_B{i}{mod_sfx}"].norm(dim=-1) for i in range(48)]
    norm_avg = torch.tensor([norm_vec.mean() for norm_vec in norm_vectors])
    norm_med = torch.tensor([torch.quantile(norm_vec, .50) for norm_vec in norm_vectors])
    norm_LB = torch.tensor([torch.quantile(norm_vec, .25) for norm_vec in norm_vectors])
    norm_UB = torch.tensor([torch.quantile(norm_vec, .75) for norm_vec in norm_vectors])
    plt.figure()
    plt.plot(norm_avg, color="k", label="average")  # / spect_tsr.sum(dim=1, keepdim=True)
    plt.plot(norm_med, label="median")  # / spect_tsr.sum(dim=1, keepdim=True)
    plt.fill_between(range(48), norm_LB, norm_UB, alpha=0.4)  # / spect_tsr.sum(dim=1, keepdim=True)
    plt.ylabel("vector L2 norm")
    plt.xlabel("layer #")
    plt.title(f"GPT2_B#{mod_sfx} activation tensor L2 norm")
    plt.tight_layout()
    saveallforms(figdir, f"vecnorm_curve_B{mod_sfx}")
    plt.show()
#%% log scale vector norm
mod_sfx = ""
norm_vectors = [act_storage[f"GPT2_B{i}{mod_sfx}"].norm(dim=-1) for i in range(48)]
norm_med = torch.tensor([torch.quantile(norm_vec, .50) for norm_vec in norm_vectors])
norm_LB = torch.tensor([torch.quantile(norm_vec, .25) for norm_vec in norm_vectors])
norm_UB = torch.tensor([torch.quantile(norm_vec, .75) for norm_vec in norm_vectors])
# norm_avg = torch.tensor([norm_vec.mean() for norm_vec in norm_vectors])
plt.plot(torch.log10(norm_med), label="Median")
plt.plot(torch.log10(norm_LB), label="25 perctl")
plt.plot(torch.log10(norm_UB), label="75 perctl")
plt.ylabel("log10(hidden state norm)")
plt.xlabel("layer #")
plt.legend()
plt.title("log linearity of the hidden state norm")
saveallforms(figdir, f"vecnorm_logcurve_Block")
plt.show()
#%%

#%% Sparsity of the MLP hidden state activations
thresh = 5E-2
layer_ids = []
sparsity_col = []
for layer_i in range(48):
    acttsr = act_storage[f'GPT2_B{layer_i}_act']
    # activation fraction for each units in the hidden layer ( # token active / # token total )
    fractions = (acttsr > thresh).sum(dim=(0, 1)) / acttsr.size(1)
    sparsity_col.append(fractions)
    layer_ids.append(layer_i * torch.ones_like(fractions, dtype=torch.int))

layer_ids = torch.cat(layer_ids)
frac_vec = torch.cat(sparsity_col)
spars_df = pd.DataFrame({"layer": layer_ids, "fraction":frac_vec})

#%%
plt.figure(figsize=[9, 6])
sns.violinplot(data=spars_df, x="layer", y="fraction",
               cut=0, width=1.2, inner="quartile", linewidth=0.5)
plt.title("GPT2-XL MLP hidden unit activated fraction")
plt.ylabel("activation fraction")
plt.tight_layout()
saveallforms(figdir, "acti_fraction_violin")
plt.show()
#%%
plt.figure(figsize=[6, 4])
sns.lineplot(data=spars_df, x="layer", y="fraction", )
# sns.scatterplot(data=spars_df, x="layer", y="fraction", alpha=0.2, size=8)
plt.title("GPT2-XL MLP hidden unit activated fraction (mean - 95% CI)")
plt.ylabel("mean activation fraction")
plt.tight_layout()
saveallforms(figdir, "acti_fraction_lineplot")
plt.show()
#%%
