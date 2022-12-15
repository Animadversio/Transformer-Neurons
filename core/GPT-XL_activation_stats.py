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
for k in tqdm(range(1, 48)):
    hidden_diff = act_storage[f"GPT2_B{k}"] - act_storage[f"GPT2_B{k-1}"]
    svals = torch.linalg.svdvals(hidden_diff[0].cuda()).cpu()
    sval_storage[f"GPT2_B{k}_delta"] = svals
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
plt.ylabel("log10(max singular value) i.e. log(mat L2 norm)")
plt.xlabel("layer #")
plt.title(f"SVD for GPT2_B#{mod_sfx} activation tensor")
plt.tight_layout()
plt.show()
#%%
mod_sfx = ""  # "_proj" "_fc", "", "_attn"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)])
plt.figure()
plt.plot(torch.log10(spect_tsr[:, :].sum(dim=1)))  # / spect_tsr.sum(dim=1, keepdim=True)
plt.ylabel("log10(sum(singular value)) i.e. log(nuclear norm)")
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
plt.plot(((spect_tsr[:, :10]**2).sum(dim=1)/(spect_tsr[:, :]**2).sum(dim=1)))  # / spect_tsr.sum(dim=1, keepdim=True)
plt.ylabel("sum(100 top singular value squared) / sum(all squared)")
plt.xlabel("layer #")
plt.title(f"SVD for GPT2_B#{mod_sfx} activation tensor")
plt.tight_layout()
plt.show()
#%%
def plot_effective_dim(spect_tsr, mod_sfx=""):
    """ Plot the effective dimension of the activation tensor
    plot the number of singular values that account for 95, 98, 99% of the total norm

    :param spect_tsr: layer x singular value tensor
    :param mod_sfx: suffix for the plot title
    :return: figure handle
    """
    expfrac_tsr = (spect_tsr[:, :] ** 2).cumsum(dim=1) / (spect_tsr[:, :] ** 2).sum(dim=1, keepdim=True)
    dim_vec_99 = (expfrac_tsr < 0.99).sum(dim=1)
    dim_vec_98 = (expfrac_tsr < 0.98).sum(dim=1)
    dim_vec_95 = (expfrac_tsr < 0.95).sum(dim=1)
    figh = plt.figure()
    plt.plot(dim_vec_99, label="99% var")  # / spect_tsr.sum(dim=1, keepdim=True)
    plt.plot(dim_vec_98, label="98% var")  # / spect_tsr.sum(dim=1, keepdim=True)
    plt.plot(dim_vec_95, label="95% var")  # / spect_tsr.sum(dim=1, keepdim=True)
    plt.legend()
    plt.ylabel("# of singular values")
    plt.xlabel("layer #")
    plt.title(
        f"Effective dimension: \n# of singular values that explain K% of the variance\nSVD for GPT2_B#{mod_sfx} activation tensor")
    plt.tight_layout()
    plt.show()
    return figh
#%% Effective dimensionality
mod_sfx = "_fc"  # "_proj" "_fc", "", "_attn"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)])
plot_effective_dim(spect_tsr, mod_sfx=mod_sfx)
saveallforms(figdir, f"hidden_state{mod_sfx}_effective_dim")
#%%
mod_sfx = "_delta"
spect_tsr = torch.stack([sval_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(1,48)])
figh = plot_effective_dim(spect_tsr, mod_sfx="_delta")
figh.gca().set_title("Effective dimension: \n# of singular values that explain K% of the variance\nSVD for GPT2 activation tensor difference")
saveallforms(figdir, f"hidden_state{mod_sfx}_effective_dim", figh)
#%% Vector norm analysis
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
    plt.legend()
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
plt.title("log linearity of the hidden state L2 norm")
plt.legend()
saveallforms(figdir, f"vecnorm_logcurve_Block")
plt.show()
#%%
"""Modification of hidden states across layer."""
mod_sfx = ""
act_lyr_tsr = torch.cat([act_storage[f"GPT2_B{i}{mod_sfx}"] for i in range(48)], dim=0)
print(act_lyr_tsr.shape)  # layer, token, hidden
act_diff_tsr = act_lyr_tsr[1:, :, :] - act_lyr_tsr[:-1, :, :]
#%%
vecdiff_norm = act_diff_tsr.norm(dim=-1)
plt.figure()
plt.plot(vecdiff_norm.mean(dim=1), label="mean", color="k")
plt.plot(torch.quantile(vecdiff_norm, .50, dim=1), label="median")
plt.fill_between(range(47),
                 torch.quantile(vecdiff_norm, .25, dim=1),
                 torch.quantile(vecdiff_norm, .75, dim=1), alpha=0.4, label="25-75 perctl")
plt.ylabel("L2 norm")
plt.xlabel("layer #")
plt.title("Norm of hidden state differences across layer.")
plt.legend()
plt.tight_layout()
saveallforms(figdir, f"hidden_state_diff_vecnorm_curve")
plt.show()
#%%
"""Similarity of hidden states across layer."""
act_norm = act_lyr_tsr.norm(dim=-1)
act_inprod = torch.einsum("lth, Lth -> lLt", act_lyr_tsr, act_lyr_tsr)
act_cosine = act_inprod / (act_norm[:, None, :] * act_norm[None, :, :] + 1e-8)
#%%
plt.figure()
sns.heatmap(act_cosine.mean(dim=-1), cmap="viridis")
plt.title("Mean cosine similarity of hidden states across layer.")
plt.xlabel("layer #")
plt.ylabel("layer #")
plt.axis("image")
plt.tight_layout()
saveallforms(figdir, f"hidden_state_cossim_heatmap_mean")
plt.show()
#%%
"""Similarity of hidden states modification across layer."""
vecdiff_norm = act_diff_tsr.norm(dim=-1)
vecdiff_inprod = torch.einsum("ljk, Ljk -> lLj", act_diff_tsr, act_diff_tsr)
vecdiff_cossim = vecdiff_inprod / (vecdiff_norm[:, None, :] * vecdiff_norm[None, :, :])
#%%
vecdiff_cossim_avg = vecdiff_cossim.mean(dim=2)
plt.figure(figsize=(7, 6))
sns.heatmap(vecdiff_cossim_avg, )
plt.title("mean cosine similarity of hidden state difference (h_{l+1} - h_l)")
plt.xlabel("layer #")
plt.ylabel("layer #")
plt.axis("image")
plt.tight_layout()
saveallforms(figdir, f"hidden_state_diff_cossim_heatmap_mean")
plt.show()
#%%
vecdiff_cossim_med = vecdiff_cossim.median(dim=2).values
plt.figure(figsize=(7, 6))
sns.heatmap(vecdiff_cossim_med, )
plt.title("median cosine similarity of hidden state difference (h_{l+1} - h_l)")
plt.xlabel("layer #")
plt.ylabel("layer #")
plt.axis("image")
plt.tight_layout()
saveallforms(figdir, f"hidden_state_diff_cossim_heatmap_med")
plt.show()
#%%
"""slice the diagonal entries 
=> hidden update similarity as a function of layer distance.
"""
clrseq = sns.color_palette("inferno", 6)#sns.color_palette("Blues_r", 7)
plt.figure()
for i, lag in enumerate([1,2,3,5,10,15]):
    plt.plot(torch.diag(vecdiff_cossim_avg, lag),
             label=f"lag={lag}", color=clrseq[i], lw=2, alpha=0.7)
plt.legend()
plt.xlabel("layer #")
plt.ylabel("cosine similarity")
plt.title("mean cosine similarity of hidden state diff\n(h_{l+1} - h_l, h_{l+k+1} - h_{l+k})")
plt.tight_layout()
saveallforms(figdir, f"hidden_state_diff_cossim_diag_lag_line_mean")
plt.show()
#%%
"""similarity of hidden states and modification of hidden states across layer."""
actdiff_inprod = torch.einsum("ljk, ljk -> lj", act_lyr_tsr[:-1, :, :], act_diff_tsr)
actdiff_cosim = actdiff_inprod / (act_lyr_tsr[:-1, :, :].norm(dim=-1) * act_diff_tsr.norm(dim=-1))
#%%
plt.figure(figsize=[6, 8])
sns.heatmap(actdiff_cosim.T, cmap="RdBu_r", center=0)
plt.title("cosine similarity of hidden state (h_{l}) and hidden state diff (h_{l+1} - h_l)")
plt.tight_layout()
saveallforms(figdir, f"hidden_state_state_diff_cossim_mat", fmts=["png"])
plt.show()
#%%
plt.figure()
plt.plot(actdiff_cosim.mean(dim=1), label="mean", color="k")
plt.fill_between(range(47),
                torch.quantile(actdiff_cosim, .25, dim=1),
                torch.quantile(actdiff_cosim, .75, dim=1), alpha=0.4, label="25-75 perctl")
plt.title("cosine similarity of hidden state (h_{l}) and hidden state diff (h_{l+1} - h_l)")
plt.ylabel("cosine similarity")
plt.xlabel("layer #")
plt.legend()
saveallforms(figdir, f"hidden_state_state_diff_cossim_line", )
plt.show()



#%%
"""Sparsity of the MLP hidden state activations"""
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
