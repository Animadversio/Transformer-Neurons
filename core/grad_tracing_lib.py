import torch
from easydict import EasyDict as edict
import warnings
from os.path import join


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



def forward_gradhook_vars(model, token_ids, module_str="block", module_fun=None):
    """older version, only hook and record one type of module"""
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


def grad_tracing(text, tokenizer, model, device="cpu", ):
    model.to(device)
    token_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    outputs, gradvar_stores, act_stores = forward_record_gradhook_vars_all(model, token_ids, )
    max_ids = torch.argmax(outputs.logits[0, -1, :])
    maxlogit = outputs.logits[0, -1, max_ids]
    print(text)
    top_token = tokenizer.convert_ids_to_tokens(max_ids.item()).replace('\u0120', '')
    print(f"Top token is '{top_token}', with logit {maxlogit.item():.3f}")
    gradsavedict = {}
    actsavedict = {}
    gradmapsavedict = {}
    actmapsavedict = {}
    for sub_module in ["block", "attn", "mlp", "ln_2"]:  #
        act_store = [*act_stores[sub_module].values()]
        grad_maps = torch.autograd.grad(maxlogit,
                                        [*gradvar_stores[sub_module].values()],
                                        retain_graph=True, )  # create_graph=True
        if grad_maps[0].ndim == 3:  # "block", "attn",
            grad_map_tsr = torch.cat(grad_maps, dim=0).cpu()
            act_map_tsr = torch.cat(act_store, dim=0).cpu()
        elif grad_maps[0].ndim == 2:  # "mlp", "ln_2", the Batch dim is suppressed
            grad_map_tsr = torch.stack(grad_maps, dim=0).cpu()
            act_map_tsr = torch.stack(act_store, dim=0).cpu()
        else:
            raise ValueError
        grad_map_norm_layers = grad_map_tsr.norm(dim=-1)
        act_map_norm_layers = act_map_tsr.norm(dim=-1)
        gradsavedict[sub_module] = grad_map_tsr
        actsavedict[sub_module] = act_map_tsr
        gradmapsavedict[sub_module] = grad_map_norm_layers
        actmapsavedict[sub_module]  = act_map_norm_layers

    return gradsavedict, actsavedict, gradmapsavedict, actmapsavedict
    # torch.save({"grad": gradsavedict,  "act": actsavedict, },
    #            join(expdir, f"grad_act_save.pt"))
    # torch.save({"gradmap": gradmapsavedict, "actmap": actmapsavedict},
    #            join(expdir, f"grad_act_map_save.pt"))


import matplotlib.pyplot as plt
import seaborn as sns
from core.plot_utils import saveallforms
def visualize_grad_trace(text, tokenizer, gradsavedict, actsavedict, expdir="", ):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
    tokens = [t.replace("Ġ", "") for t in tokens]
    for sub_module in ["block", "attn", "mlp", "ln_2"]:  #
        grad_map_tsr = gradsavedict[sub_module]
        act_map_tsr = actsavedict[sub_module]
        grad_map_norm_layers = grad_map_tsr.norm(dim=-1)
        plt.figure()
        sns.heatmap(grad_map_norm_layers.T)
        plt.gca().set_yticks(0.5 + torch.arange(len(tokens)))
        plt.gca().set_yticklabels(tokens, rotation=0)
        plt.xlabel("Layer")
        plt.title(f"Gradient L2 norm heat map ({sub_module})")
        plt.tight_layout()
        saveallforms(expdir, f'grad_norm_{sub_module}_heatmap')
        plt.show()

        grad_act_prod = torch.einsum("lth,lth->lt", grad_map_tsr, act_map_tsr)
        plt.figure()
        sns.heatmap(grad_act_prod.T)
        plt.gca().set_yticks(0.5 + torch.arange(len(tokens)))
        plt.gca().set_yticklabels(tokens, rotation=0)
        plt.xlabel("Layer")
        plt.title(f"Gradient Activation Projection heat map ({sub_module})")
        plt.tight_layout()
        saveallforms(expdir, f'grad_act_dot_{sub_module}_heatmap')
        plt.show()

