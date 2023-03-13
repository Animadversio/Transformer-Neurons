import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from core.layer_hook_utils import featureFetcher_module
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Initializing a GPT2 configuration
# configuration = GPT2Config()

# Initializing a model from the configuration
# model = GPT2Model(configuration) #Bug: this is not a pretrained model
model = GPT2Model.from_pretrained("gpt2")

# Accessing the model configuration
configuration = model.config


#%%
from core.interp_utils import top_tokens_based_on_activation
from textwrap import wrap
from colorama import Fore, Back, Style
def highlight_text_top_words(tokens, activation, tokenizer):
    """Highlight top words regardless of context. """
    top_words, bottom_words = top_tokens_based_on_activation(activation, tokens, tokenizer, topk=10, bottomk=10)
    #%
    text = tokenizer.convert_ids_to_tokens(tokens, )
    #%
    text = [Fore.RED + word + Fore.RESET if word in top_words else word for word in text]
    text = [Fore.BLUE + word + Fore.RESET if word in bottom_words else word for word in text]
    text = "".join(text).replace("Ġ", " ") #.replace("Ċ", "\t")
    # tokenizer.convert_tokens_to_string(text)
    return text


def highlight_text_quantile(tokens, activation, tokenizer, topq=0.75, bottomq=0.25):
    """Highlight words in context evoking above thresh activity.

    # text_hl_thr = highlight_text_quantile(tok_th[0], fetcher.activations['GPT2_B3_fc'][0, :, 10], tokenizer)
    # print(text_hl_thr)
    """
    top_thr = torch.quantile(activation, topq)
    bottom_thr = torch.quantile(activation, bottomq)
    above_top = activation > top_thr
    below_bottom = activation < bottom_thr
    text = tokenizer.convert_ids_to_tokens(tokens, )
    # text = "\n".join(wrap(" ".join(text), 80)).split()
    text = [Fore.RED + word + Fore.RESET if above_top[i] else word for i, word in enumerate(text)]
    text = [Fore.BLUE + word + Fore.RESET if below_bottom[i] else word for i, word in enumerate(text)]
    text = "".join(text).replace("Ġ", " ").replace("Ċ", "\n")
    return text


def model_forward_for_text(text, model, tokenizer, ):
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens)

    return tokens, outputs


def record_activations_for_text(text, model, tokenizer, layer_num=11):
    fetcher = featureFetcher_module()
    if type(layer_num) not in [tuple, list, range]:
        layer_num = [layer_num]
    for layer_i in layer_num:
        fetcher.record_module(model.h[layer_i].mlp.act, f"GPT2_B{layer_i}_act")
        fetcher.record_module(model.h[layer_i].mlp.c_fc, f"GPT2_B{layer_i}_fc")
        fetcher.record_module(model.h[layer_i].mlp.c_proj, f"GPT2_B{layer_i}_proj")
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens)
    fetcher.cleanup()
    for k, v in fetcher.activations.items():
        print(k, v.shape)
    return tokens, outputs, fetcher

#%%

#%%
text = """
Plato was an innovator of the written dialogue and dialectic forms in philosophy. 
He raised problems for what later became all the major areas of both theoretical philosophy and practical philosophy. His most famous contribution is the theory of Forms known by pure reason, in which Plato presents a solution to the problem of universals, known as Platonism (also ambiguously called either Platonic realism or Platonic idealism).
He is also the namesake of Platonic love and the Platonic solids.
"""
#%%
text = """
I went down yesterday to the Piraeus with Glaucon the son of Ariston, that I might offer up my prayers to the goddess; and also because I wanted to see in what manner they would celebrate the festival, which was a new thing. I was delighted with the procession of the inhabitants; but that of the Thracians was equally, if not more, beautiful. When we had finished our prayers and viewed the spectacle, we turned in the direction of the city; and at that instant Polemarchus the son of Cephalus chanced to catch sight of us from a distance as we were starting on our way home, and told his servant to run and bid us wait for him. The servant took hold of me by the cloak behind, and said: Polemarchus desires you to wait.
I turned round, and asked him where his master was.
There he is, said the youth, coming after you, if you will only wait.
Certainly we will, said Glaucon; and in a few minutes Polemarchus appeared, and with him Adeimantus, Glaucon's brother, Niceratus the son of Nicias, and several others who had been at the procession.
Socrates - POLEMARCHUS - GLAUCON - ADEIMANTUS
Polemarchus said to me: I perceive, Socrates, that you and our companion are already on your way to the city.
You are not far wrong, I said.
But do you see, he rejoined, how many we are?
Of course.
And are you stronger than all these? for if not, you will have to remain where you are.
May there not be the alternative, I said, that we may persuade you to let us go?
But can you persuade us, if we refuse to listen to you? he said.
Certainly not, replied Glaucon.
Then we are not going to listen; of that you may be assured.
Adeimantus added: Has no one told you of the torch-race on horseback in honour of the goddess which will take place in the evening?
With horses! I replied: That is a novelty. Will horsemen carry torches and pass them one to another during the race?
Yes, said Polemarchus, and not only so, but a festival will he celebrated at night, which you certainly ought to see. Let us rise soon after supper and see this festival; there will be a gathering of young men, and we will have a good talk. Stay then, and do not be perverse.
"""

# layer_num = 0
tokens, _, fetcher = record_activations_for_text(text, model, tokenizer, layer_num=range(12))
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
thresh = 1E-2
#%%
sparsity_col = []
for i in range(12):
    acttsr = fetcher[f'GPT2_B{i}_fc']
    fraction = (acttsr > thresh).sum() / acttsr.numel()
    sparsity_col.append(fraction)

plt.plot(range(12), sparsity_col)
plt.ylabel("pos activations fraction")
plt.show()
#%% Record the fraction of activated hidden units in MLP
layer_ids = []
sparsity_col = []
for layer_i in range(12):
    acttsr = fetcher[f'GPT2_B{layer_i}_fc']
    fractions = (acttsr > thresh).sum(dim=(0,1)) / acttsr.size(1)
    sparsity_col.append(fractions)
    layer_ids.append(layer_i * torch.ones_like(fractions, dtype=torch.int))

layer_ids = torch.cat(layer_ids)
frac_vec = torch.cat(sparsity_col)
spars_df = pd.DataFrame({"layer": layer_ids, "fraction":frac_vec})
#%%
#%% Record the fraction of activated hidden units in MLP
layer_ids = []
sparsity_col = []
for layer_i in range(12):
    acttsr = fetcher[f'GPT2_B{layer_i}_proj']
    fractions = (acttsr > thresh).sum(dim=(0,1)) / acttsr.size(1)
    sparsity_col.append(fractions)
    layer_ids.append(layer_i * torch.ones_like(fractions, dtype=torch.int))

layer_ids = torch.cat(layer_ids)
frac_vec = torch.cat(sparsity_col)
spars_df_proj = pd.DataFrame({"layer": layer_ids, "fraction":frac_vec})
#%%
plt.figure(figsize=[6, 8])
sns.violinplot(data=spars_df, x="layer", y="fraction", cut=0)
plt.title("GPT2 MLP hidden unit activated fraction")
plt.show()
#%%
sns.lineplot(data=spars_df, x="layer", y="fraction", )
plt.title("GPT2 MLP hidden unit activated fraction")
plt.show()
#%%
sns.lineplot(data=spars_df_proj, x="layer", y="fraction", )
plt.title("GPT2 MLP proj out unit activated fraction")
plt.show()
"""The projection out unit roughly has half unit > 0 half < 0 """
#%%
acttsr = fetcher[f'GPT2_B{layer_num}_act']
unit_idx = torch.randint(3000, size=(1,)).item() #2000
text_hl_thr = highlight_text_quantile(tokens[0],
                  acttsr[0, :, unit_idx], tokenizer)
topwords, botwords, top_acts, bottom_acts = top_tokens_based_on_activation(acttsr[0, :, unit_idx],
                            tokens[0], tokenizer, topk=20, bottomk=20)
print(f"Unit id {unit_idx}")
print(f"Top words: {topwords} ({top_acts[0]:.3f}-{top_acts[-1]:.3f})")
print(f"Bottom words: {botwords} ({bottom_acts[0]:.3f}-{bottom_acts[-1]:.3f})")
print(text_hl_thr)