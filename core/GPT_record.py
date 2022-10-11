import torch
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from core.layer_hook_utils import featureFetcher_module
#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Initializing a GPT2 configuration
configuration = GPT2Config()

# Initializing a model from the configuration
model = GPT2Model(configuration)

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
    fetcher.record_module(model.h[layer_num].mlp.act, f"GPT2_B{layer_num}_act")
    fetcher.record_module(model.h[layer_num].mlp.c_fc, f"GPT2_B{layer_num}_fc")
    fetcher.record_module(model.h[layer_num].mlp.c_proj, f"GPT2_B{layer_num}_proj")
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens)
    fetcher.cleanup()
    for k, v in fetcher.activations.items():
        print(k, v.shape)
    return tokens, outputs, fetcher
# text_hl_thr = highlight_text_quantile(tokens[0], fetcher.activations['GPT2_B10_fc'][0, :, 10], tokenizer)
# print(text_hl_thr)
# text_hl = highlight_text_top_words(tok_th[0], fetcher.activations['GPT2_B10_fc'][0, :, 10], tokenizer)
# print(text_hl)
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

layer_num = 12
tokens, _, fetcher = record_activations_for_text(text, model, tokenizer, layer_num=layer_num)
unit_idx = torch.randint(3000, size=(1,)).item() #2000
text_hl_thr = highlight_text_quantile(tokens[0],
                  fetcher[f'GPT2_B{layer_num}_fc'][0, :, unit_idx], tokenizer)
print(text_hl_thr)

#%%
text="""
Little is known about Plato's early life and education. He belonged to an aristocratic and influential family. According to a disputed tradition, reported by doxographer Diogenes Laërtius, Plato's father Ariston traced his descent from the king of Athens, Codrus, and the king of Messenia, Melanthus. According to the ancient Hellenic tradition, Codrus was said to have been descended from the mythological deity Poseidon.
Through his mother, Plato was related to Solon. Plato's mother was Perictione, whose family boasted of a relationship with the famous Athenian lawmaker and lyric poet Solon, one of the seven sages, who repealed the laws of Draco (except for the death penalty for homicide). Perictione was sister of Charmides and niece of Critias, both prominent figures of the Thirty Tyrants, known as the Thirty, the brief oligarchic regime (404–403 BC), which followed on the collapse of Athens at the end of the Peloponnesian War (431–404 BC). According to some accounts, Ariston tried to force his attentions on Perictione, but failed in his purpose; then the god Apollo appeared to him in a vision, and as a result, Ariston left Perictione unmolested.
The exact time and place of Plato's birth are unknown. Based on ancient sources, most modern scholars believe that he was born in Athens or Aegina[c] between 429 and 423 BC, not long after the start of the Peloponnesian War.[d] The traditional date of Plato's birth during the 87th or 88th Olympiad, 428 or 427 BC, is based on a dubious interpretation of Diogenes Laërtius, who says, "When [Socrates] was gone, [Plato] joined Cratylus the Heracleitean and Hermogenes, who philosophized in the manner of Parmenides. Then, at twenty-eight, Hermodorus says, [Plato] went to Euclides in Megara." However, as Debra Nails argues, the text does not state that Plato left for Megara immediately after joining Cratylus and Hermogenes. In his Seventh Letter, Plato notes that his coming of age coincided with the taking of power by the Thirty, remarking, "But a youth under the age of twenty made himself a laughingstock if he attempted to enter the political arena." Thus, Nails dates Plato's birth to 424/423.
"""

layer_num = 0
tokens, _, fetcher = record_activations_for_text(text, model, tokenizer, layer_num=layer_num)
acttsr = fetcher[f'GPT2_B{layer_num}_act']
unit_idx = torch.randint(acttsr.size(2), size=(1,)).item()  # 2000
text_hl_thr = highlight_text_quantile(tokens[0], acttsr[0, :, unit_idx], tokenizer)
print(text_hl_thr)
#%%

U, S, V = torch.svd(acttsr[0])
#%%
pc_idx = torch.randint(V.size(0), size=(1,)).item()
text_hl_thr = highlight_text_quantile(tokens[0], V[pc_idx, :], tokenizer)
print(text_hl_thr)

#%% SVD
mat = fetcher[f'GPT2_B{layer_num}_proj'][0]
U, S, V = torch.svd(mat)
#%%
plt.plot(torch.cumsum(S[0:], dim=0)/torch.sum(S[0:]))
plt.show()
#%%


#%% Dev zone

layer_num = 11
fetcher = featureFetcher_module()
# fetcher.record_module(model.h[10].attn, "GPT2_B10_attn")
fetcher.record_module(model.h[layer_num].mlp.act, f"GPT2_B{layer_num}_act")
fetcher.record_module(model.h[layer_num].mlp.c_fc, f"GPT2_B{layer_num}_fc")
fetcher.record_module(model.h[layer_num].mlp.c_proj, f"GPT2_B{layer_num}_proj")

input = tokenizer(text)['input_ids']
tok_th = torch.tensor(input).cpu().unsqueeze(0)
#%%
with torch.no_grad():
    out = model(tok_th)

#%%
# print("atten activation:", fetcher.activations['GPT2_B10_attn'].shape)
print("hidden activation fc:", fetcher.activations['GPT2_B3_fc'].shape)
print("hidden activation:",    fetcher.activations['GPT2_B3_act'].shape)
print("hidden activation proj:", fetcher.activations['GPT2_B3_proj'].shape)
print("output hidden state:", out.last_hidden_state.shape)
#%%
fetcher.cleanup()

#%%
