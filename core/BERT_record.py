import torch
import matplotlib.pyplot as plt
from core.layer_hook_utils import featureFetcher_module
from transformers import BertModel, BertConfig, BertTokenizer
from core.interp_utils import top_tokens_based_on_activation
# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()

# Initializing a model from the bert-base-uncased style configuration
model = BertModel(configuration)

# Accessing the model configuration
configuration = model.config

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#%%
def record_activations_for_text(text, model, tokenizer, layer_num=11):
    fetcher = featureFetcher_module()
    fetcher.record_module(model.encoder.layer[layer_num].intermediate.intermediate_act_fn, f"BERT_B{layer_num}_act")
    fetcher.record_module(model.encoder.layer[layer_num].intermediate.dense, f"BERT_B{layer_num}_fc")
    fetcher.record_module(model.encoder.layer[layer_num].output.dense, f"BERT_B{layer_num}_proj")
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens)
    fetcher.cleanup()
    for k, v in fetcher.activations.items():
        print(k, v.shape)
    return tokens, outputs, fetcher

from colorama import Fore, Back, Style
def highlight_text_top_words(tokens, activation, tokenizer):
    """Highlight top words regardless of context. """
    top_words, bottom_words = top_tokens_based_on_activation(activation, tokens, tokenizer, topk=10, bottomk=10)
    #%
    text = tokenizer.convert_ids_to_tokens(tokens, )
    #%
    text = [Fore.RED + word + Fore.RESET if word in top_words else word for word in text]
    text = [Fore.BLUE + word + Fore.RESET if word in bottom_words else word for word in text]
    text = "".join(text).replace("Ġ", " ").replace("Ċ", "\n").replace(" ##", "")\
        .replace(f" {Fore.RED}##", f"{Fore.RED}").replace(f" {Fore.BLUE}##", f"{Fore.BLUE}")
    # tokenizer.convert_tokens_to_string(text)
    return text


def highlight_text_quantile(tokens, activation, tokenizer, topq=0.75, bottomq=0.25):
    """Highlight words in context evoking above thresh activity. """
    top_thr = torch.quantile(activation, topq)
    bottom_thr = torch.quantile(activation, bottomq)
    above_top = activation > top_thr
    below_bottom = activation < bottom_thr
    text = tokenizer.convert_ids_to_tokens(tokens, )
    # text = "\n".join(wrap(" ".join(text), 80)).split()
    text = [Fore.RED + word + Fore.RESET if above_top[i] else word for i, word in enumerate(text)]
    text = [Fore.BLUE + word + Fore.RESET if below_bottom[i] else word for i, word in enumerate(text)]
    text = " ".join(text).replace("Ġ", " ").replace("Ċ", "\n").replace(" ##", "")\
        .replace(f" {Fore.RED}##", f"{Fore.RED}").replace(f" {Fore.BLUE}##", f"{Fore.BLUE}")
    # tokenizer.convert_tokens_to_string(text)
    return text


text="""
Little is known about Plato's early life and education. He belonged to an aristocratic and influential family. According to a disputed tradition, reported by doxographer Diogenes Laërtius, Plato's father Ariston traced his descent from the king of Athens, Codrus, and the king of Messenia, Melanthus. According to the ancient Hellenic tradition, Codrus was said to have been descended from the mythological deity Poseidon.
Through his mother, Plato was related to Solon. Plato's mother was Perictione, whose family boasted of a relationship with the famous Athenian lawmaker and lyric poet Solon, one of the seven sages, who repealed the laws of Draco (except for the death penalty for homicide). Perictione was sister of Charmides and niece of Critias, both prominent figures of the Thirty Tyrants, known as the Thirty, the brief oligarchic regime (404–403 BC), which followed on the collapse of Athens at the end of the Peloponnesian War (431–404 BC). According to some accounts, Ariston tried to force his attentions on Perictione, but failed in his purpose; then the god Apollo appeared to him in a vision, and as a result, Ariston left Perictione unmolested.
"""#The exact time and place of Plato's birth are unknown. Based on ancient sources, most modern scholars believe that he was born in Athens or Aegina[c] between 429 and 423 BC, not long after the start of the Peloponnesian War.[d] The traditional date of Plato's birth during the 87th or 88th Olympiad, 428 or 427 BC, is based on a dubious interpretation of Diogenes Laërtius, who says, "When [Socrates] was gone, [Plato] joined Cratylus the Heracleitean and Hermogenes, who philosophized in the manner of Parmenides. Then, at twenty-eight, Hermodorus says, [Plato] went to Euclides in Megara." However, as Debra Nails argues, the text does not state that Plato left for Megara immediately after joining Cratylus and Hermogenes. In his Seventh Letter, Plato notes that his coming of age coincided with the taking of power by the Thirty, remarking, "But a youth under the age of twenty made himself a laughingstock if he attempted to enter the political arena." Thus, Nails dates Plato's birth to 424/423.

layer_num = 9
tokens, _, fetcher = record_activations_for_text(text, model, tokenizer, layer_num=layer_num)
acttsr = fetcher[f'BERT_B{layer_num}_fc']
#%%
unit_idx = torch.randint(acttsr.size(2), size=(1,)).item() #2000
text_hl_thr = highlight_text_quantile(tokens[0],
                  acttsr[0, :, unit_idx], tokenizer)
topwords, botwords = top_tokens_based_on_activation(acttsr[0, :, unit_idx],
                            tokens[0], tokenizer, topk=20, bottomk=20)
print(text_hl_thr)
print(topwords, "\n", botwords )
#%%
from sklearn.cluster import KMeans
from umap import UMAP
umapper = UMAP(n_components=2, n_neighbors=10, min_dist=0.0, metric='cosine')
umapper.fit(acttsr[0].cpu().numpy())
u_embed = umapper.transform(acttsr[0].cpu().numpy())
#%%
plt.figure(figsize=(8, 8))
plt.scatter(u_embed[:, 0], u_embed[:, 1], s=9, alpha=0.5, cmap='Spectral');
plt.show()
