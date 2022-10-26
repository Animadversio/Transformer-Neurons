"""
Loading and basic analysis of the wikitext corpus.
"""
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
#%%
dataset_len = dataset.filter(lambda x: len(x['text']) > 20)
