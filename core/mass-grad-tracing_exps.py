import os
from os.path import join
import torch
import torch.nn as nn  # import ModuleList
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertModel, BertForMaskedLM, BertConfig, BertTokenizer
from core.layer_hook_utils import featureFetcher_module
from core.interp_utils import top_tokens_based_on_activation, topk_decode
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
# Accessing the model configuration
configuration = model.config
model.requires_grad_(False)
model.eval()
#%%
import json
k1000_data = json.load(open("dataset\\known_1000.json", 'r'))
#%%
