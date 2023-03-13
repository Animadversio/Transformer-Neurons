import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
# Accessing the model configuration
configuration = model.config
model.requires_grad_(False)
model.eval()
#%%
text = "During the class on Tuesday, James learned that, the residence of the Pope, Vatican is in the city of"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"],
                    output_hidden_states=True, output_attentions=True)
loss = outputs.loss
logits = outputs.logits
all_hidden = (outputs.hidden_states)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from core.plot_utils import saveallforms
tok_ids = tokenizer(text)["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(tok_ids)
tokens = [token.replace("Ġ", "") for token in tokens]

plt.figure(figsize=[8, 6])
sns.heatmap(outputs.attentions[0][0, :].sum(dim=0))
plt.gca().set_xticklabels(tokens, rotation=90, fontsize=12)
plt.gca().set_yticklabels(tokens, rotation=0, fontsize=12)
plt.ylabel("Source of Attn", fontsize=14)
plt.xlabel("Target of Attn", fontsize=14)
plt.tight_layout()
plt.show()
#%%
posenc_mat = model.transformer.wpe(torch.arange(len(tok_ids)))
#%%
inprodmat = posenc_mat @ posenc_mat.T
attnmask = torch.triu(torch.ones_like(inprodmat), 1)*torch.finfo(torch.float).min
attnmat = torch.softmax(inprodmat+attnmask, dim=-1)
#%%
sns.heatmap(attnmat)
plt.show()
