def top_tokens_based_on_activation(activation, tokens, tokenizer, topk=5, bottomk=5, threshold=0.5):
    """Highlight the tokens based on the activation
    Args:
        activation: (torch.tensor) the activation of the tokens
        text: (str) the text
        topk: (int) the number of tokens to highlight
        threshold: (float) the threshold of the activation to highlight

    top_tokens_based_on_activation(fetcher.activations['GPT2_B10_fc'][0, :, 50], tok_th[0], tokenizer, topk=10, bottomk=10)
    """
    activation = activation.squeeze(0)
    assert len(activation.shape) == 1
    assert activation.shape[0] == len(tokens)
    #%
    activation = activation.cpu().numpy()
    activation = (activation - activation.min()) / (activation.max() - activation.min())
    # activation = activation / activation.sum()
    #%
    topk = min(topk, activation.shape[0])
    topk_idx = (-activation).argsort()[:topk]
    bottomk = min(bottomk, activation.shape[0])
    bottomk_idx = activation.argsort()[:bottomk]
    top_acts = activation[topk_idx]
    bottom_acts = activation[bottomk_idx]
    #%
    top_words = tokenizer.convert_ids_to_tokens(tokens[topk_idx])  # .decode
    bottom_words = tokenizer.convert_ids_to_tokens(tokens[bottomk_idx])  # .decode
    return top_words, bottom_words, top_acts, bottom_acts  # " ".join(text)
