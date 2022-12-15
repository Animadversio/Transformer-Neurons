def match_subject_decode(tokenizer, token_ids, subject_str):
    for subj_strt in range(len(token_ids)):
        for subj_end in range(subj_strt + 1, len(token_ids)):
            subj_prefix_raw = tokenizer.decode(token_ids[subj_strt:subj_end])
            subj_prefix = subj_prefix_raw.strip()
            if subj_prefix == subject_str[:len(subj_prefix)]:
                if subj_prefix == subject_str:
                    return subj_strt, subj_end
            elif subj_prefix == (subject_str + "'"):
                return subj_strt, subj_end
            else:
                break
    raise ValueError


def find_substr_loc(tokenizer, text, substr, uncase=False):
    """Find the location of a substring in a text in terms of token ids"""
    if uncase:
        substr = substr.lower()
        text = text.lower()
    token_ids = tokenizer.encode(text,)
    for subj_strt in range(len(token_ids)):
        for subj_end in range(subj_strt + 1, len(token_ids)):
            subj_prefix_raw = tokenizer.decode(token_ids[subj_strt:subj_end])
            subj_prefix = subj_prefix_raw.strip()
            if subj_prefix == substr[:len(subj_prefix)]:
                if subj_prefix == substr:
                    return subj_strt, subj_end
            elif subj_prefix == (substr + "'"):
                return subj_strt, subj_end
            else:
                continue
    raise ValueError