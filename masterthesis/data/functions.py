def _tokenize(examples, tokenizer, max_length: int):
    tokens = [tokenizer(example, max_length=max_length, truncation=True) for example in examples['text']]
    token_list = [token['input_ids'] for token in tokens]
    attention_masks = [token['attention_mask'] for token in tokens]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['abstract'], max_length=max_length, truncation=True)

    return {"input_ids": token_list, 'attention_mask': attention_masks, 'labels': labels['input_ids']}