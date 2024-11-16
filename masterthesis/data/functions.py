from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch

from models import TextRank, LSTMExtractor
from models.functions import encode, select_top_k, generate_text

def _calculate_text_rank(examples, model: TextRank):
    summaries = []
    for example in examples['text']:
        sentences, embeddings = model.encode(example)
        summaries.append(model.textrank_raw(sentences=sentences, embeddings=embeddings))
    return {'preselect_raw': summaries}

def _tokenize(examples, tokenizer, max_length: int, use_text_rank:bool = False):
    key = 'text_summary' if use_text_rank else 'text'
    tokens = [tokenizer(example, max_length=max_length, truncation=True) for example in examples[key]]
    tokens_length = [len(x['input_ids']) for x in tokens]
    token_list = [token['input_ids'] for token in tokens]
    attention_masks = [token['attention_mask'] for token in tokens]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['abstract'], max_length=max_length, truncation=True)

    return {"input_ids": token_list, 'attention_mask': attention_masks, 'labels': labels['input_ids'], 'lengths': tokens_length}

def _embed_sentences(examples, model: SentenceTransformer):
    sentences = []
    embeddings = []
    for example in examples['text']:
        (sent, emb) = encode(example, model)
        sentences.append(sent)
        embeddings.append(emb)
    return {'embeddings': embeddings, 'sentences': sentences}

def _embed_abstract(examples, model: SentenceTransformer):
    embeddings = []
    for example in examples['abstract']:
        (_, emb) = encode(example, model)
        embeddings.append(emb)
    return {'embedded_abstract': embeddings}

def _calculate_targets(examples):
    results = []
    for i, example in enumerate(examples['embeddings']):
        abstract_embeddings = examples['embedded_abstract'][i]
        sim_matrix = util.cos_sim(example, abstract_embeddings)
        result = sim_matrix.max(dim=1).values
        assert sim_matrix.shape[0] == result.shape[0]
        results.append(result)
    return {'target': results}

def _calculate_lstm(examples, model: LSTMExtractor):
    summaries = []
    model.eval()
    for i, sentences in enumerate(examples['sentences']):
        summaries.append(model.extract_raw(sentences=sentences, embeddings=torch.Tensor(examples['embeddings'][i]).cuda()))
    return {'preselect_raw': summaries}

def _calculate_top_k(examples, top_k: int):
    summaries = []
    for example in examples['preselect_raw']:
        sentences = select_top_k(ranked_sentences=example, top_k=top_k)
        summaries.append(generate_text(sentences))
    return {'text_summary': summaries}