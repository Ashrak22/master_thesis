from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

def encode(text:str, model:SentenceTransformer) -> Tuple[List[str], torch.Tensor]:
    sentences = [s.lower().strip() for s in sent_tokenize(text)]
    # TODO: Catch that punkt is not downloaded ba catching LookupError
    # nltk.download('punkt_tab')
    if len(sentences) < 1:
        print(text)
        return sentences, torch.from_numpy(np.zeros((1,1)))
    return sentences, model.encode(sentences, convert_to_numpy=False, convert_to_tensor=True)

def select_top_k(ranked_sentences: Dict[str, List[Any]], top_k:int) -> List[Tuple[int, str]]:
    max_len = len(ranked_sentences['positions'])
    positions = ranked_sentences['positions'][:min(top_k, max_len)]
    sentences = ranked_sentences['sentences'][:min(top_k, max_len)]
    ranked_sentences = zip(positions, sentences)
    ranked_sentences = sorted(((i,s) for _,(i,s) in enumerate(ranked_sentences)))
    return ranked_sentences

def generate_text(ranked_sentences: List[Tuple[int, str]]):
    return ' '.join([s for i, s in ranked_sentences])