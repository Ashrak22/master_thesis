from argparse import ArgumentError
from typing import Any, Dict
from datasets import DatasetDict

from .parse_cord import _cord_19_dataset
from .parse_pubmed import _pubmed_dataset

def get_dataset(name: str, tokenizer: str, max_length: int, retokenize:bool = False, prefix:str = None, text_rank_config: Dict[str, Any] = None, lstm_config: Dict[str, Any] = None, type: str = 'transformer') -> DatasetDict:
    if name == 'cord-19':
        return _cord_19_dataset(tokenizer_name=tokenizer, max_length=max_length, retokenize=retokenize, prefix=prefix, text_rank_config=text_rank_config, type=type)
    elif name == 'pubmed':
        return _pubmed_dataset(tokenizer_name=tokenizer, max_length=max_length, retokenize=retokenize, prefix=prefix, text_rank_config=text_rank_config, lstm_config=lstm_config, type=type)
    else:
        raise ArgumentError(name, 'Uknown dataset name')


