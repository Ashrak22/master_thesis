from argparse import ArgumentError
from datasets import DatasetDict

from .parse_cord import _cord_19_dataset
from .parse_pubmed import _pubmed_dataset

def get_dataset(name: str, tokenizer: str, max_length: int, retokenize:bool = False, prefix:str = None) -> DatasetDict:
    if name == 'cord-19':
        return _cord_19_dataset(tokenizer_name=tokenizer, max_length=max_length, retokenize=retokenize, prefix=prefix)
    elif name == 'pubmed':
        return _pubmed_dataset(tokenizer_name=tokenizer, max_length=max_length, retokenize=retokenize, prefix=prefix)
    else:
        raise ArgumentError(name, 'Uknown dataset name')


