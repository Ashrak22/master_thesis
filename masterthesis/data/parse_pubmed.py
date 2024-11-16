import json
import os
from typing import Dict, List
import timeit
import difflib
import datasets
import psutil
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer

from .functions import _tokenize

#_pubmed_location = os.path.join("masterthesis", "data","pubmed")

def _pubmed_dataset(tokenizer_name: str = "sshleifer/distilbart-cnn-12-6", max_length: int = 1024, retokenize:bool = False, prefix:str = None) -> DatasetDict:
    _pubmed_location = os.path.join("masterthesis", "data","pubmed")
    if prefix is not None:
        _pubmed_location = os.path.join(prefix, "pubmed")
    data_files = {"train": os.path.join(_pubmed_location, "train.txt"), "val": os.path.join(_pubmed_location, "val.txt"), "test": os.path.join(_pubmed_location, "test.txt")}
    ds_path = os.path.join(_pubmed_location, 'ds')
    changed = False

    if not os.path.exists(ds_path):
        ds = load_dataset("json", data_files=data_files)
    else:
        ds = datasets.load_from_disk(ds_path)
    
    if 'abstract' not in ds['train'].column_names:
        ds = ds.map(lambda x: _clean_dataset_batched(x), batched=True)
        ds = ds.filter(lambda x: x['text_length'] > 100 and x['abstract_length'] > 50)
        changed = True

    if 'input_ids' not in ds['train'].column_names or retokenize:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        ds = ds.map(lambda x: _tokenize(x, tokenizer, max_length), batched=True)
        changed = True
    
    
    if changed:
        ds.save_to_disk(ds_path)
    return ds

    

def _clean_dataset_batched(examples):
    texts = ['\n'.join(example) for example in examples['article_text']]
    abstracts = ['\n'.join(example).replace('<S>','').replace('</S>', '') for example in examples['abstract_text']]
    return{"text": texts, "abstract": abstracts, "text_length": [len(article.replace('\n', ' ').split(' ')) for article in texts], "abstract_length": [len(abstract.replace('\n', ' ').split(' ')) for abstract in abstracts]}

def _get_sample(ds: DatasetDict) -> List[Dict]:
    sample = ds['train'].shuffle(seed=42).select(range(1000))
    return sample[:3]


if __name__ == "__main__":
    datasets.disable_caching()
    ds = _pubmed_dataset()
    print(ds["train"].features)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    ds.set_format("pandas")
    df_train = ds["train"][:]
    #print(df_train.loc[df_train["labels"] != None, ["labels"]])
    print(ds['train'][:][[ 'text_length', 'abstract_length']].describe())
    print(ds['val'][:][[ 'text_length', 'abstract_length']].describe())
    print(ds['test'][:][[ 'text_length', 'abstract_length']].describe())
    #print(f"Number of files in dataset : {ds.cache_files}")