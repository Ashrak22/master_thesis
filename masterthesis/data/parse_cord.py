import os
from typing import Dict, List
import json

import psutil
import pandas
import datasets
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
import numpy as np

from .functions import _tokenize

_cord_location = os.path.join("masterthesis", "data","2022-04-28")
# datasets.set_caching_enabled(False)
def _cord_19_dataset(tokenizer_name: str = "sshleifer/distilbart-cnn-12-6", max_length: int = 1024, recreate_split: bool = False, retokenize: bool = False, prefix:str = None, test: bool = False) -> DatasetDict:
    if prefix is not None:
        _cord_location = os.path.join(prefix, "2022-04-28")
    ds_csv = os.path.join(_cord_location, "metadata.csv")
    ds_path = os.path.join(_cord_location, 'ds')
    changed = False
    if not os.path.exists(ds_path):
        df = pandas.read_csv(ds_csv, dtype={'cord_uid': "string", 'sha': "string", 'source_x': "string", 'title': "string", 'doi': "string", 'pmcid': "string", 'pubmed_id': "string",
        'license': "string", 'abstract': "string", 'publish_time': "string", 'authors': "string", 'journal': "string", 'mag_id': "string",
        'who_covidence_id': "string", 'arxiv_id': "string", 'pdf_json_files': "string", 'pmc_json_files': "string",
        'url': "string", 's2_id': "string"})
        df = df[['cord_uid', 'title', 'license', 'abstract', 'pdf_json_files', 'pmc_json_files', 'pmcid', 'doi']]
        df['id'] = np.arange(len(df))
        print(df)
        ds = Dataset.from_pandas(df)
        ds = ds.filter(filter_non_existing)
        changed = True
        
    else:
        ds = datasets.load_from_disk(ds_path)
    
    if 'text' not in ds.column_names and 'text' not in ds['train'].column_names:
        ds = ds.map(read_text)
        changed = True
    
    if 'text_length' not in ds.column_names and 'text_length' not in ds['train'].column_names:
        ds = ds.filter(lambda x: x['abstract'] != None).map(text_lengths)
        ds = ds.filter(lambda x: x['text_length'] > 100 and x['abstract_length'] > 50)
        changed = True
    
    if ('input_ids' not in ds.column_names and 'input_ids' not in ds['train'].column_names) or retokenize:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        ds = ds.map(lambda x: _tokenize(x, tokenizer, max_length), batched=True)
        changed = True

    if test:
        ds = ds.map(lambda x: lengths(x))
        changed = True

    if recreate_split:
        ds = create_split(ds)
        changed = True

    if changed:
        ds.save_to_disk(ds_path)
    
    return ds #kill dat cpu

def read_text(example):    
    if example['pmc_json_files'] is not None:
        path = example['pmc_json_files'].split('; ')[0]
        path = os.path.join(_cord_location, path.replace('/', os.path.sep))
        type_name = 'pmc'
    elif example['pdf_json_files'] is not None:
        path = example['pdf_json_files'].split('; ')[0]
        path = os.path.join(_cord_location, path.replace('/', os.path.sep))
        type_name = 'pdf'
    else:
        return {}
    
    with open(path) as fp:
        json_dict = json.load(fp)
        text = []
        for sec in json_dict['body_text']:
            text.append(sec['text'])
        return {'text': '\n'.join(text), 'type': type_name}

def create_split(ds: Dataset) -> DatasetDict:
    train_test = ds.train_test_split(test_size=0.01)
    val_test = train_test['test'].train_test_split(test_size=0.5)
    return DatasetDict(
        {
            'train': train_test['train'],
            'val': val_test['train'],
            'test': val_test['test']
        }
    )

def lengths(example):
    return {'labels_length': len(example['labels']), 'input_ids_length': len(example['input_ids'])}

def text_lengths(example):
    return {'text_length': len(example['text'].replace('\n', ' ').split(' ')), 'abstract_length': len(example['abstract'].replace('\n', ' ').split(' '))}

def filter_non_existing(x):
    return not(x['pdf_json_files'] == None and x['pmc_json_files'] == None)


def get_sample(ds: DatasetDict) -> List[Dict]:
    sample = ds['train'].shuffle(seed=42).select(range(1000))
    return sample[:3]

if __name__ == "__main__":
    ds = _cord_19_dataset()
    print(ds.features)
    #print(ds.unique('license'))
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Number of files in dataset : {ds.dataset_size}")
    ds.set_format("pandas")
    df = ds[:]
    print(df[['text_length', 'abstract_length']].describe())
    print(df[df['text_length'] == 1].head(10))
    print(df[df['abstract_length'] == 2].head(10))
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")