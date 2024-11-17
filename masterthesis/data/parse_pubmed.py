import json
import os
from typing import Any, Dict, List
import timeit
import difflib
import datasets
import psutil
import shutil
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from data.functions import _tokenize, _calculate_text_rank, _calculate_top_k, _embed_sentences, _embed_abstract, _calculate_targets, _calculate_lstm
from models import TextRank, LSTMExtractor

#_pubmed_location = os.path.join("masterthesis", "data","pubmed")

def _pubmed_dataset(tokenizer_name: str = "sshleifer/distilbart-cnn-12-6", max_length: int = 1024, retokenize:bool = False, prefix:str = None, text_rank_config: Dict[str, Any] = None, lstm_config: Dict[str, Any] = None,type: str = None) -> DatasetDict:
    _pubmed_location = os.path.join("masterthesis", "data","pubmed")
    if prefix is not None:
        _pubmed_location = os.path.join(prefix, "pubmed")
    data_files = {"train": os.path.join(_pubmed_location, "train.txt"), "val": os.path.join(_pubmed_location, "val.txt"), "test": os.path.join(_pubmed_location, "test.txt")}
    ds_path = os.path.join(_pubmed_location, 'ds'+'lstm')
    changed = False

    if not os.path.exists(ds_path):
        ds = load_dataset("json", data_files=data_files)
    else:
        ds = datasets.load_from_disk(ds_path)
    
    if 'abstract' not in ds['train'].column_names:
        print('selecting abstracts')
        ds = ds.map(lambda x: _clean_dataset_batched(x), batched=True)
        ds = ds.filter(lambda x: x['text_length'] > 100 and x['abstract_length'] > 50)
        changed = True
    
    if text_rank_config is not None and 'use_text_rank' in text_rank_config and text_rank_config['use_text_rank']:
        if 'preselect_raw' not in ds['train'].column_names or text_rank_config['recalculate_text_rank']:
            print('calculating Textrabk')
            model = TextRank(text_rank_config['model_name'], text_rank_config['similarity_function'])
            ds = ds.map(lambda x: _calculate_text_rank(x, model), batched=True)
            changed = True
    
    if type == 'lstm' or (lstm_config is not None and 'use_lstm' in lstm_config and lstm_config['use_lstm']):
        if 'sentences' not in ds['train'].column_names:
            print('embedding for LSTM')
            sbert = SentenceTransformer(text_rank_config['model_name'])
            ds = ds.map(lambda x: _embed_sentences(x, sbert), batched=True)
            changed = True
    
    if lstm_config is not None and 'use_lstm' in lstm_config and lstm_config['use_lstm']:
        if 'preselect_raw' not in ds['train'].column_names or lstm_config['recalculate_lstm']:
            print('calculating LSTM')
            model = LSTMExtractor.load(lstm_config['path']).cuda()
            ds = ds.map(lambda x: _calculate_lstm(x, model), batched=True)
            changed = True

    if (lstm_config is not None and 'use_lstm' in lstm_config and lstm_config['use_lstm']) or (text_rank_config is not None and 'use_text_rank' in text_rank_config and text_rank_config['use_text_rank']):
        print('using pre-select')
        config = lstm_config if lstm_config is not None and lstm_config['use_lstm'] else text_rank_config
        if config is not None and 'text_summary' not in ds['train'].column_names or config['reselect_top_k']:
            print('preselecting Top K sentences')
            ds = ds.map(lambda x: _calculate_top_k(x, config['top_k_sentences']), batched=True)
            changed = True
        
    if 'input_ids' not in ds['train'].column_names or retokenize:
        print('tokenizing for BART')
        use_preselect = False if config is None else (config['use_lstm'] if 'use_lstm' in config else config['use_text_rank']) 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        ds = ds.map(lambda x: _tokenize(x, tokenizer, max_length, use_preselect), batched=True)
        changed = True
        

    if type == 'lstm':
        if 'embedded_abstract' not in ds['train'].column_names:
            sbert = SentenceTransformer(text_rank_config['model_name'])
            ds = ds.map(lambda x: _embed_abstract(x, sbert), batched=True)
            changed = True
        
        if 'target' not in ds['train'].column_names:
            ds = ds.map(lambda x: _calculate_targets(x), batched=True)
            changed = True

    if changed:
        ds.save_to_disk(ds_path+'temp')
        del ds
        shutil.rmtree(ds_path)
        os.rename(ds_path+'temp', ds_path)
        ds = datasets.load_from_disk(ds_path)
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
    for k in [40, 30, 20, 15, 10, 5]:
        tr_config = {
            'use_text_rank': True,
            'model_name': "all-mpnet-base-v2", 
            'similarity_function': "euclidean_distance",
            'top_k_sentences': k,
            'reselect_top_k': True,
            'recalculate_text_rank': False
        }
        print(f"sentence count: {k}")
        ds = _pubmed_dataset(max_length=10240, retokenize=True, text_rank_config=tr_config)
        print(ds["train"].features)
        print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        ds.set_format("pandas")
        df_train = ds["train"][:]
        #print(df_train.loc[df_train["labels"] != None, ["labels"]])
        print(ds['train'][:][[ 'lengths']].describe())
        #print(f"Number of files in dataset : {ds.cache_files}")