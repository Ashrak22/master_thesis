from __future__ import annotations
from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from sentence_transformers import SentenceTransformer

from models.functions import encode, select_top_k, generate_text

class LSTMExtractor(nn.Module):
    def __init__(self, distance_metric:str, input_size:int, hidden_size:int, use_batch_norm: bool = False, use_layer_norm:bool = False, dropout:float = 0.0, function:str = 'relu') -> None:
        super().__init__()
        self.metric = distance_metric
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.nn_1 = nn.Linear(hidden_size*2, hidden_size)
        self.nn_2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.nn_3 = nn.Linear(int(hidden_size/2), 1)
        self.relu = nn.LeakyReLU() if function == 'relu' else nn.GELU()
        self.dropout = nn.Dropout(p = dropout)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = use_batch_norm
        self.layer_norm = use_layer_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size*2)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(int(hidden_size/2))
        
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(hidden_size*2)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.ln3 = nn.LayerNorm(int(hidden_size/2))
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        (batch_size, length, hidden_size) = batch.size() 
        lstm_rs, _ = self.lstm(batch)
        lstm_rs = lstm_rs.reshape((batch_size*length, self.hidden_size*2))
        if(self.batch_norm and batch_size*length > 1):
            lstm_rs = self.bn1(lstm_rs)
        if(self.layer_norm):
            lstm_rs = self.ln1(lstm_rs)
        lstm_rs = self.dropout(lstm_rs)

        lstm_rs = self.relu(self.nn_1(lstm_rs))
        if(self.batch_norm and batch_size*length > 1):
            lstm_rs = self.bn2(lstm_rs)
        if(self.layer_norm):
            lstm_rs = self.ln2(lstm_rs)
        lstm_rs = self.dropout(lstm_rs)

        lstm_rs = self.relu(self.nn_2(lstm_rs))
        if(self.batch_norm and batch_size*length > 1):
            lstm_rs = self.bn3(lstm_rs)
        if(self.layer_norm):
            lstm_rs = self.ln3(lstm_rs)
        lstm_rs = self.dropout(lstm_rs)

        lstm_rs = self.relu(self.nn_3(lstm_rs))
        lstm_rs = lstm_rs.reshape(batch_size, length)
        return self.sigmoid(lstm_rs)
    
    def extract_raw(self, text: str, sbert: SentenceTransformer) -> Dict[str, List[Union[float, int, str]]]:
        sentences, embeddings = encode(text, sbert)
        return self.extract_raw(sentences, embeddings)
    
    def extract_raw(self, sentences: List[str], embeddings: torch.Tensor) ->  Dict[str, List[Union[float, int, str]]]:
        self.eval()
        input = embeddings.unsqueeze(0)
        output = self.forward(input)
        output = output.squeeze(0).cpu().detach().numpy()
        ranked_sentences = sorted(((output[i], i,s) for i,s in enumerate(sentences)), reverse=True)
        ranked_sentences = list(zip(*ranked_sentences))
        ranked_sentences = {
            'ranks': list(ranked_sentences[0]),
            'positions': list(ranked_sentences[1]),
            'sentences': list(ranked_sentences[2])
        }
        return ranked_sentences

    def extract_text(self, text: str, sbert: SentenceTransformer,  top_k: int) -> str:
        sentences, embeddings = encode(text, sbert)
        ranked_sentences = self.extract_raw(sentences, embeddings)
        ranked_sentences = select_top_k(ranked_sentences, top_k)
        return generate_text(ranked_sentences)
    
    @classmethod
    def load(cls, path: str) -> LSTMExtractor:
        dic = torch.load(path)
        config = dic['config']
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        if 'use_batch_norm' in config:
            batch_norm = config['use_batch_norm']
        else:
            batch_norm = False
        if 'use_layer_norm' in config:
            layer_norm = config['use_layer_norm']
        else:
            layer_norm = False
        if 'dropout' in config:
            dropout = config['dropout']
        else:
            dropout = 0.0
        if 'activation' in config:
            activation_fct = config['activation']
        else:
            activation_fct = 'relu'

        model = cls(None, input_size, hidden_size, batch_norm, layer_norm, dropout, activation_fct)
        model.load_state_dict(dic['model_state_dict'])
        return model