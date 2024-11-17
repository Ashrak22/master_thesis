import argparse
import os

import datasets
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

from masterthesis.models import DistillBartSummarizer, LSTMExtractor, TextRank

def parse():
    parser = argparse.ArgumentParser(description='Seq2Seq Inference')
    parser.add_argument('-l', '--use_lstm', action='store_true',
                        help='Uses lstm as preselector')
    parser.add_argument('--use_text_rank', action='store_true',
                        help='Uses lstm as preselector')
    parser.add_argument('-w', '--weights', type=str, default=None, help="Path to summarizer weights")
    parser.add_argument('--lstm_weights', type=str, default=None, help="Path to lstm weights")
    parser.add_argument('-t', '--text', type=str, default=None, help="Path to the text file to be summarized")
    parser.add_argument('-k', '--top_k', type=int, default=None, help="Number of sentences to be summarized")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #datasets.disable_caching()
    args = parse()
    summarizer = DistillBartSummarizer.load(args.weights).cuda()
    preselector = None
    if args.use_text_rank:
        preselector = TextRank('all-mpnet-base-v2', 'euclidean_distance')
    if args.use_lstm:
        preselector = LSTMExtractor.load(args.lstm_weights).cuda()
    
    with open(args.text, 'r', encoding="utf-8") as file:
        text = file.read()
    
    if preselector is not None:
        sbert = SentenceTransformer('all-mpnet-base-v2')
        text = preselector.extract_text(text, sbert, args.top_k)
    
    tokenizer = AutoTokenizer.from_pretrained(summarizer.checkpoint)
    tokens = tokenizer(text, max_length=1024, truncation=True, padding='max_length', pad_to_multiple_of=1024, return_tensors='pt')
    summary = summarizer.generate(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
    print(tokenizer.batch_decode(summary, skip_special_tokens=True))

