import argparse
import os

import datasets
from transformers import AutoTokenizer
import torch

from masterthesis.trainers.lstm_trainer import LSTMTrainer

config = {
    'sbert_model': "all-mpnet-base-v2", 
    'distance_metric': "cosine_similarity",
    'input_size': 768,
    'hidden_size': 512,
    'extract_top_k': 8,
    'use_batch_norm': True,
    'use_layer_norm': False,
    'dropout': 0.2,
    'activation': 'relu',
    # -------
    'log_steps': 500,
    'dataset': 'pubmed',
    'val_steps': 2000,
    'metric': 'rouge',
    'batch_size': 1,
    'max_steps': 500000,
    'early_stopping': 20,
    'dataset_prefix': None,
    'local_attention': True,
    'attention_length': 1024,

    'optimizer_settings': {
        'name': 'Adam',
        'lr': 1e-5
    },
    'scheduler_settings': {
        'scheduler_type': 'linear',
        'warmup_steps': 8000
    },
    'use_amp': False,
}

def parse():
    parser = argparse.ArgumentParser(description='Seq2Seq Training pytorch')
    parser.add_argument('-b', '--batch-size', default=config['batch_size'], type=int,
                        metavar='N', help='mini-batch size per process (default: from config)')
    parser.add_argument('-d', '--deterministic', action='store_true')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--print-steps', type=int, default=1000)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #datasets.disable_caching()
    args = parse()
    config['batch_size'] = args.batch_size
    config['optimization_level'] = args.opt_level
    config['local_rank'] = args.local_rank
    config['sync_bn'] = args.sync_bn
    config['keep_batchnorm_fp32'] = args.keep_batchnorm_fp32
    config['loss_scale'] = args.loss_scale
    config['print_steps'] = args.print_steps
    config['dataset_prefix'] = args.prefix
    config['weights'] = args.weights

    config['distributed'] = False
    if 'WORLD_SIZE' in os.environ:
        config['distributed'] = int(os.environ['WORLD_SIZE']) > 1

    config['local_rank'] = 0
    config['gpu'] = 0
    config['world_size'] = 1

    if config['distributed']:
        config['gpu'] = args.local_rank
        torch.cuda.set_device(config['gpu'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        config['world_size'] = torch.distributed.get_world_size()

    trainer = LSTMTrainer(config)
    trainer.train()

