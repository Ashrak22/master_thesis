import argparse
import os

import datasets
from transformers import AutoTokenizer
import torch

from masterthesis.trainers.summarization_trainer import Trainer

config = {
    'max_length': 1024,
    'retokenize': False,
    'dataset': 'pubmed',
    'base_model': 'sshleifer/distilbart-cnn-12-6',
    'dataset_prefix': None,
    'local_attention': False,
    'attention_length': 1024,
    'metric': 'rouge',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 1,
    'max_steps': 500000,
    'val_steps': 10000,
    'log_steps': 500,
    'early_stopping': 3,
    'generate_max_length': 500,
    'optimizer_settings': {
        'name': 'AdamW',
        'lr': 1e-5
    },
    'scheduler_settings': {
        'scheduler_type': 'linear',
        'warmup_steps': 10000
    },
    'use_amp': False
}

def parse():
    parser = argparse.ArgumentParser(description='Seq2Seq Training pytorch')
    #parser.add_argument('-b', '--batch-size', default=1, type=int,
    #                    metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--print-steps', type=int, default=1000)
    parser.add_argument('--prefix', type=str, default=None)
    args = parser.parse_args()
    return args


def generate_summary(text: str, model: torch.nn.Module, tokenizer: AutoTokenizer, print_summaries: bool = False)  -> str:
    tokens = tokenizer([text], truncation=True, max_length=2048, return_tensors='pt')
    tokens['input_ids'] = tokens['input_ids'].to(config['device'])
    tokens['attention_mask'] = tokens['attention_mask'].to(config['device'])
    res = model.generate(tokens['input_ids'], tokens['attention_mask'])
    #summary = res.logits.cpu().softmax(dim=2).argmax(dim=2)[0]
    summary_text = tokenizer.decode(res.cpu())

    if print_summaries:
        print("==== Original Text ====\r\n", text)
        print("==== Summary ====\r\n", summary_text)

    return summary_text


if __name__ == '__main__':
    #datasets.disable_caching()
    args = parse()
    #config['batch_size'] = args.batch_size
    # config['optimization_level'] = args.opt_level
    # config['local_rank'] = args.local_rank
    # config['sync_bn'] = args.sync_bn
    # config['keep_batchnorm_fp32'] = args.keep_batchnorm_fp32
    # config['loss_scale'] = args.loss_scale
    # config['print_steps'] = args.print_steps
    config['dataset_prefix'] = args.prefix

    config['distributed'] = False
    if 'WORLD_SIZE' in os.environ:
        config['distributed'] = int(os.environ['WORLD_SIZE']) > 1

    config['local_rank'] = 0
    config['gpu'] = 0
    config['world_size'] = 1

    if config['distributed']:
        #config['gpu'] = args.local_rank
        torch.cuda.set_device(config['gpu'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        config['world_size'] = torch.distributed.get_world_size()

    trainer = Trainer(config)
    trainer.train()

