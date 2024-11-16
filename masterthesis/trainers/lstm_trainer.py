from datetime import datetime
import os
from typing import Any, Dict, Tuple

import psutil
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import DatasetDict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import get_scheduler
from colorama import Fore, Style

from models import encode, LSTMExtractor
from logger import Logger, LoggerKeys, Colors, Mapping, DurationFormats
from data.helpers import get_dataset

class LSTMTrainer():
    def __init__(self, config:Dict[str, Any]) -> None:
        self.config = config

        #Sbert settings
        self.sbert_model_name = config['sbert_model']
        self.stransformer = SentenceTransformer(self.sbert_model_name)

        # Model settings
        self.distance_metric = config['distance_metric']
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.top_k = config['extract_top_k']
        self.val_steps = config['val_steps']
        self.metric = config['metric']
        self.use_batch_norm = config['use_batch_norm']
        self.use_layer_norm = config['use_layer_norm']
        self.activation_function = config['activation']
        self.dropout = config['dropout']
        self.best_r1 = 0

        # Logging settings
        self.writer = SummaryWriter()
        self.field_mapping = [
            Mapping(LoggerKeys.DURATION, 'Tick Duration', '', DurationFormats.MinutesSeconds, 7, Colors.LIGHTMAGENTA_EX),
            Mapping(LoggerKeys.AVG_DURATION, 'Avg. Duration', '', DurationFormats.MinutesSeconds, 6, Colors.LIGHTGREEN_EX),
            Mapping(LoggerKeys.TOTAL_DURATION, 'Total Duration', '', DurationFormats.DaysHoursMinutes, 8, Colors.LIGHTYELLOW_EX),
            Mapping('gpu', 'GPU', '', length=2),
            Mapping('lr', 'Learning Rate', 'training/Learning Rate', length=22),
            Mapping('loss', 'Loss', 'training/Loss', length=22),
            Mapping('ram', 'RAM Usage (GB)', 'performance/ram_usage', format='{0:.2f}', length=5)  
        ]
        self.eval_mapping = [
            Mapping(None, '', '', 'Eval at step:'),
            Mapping(LoggerKeys.PREVIOUS_STEP, '', '', color=Colors.LIGHTRED_EX),
            Mapping(None, '', '', 'R1:'),
            Mapping('r1', 'R1', 'eval/Rouge-1 F1-Score', color=Colors.LIGHTGREEN_EX, length=7, use_current_step=False),
            Mapping(None, '', '', 'R2:'),
            Mapping('r2', 'R2', 'eval/Rouge-2 F1-Score', color=Colors.LIGHTYELLOW_EX, length=7, use_current_step=False),
            Mapping(None, '', '', 'RL:'),
            Mapping('rl', 'RL', 'eval/Rouge-L F1-Score', color=Colors.LIGHTBLUE_EX, length=7, use_current_step=False),
            Mapping(None, '', '', 'RLSum:'),
            Mapping('rls', 'RLS', 'eval/Rouge-Lsum F1-Score', color=Colors.LIGHTBLUE_EX, length=7, use_current_step=False),
        ]
        self.log_steps = config['log_steps']
        self.logger = Logger(self.writer, self.field_mapping, header_frequency=int(self.val_steps/self.log_steps), avg_window=10)

        # Training settings
        self.dataset_name = config['dataset']
        self.dataset_prefix = config['dataset_prefix']
        self.batch_size = config['batch_size']
        self.max_steps = config['max_steps']
        self.early_stopping_value = config['early_stopping']
        self.evals_since_best = 0

        # Optimizer settings
        self.lr = config['optimizer_settings']['lr']
        self.scheduler_type = config['scheduler_settings']['scheduler_type']
        self.warmup_steps = config['scheduler_settings']['warmup_steps']

        #Distributed settings
        self.local_rank = config['local_rank']
        self.distributed = config['distributed']
        self.world_size = config['world_size']

        self.create_model()

        if config['weights'] is not None:
            self.weight_path = config['weights']
            self.load()

    def create_model(self):
        self.model = LSTMExtractor(self.distance_metric, self.input_size, self.hidden_size, self.use_batch_norm, self.use_layer_norm, self.dropout, self.activation_function)

    def save(self):
        directory = self.writer.get_logdir()
        dic = {
                "config": self.config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            }
        torch.save(
            dic, os.path.join(directory, "summarizer.h5")
        ) 

    def load(self):
        dic = torch.load(self.weight_path)
        self.model = LSTMExtractor.load(self.weight_path)
        self.optimizer.load_state_dict(dic['optimizer_state_dic'])
        self.lr_scheduler.load_state_dict(dic['lr_scheduler_state_dict'])

    def get_loaders(self, ds: DatasetDict) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(ds['train'], 
                                  shuffle=True, 
                                  batch_size=1, 
                                  pin_memory=True)

        val_loader = DataLoader(ds['val'].select(range(500)), 
                                batch_size=1)

        return (train_loader, val_loader)
    
    def eval(self, ds: DatasetDict, i: int) -> None:
        # When distributed evaluate only on rank 0
        if self.local_rank not in [-1, 0]:
            dist.barrier()
            return
        
        model = self.model
        # if self.distributed:
        #     model = self.model.module

        data = iter(ds['val'].select(range(500)))
        rouge = evaluate.load('rouge')
        summaries = []
        abstracts = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data, "Evaluating: "):
                res = model.extract_text(batch['text'], self.stransformer, self.top_k)             
                summaries.append(' .\n'.join(res.split('.')))
                abstracts.append(' .\n'.join(batch['abstract'].split('.')))
                
            self.writer.add_text('Summary[0]', summaries[0], i)
            results = rouge.compute(predictions=summaries, references=abstracts, use_stemmer=True)
            self.logger.add_value('r1', results['rouge1'])
            self.logger.add_value('r2', results['rouge2'])
            self.logger.add_value('rl', results['rougeL'])
            self.logger.add_value('rls', results['rougeLsum'])
            self.logger.print_line(self.eval_mapping)
            
            if self.best_r1 < results['rouge1']:
                self.best_r1 = results['rouge1']
                self.save()
                self.evals_since_best = 0
            else:
               self.evals_since_best += 1 
        
        if self.local_rank == 0 and self.distributed:
            dist.barrier()

    def train(self):
         # Only first should prepare 
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
        
        self.model = self.model.cuda()

        if self.local_rank not in [-1, 0]:
            dist.barrier()
            ds = get_dataset(self.dataset_name, None, -1, False, self.dataset_prefix, None, 'lstm')
        else:
            ds = get_dataset(self.dataset_name, None, -1, False, self.dataset_prefix, {'model_name': self.sbert_model_name}, 'lstm')
            ds_eval = get_dataset(self.dataset_name, None, -1, False, self.dataset_prefix, None, 'lstm')

        if self.local_rank == 0 and self.distributed:
            dist.barrier()
        
        ds.set_format('torch', columns=['embeddings', 'target'])
        dl_train, dl_eval = self.get_loaders(ds)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.lr_scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )

        i = 0
        self.start_timestamp = datetime.now()
        print(f"Pre-trainig eval {Fore.YELLOW}(baseline){Style.RESET_ALL}")
        self.loss_fct = torch.nn.MSELoss()
        #self.eval(ds_eval, i)
        epoch = 0
        iterable = iter(dl_train)
        while True:
            torch.cuda.empty_cache()
            self.model.train()
            info = 0
            for j in range(self.val_steps):
                try:
                    try:
                        batch = next(iterable)
                    except StopIteration:
                        epoch += 1
                        if self.distributed:
                            self.train_sampler.set_epoch(epoch)
                        iterable = iter(dl_train)
                        batch = next(iterable)
                    i += 1
                    # info = batch['id']
                    # batch.pop('id')
                    #batch = {k: v.cuda() for k, v in batch.items()}
                    input = torch.stack(batch['embeddings'], dim=1)
                    outputs = self.model(input.cuda())
                    loss = self.loss_fct(outputs, torch.Tensor(batch['target']).cuda())
                    loss.backward()
                    
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if (j+1) % self.log_steps == 0:
                        self.logger.add_value('gpu', self.local_rank)
                        self.logger.add_value('lr', self.lr_scheduler.get_last_lr()[-1])
                        self.logger.add_value('loss', loss)
                        self.logger.add_value('ram', psutil.virtual_memory().used / (1024*1024*1024))
                        self.logger.step(i)
                except RuntimeError:
                    print(info)
                    raise RuntimeError
            
            self.eval(ds_eval, i)

            if i >= self.max_steps or self.evals_since_best == self.early_stopping_value:
                break