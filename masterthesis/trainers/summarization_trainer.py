from datetime import datetime
import os
import re
from typing import Any, Dict, Tuple

from datasets import DatasetDict
import evaluate
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from colorama import Fore, Style

from data import get_dataset
from models import DistillBartSummarizer
from logger import Logger, LoggerKeys, Colors, Mapping, DurationFormats


class Trainer():
    def __init__(self, config:Dict[str, Any]):
        print(f'{Fore.LIGHTBLUE_EX}Initializing trainer{Style.RESET_ALL}')
        self.config = config

        # Eval settings
        self.val_steps = config['val_steps']
        self.retokenize = config['retokenize']
        self.metric = config['metric']
        self.best_r1 = 0

        # Logging       
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

        # Model settings
        self.max_length = config['max_length']
        self.checkpoint = config['base_model']
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.attention_window = config['attention_length']
        self.model = DistillBartSummarizer(self.checkpoint, self.max_length, config['generate_max_length'], 
                                        config['local_attention'], config['attention_length'])

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

        #AMP Settings
        self.use_amp = config['use_amp']
        self.local_rank = config['local_rank']
        if self.use_amp:
            #only import apex if needed
            from apex import amp
            
            
            self.optimization_level = config['optimization_level']      
            self.keep_batchnorm_fp32 = config['keep_batchnorm_fp32']
            self.loss_scale = config['loss_scale']
        self.distributed = config['distributed']
        if self.distributed:
            from apex.parallel import DistributedDataParallel, convert_syncbn_model
            
            self.sync_bn = config['sync_bn']
            
        self.world_size = config['world_size']
        self.train_sampler = None

        if  config['weights'] is not None:
            self.weight_path = config['weights']
            self.load()
        
        if 'text_rank_settings' in config:
            self.text_rank_config = config['text_rank_settings']
        else:
            self.text_rank_config = None
        
        if 'lstm_settings' in config:
            self.lstm_config = config['lstm_settings']
        else:
            self.lstm_config = None

    
    def get_loaders(self, ds: DatasetDict) -> Tuple[DataLoader, DataLoader]:
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model.inner_bart, return_tensors="pt", pad_to_multiple_of=self.attention_window)

        if self.distributed:
            self.train_sampler = DistributedSampler(ds['train'])

        train_loader = DataLoader(ds['train'], 
                                  shuffle=(self.train_sampler is None), 
                                  batch_size=self.batch_size, 
                                  pin_memory=True, 
                                  sampler=self.train_sampler,
                                  collate_fn=data_collator)

        val_loader = DataLoader(ds['val'].select(range(500)), 
                                batch_size=1, 
                                collate_fn=data_collator)

        return (train_loader, val_loader)

    def eval(self, ds: DataLoader, i: int) -> None:
        # When distributed evaluate only on rank 0
        if self.local_rank not in [-1, 0]:
            dist.barrier()
            return
        
        model = self.model
        if self.distributed:
            model = self.model.module

        data = iter(ds)
        rouge = evaluate.load('rouge')
        summaries = []
        abstracts = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data, "Evaluating: "):
                input_ids = batch['input_ids'].cuda()
                att_masks = batch['attention_mask'].cuda()
                res = model.inner_bart.generate(inputs=input_ids, attention_mask=att_masks).cpu()
                labels = batch["labels"].numpy()
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)                
                summaries.extend([' .\n'.join(x.split('.')) for x in self.tokenizer.batch_decode(res, skip_special_tokens=True)])
                abstracts.extend([' .\n'.join(x.split('.')) for x in self.tokenizer.batch_decode(labels, skip_special_tokens=True)])
                
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

    def save(self):
        directory = self.writer.get_logdir()
        dic = {
                "config": self.config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            }
        if self.use_amp:
            dic["amp_state_dict"] = amp.state_dict()
        torch.save(
            dic, os.path.join(directory, "summarizer.h5")
        )        

    def load(self):
        dic = torch.load(self.weight_path)
        self.model.load_state_dict(dic['model_state_dict'])
        self.optimizer.load_state_dict(dic['optimizer_state_dic'])
        self.lr_scheduler.load_state_dict(dic['lr_scheduler_state_dict'])

        if self.use_amp and 'amp_state_dict' in dic:
            amp.load_state_dict(dic['amp_state_dict'])

    def train(self):
        # Only first should prepare 
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
        
        self.model = self.model.cuda()

        if self.local_rank not in [-1, 0]:
            dist.barrier()
            ds = get_dataset(self.dataset_name, self.checkpoint, self.max_length, False, self.dataset_prefix, self.text_rank_config, self.lstm_config)
        else:
            ds = get_dataset(self.dataset_name, self.checkpoint, self.max_length, self.retokenize, self.dataset_prefix, self.text_rank_config, self.lstm_config)

        if self.local_rank == 0 and self.distributed:
            dist.barrier()
        
        ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        dl_train, dl_eval = self.get_loaders(ds)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        

        if self.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, 
                                                        self.optimizer, 
                                                        opt_level=self.optimization_level, 
                                                        keep_batchnorm_fp32=self.keep_batchnorm_fp32)

        if self.distributed:
            if self.sync_bn:
                self.model = convert_syncbn_model(self.model)
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)

        self.lr_scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )
        i = 0
        self.start_timestamp = datetime.now()
        print(f"Pre-trainig eval {Fore.YELLOW}(baseline){Style.RESET_ALL}")
        self.eval(dl_eval, i)
        epoch = 0
        if self.distributed:
                self.train_sampler.set_epoch(epoch)
        self.model.train()
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
                    batch = {k: v.cuda() for k, v in batch.items()}
                    outputs = self.model(**batch, return_dict=True)

                    if self.use_amp:
                        with amp.scale_loss(outputs.loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        outputs.loss.backward()
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if (j+1) % self.log_steps == 0:
                        self.logger.add_value('gpu', self.local_rank)
                        self.logger.add_value('lr', self.lr_scheduler.get_last_lr()[-1])
                        self.logger.add_value('loss', outputs.loss)
                        self.logger.add_value('ram', psutil.virtual_memory().used / (1024*1024*1024))
                        self.logger.step(i)
                except RuntimeError:
                    print(info)
                    raise RuntimeError
            
            self.eval(dl_eval, i)

            if i >= self.max_steps or self.evals_since_best == self.early_stopping_value:
                break

