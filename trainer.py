
import os
import sys
import logging
import numpy as np
from pathlib import Path    
from typing import Union
from time import perf_counter
from datetime import datetime
from dataclasses import dataclass

import fire
import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report as clf_rpt

from model.vit import ViTConfig, ViTModel
from data.celeb_utils import get_dataset, get_weights_for_loss


@dataclass
class TrainingArgs:
    dataset_path: Path = None
    device: torch.device = torch.device('cuda:0')
    dtype: torch.dtype = torch.float32
    chkpt_base_dir: Path = Path('runs/')
    
    train_batch_sz: int = 384
    test_batch_sz: int = 3072

    beta1: float = 0.9
    beta2: float = 0.95
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    
    max_iters: int = 4001
    scale_class_weights: bool = True

    model_cfg: ViTConfig = ViTConfig()
    clf_thresh: float = 0.5

    curr_iter: int = 1
    log_interval: int = 10
    eval_iterval: int = 150
    plot_interval: int = 50
    best_model_stat: float = None
    worse_report_counter: int = 0

    save_plots: bool = True


class Trainer:
    def __init__(self, args: TrainingArgs = None, 
                 checkpoint_path: Union[Path , str] = None,
                 override_args: dict = {}):
        '''
        Either one of `args` or `checkpoint_path` must be provided.
        
        To initiate a new run, provide `args`.
        To resume from checkpoint, provide `checkpoint_path`.

        If both are provided, will resume from `checkpoint_path` by default.

        To override arguments when resuming from checkpoints, 
            provide a dictionary of arguments in `override_args`.
        '''
        self.chkpt_file_name = "model.th"
        
        if checkpoint_path is not None:
            if self.checkpoint_exists(checkpoint_path):
                self.checkpoint_path = Path(checkpoint_path)
                self.load_checkpoint()
                self.logger = self.create_logger()
                self.logger.info(f"Successfully loaded checkpoint from iter:{self.args.curr_iter}")
                self._override_args(override_args)
            else:
                raise ValueError('Checkpoint not present at provided {checkpoint_path=}')
        
        elif args is not None:
            self.args = args
            self.checkpoint_path = self.create_checkpoint_dir()
            
            self.logger = self.create_logger()
            self.logger.info('checkpoint_path not provided. starting new run.')
            self.logger.info('dumping args...')
            self.logger.info(str(self.args.__dict__))

            self.model = self.init_model()
            self.optimizer = self.get_optimizer()
            self.scheduler = self.get_scheduler()
        else:
            raise ValueError('Expected either args or checkpoint_path')

        self.tr_dataloader = self.get_train_dataloader()
        self.te_dataloader = self.get_test_dataloader()
        self.criterion = self.get_loss_fn()
        self.logger.info("All initializations complete.")

        self.writer = SummaryWriter(self.checkpoint_path)

    def train(self):
        self.logger.info(f"Resuming training from iter={self.args.curr_iter}")
        for itern in range(self.args.curr_iter, self.args.max_iters):
            self.args.curr_iter = itern
            
            images, labels = next(iter(self.tr_dataloader))
            
            self.model.train()
            self.optimizer.zero_grad()
            
            images = images.to(self.args.dtype).to(self.args.device)
            labels = labels.to(self.args.dtype).to(self.args.device)

            logits = self.model(images, labels)
            loss = self.compute_loss(logits, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.write_tensorboard(dict(
                train_loss=loss.detach().item(),
                learning_rate=self.optimizer.param_groups[0]['lr']))

            if (itern % self.args.log_interval) == 0:
                tr_loss = loss.detach().item()
                self.logger.info(f"{itern=} {tr_loss=}")
            
            if (itern % self.args.eval_iterval) == 0:
                self.logger.info(f"Running evaluation at {itern=}...")
                report = self.evaluate()
                self.write_tensorboard(report)
                self.logger.info(str(report))
                
                self.checkpoint_logic(report)

            if (itern % self.args.plot_interval) == 0:
                pe_filename = self.checkpoint_path / "plots" / f"pe_sim_{itern}.png"
                self.model.get_pe_similarity_plot(f"iter:{itern}").savefig(pe_filename)



    def checkpoint_logic(self, current_report):
        '''
        makes decision for:
        - checkpoint model at current iteration
        - skip checkpointing
        - stop training.
        using the current evaluation report.
        '''
        if self.args.best_model_stat == None:
            self.logger.info('Updated best model. best_model_stat empty.')
            self.args.best_model_stat = current_report['eval_loss']
            self.args.worse_report_counter = 0
            self.save_checkpoint()
        elif self.args.best_model_stat > current_report['eval_loss']:
            self.logger.info('Updating best model.')
            self.logger.info('loss improved from {} to {}'.format(
                self.args.best_model_stat, current_report['eval_loss']))
            self.args.best_model_stat = current_report['eval_loss']
            self.args.worse_report_counter = 0
            self.save_checkpoint()
        else:
            self.args.worse_report_counter += 1
            self.logger.info(F"Updated {self.args.worse_report_counter=}")
        
        if self.args.worse_report_counter > 3:
            self.logger.info(F"evaluation worse over multiple successive runs.")
            self.logger.info(F"Abort Training at {self.args.curr_iter=}")
            sys.exit()

    def save_checkpoint(self):
        chkpt = dict(
            args = self.args,
            model = self.model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        filename = self.checkpoint_path / self.chkpt_file_name
        torch.save(chkpt, filename)
        self.logger.info(f"Saved checkpoint at {filename}")

    def load_checkpoint(self, ):
        chkpt = torch.load(self.checkpoint_path / self.chkpt_file_name)
        self.args = chkpt['args']
        self.model = ViTModel(self.args.model_cfg)
        self.model.load_state_dict(chkpt['model'])
        self.model.to(self.args.dtype).to(self.args.device)

        self.optimizer = self.get_optimizer()
        self.optimizer.load_state_dict(chkpt['optimizer'])

        self.scheduler = self.get_scheduler()
        self.scheduler.load_state_dict(chkpt['scheduler'])
        
    def evaluate(self):
        '''
        Runs an evaluation loop over the provided dataloader
        Returns a dictionary of metrics.
        '''
        self.model.eval()
        predictions = []
        targets = []
        losses = []
        tick = perf_counter()
        for images, target in self.te_dataloader:
            images = images.to(self.args.dtype).to(self.args.device)
            target = target.to(self.args.dtype).to(self.args.device)
            with torch.no_grad():
                logits = self.model(images, target)
                te_loss = self.compute_loss(logits, target).detach().cpu().item()
        
            preds = torch.sigmoid(logits.detach())
            predictions.append(preds.cpu().to(torch.float32).numpy())
            targets.append(target.cpu().to(torch.float32).numpy())
            losses.append(te_loss)
        tock = perf_counter() - tick
        
        targets = np.vstack(targets)
        predictions = np.vstack(predictions)
        avg_loss = sum(losses) / len(losses)

        report = clf_rpt(targets.ravel(), 
                         (predictions.ravel() > self.args.clf_thresh).astype(int),
                         output_dict=True)
        
        return dict(precision = round(report['macro avg']['precision'], 3),
                    recall = round(report['macro avg']['recall'], 3),
                    F1 = round(report['macro avg']['f1-score'], 3),
                    accuracy = round(report['accuracy'], 3),
                    elapsed = round(tock, 3),
                    eval_loss = round(avg_loss, 3)
                    )

    def get_scheduler(self):
        return CosineAnnealingLR(self.optimizer, 
                              T_max = self.args.max_iters,
                              eta_min = self.args.min_lr,
                              last_epoch = -1
                              )
    
    def get_optimizer(self):
        return AdamW(self.model.parameters(),
                     lr=self.args.learning_rate,
                     betas=(self.args.beta1, self.args.beta2)
                     )

    def checkpoint_exists(self, path: Union[Path , str]):
        chkpt_file = Path(path) / self.chkpt_file_name
        return os.path.exists(chkpt_file)

    def _override_args(self, override_args: dict):
        # inplace update self.args with entries in override_args/
        self.args.__dict__.update(override_args)
      
    def get_train_dataloader(self):
        tr_dataset = get_dataset(self.args.dataset_path, split='train')
        tr_dataloader = DataLoader(tr_dataset, 
                                   batch_size=self.args.train_batch_sz, 
                                   shuffle=True)
        return tr_dataloader

    def get_test_dataloader(self):
        te_dataset = get_dataset(self.args.dataset_path, split='test')
        te_dataloader = DataLoader(te_dataset,
                                   batch_size=self.args.test_batch_sz, 
                                   shuffle=False)
        return te_dataloader

    def get_loss_fn(self):
        if self.args.scale_class_weights:
            class_weights = get_weights_for_loss(self.tr_dataloader.dataset.attrib_df)
            pos_weight=torch.tensor(class_weights).to(self.args.device)
        else:
            pos_weight=torch.ones([self.cfg.n_class]).to(self.args.device)
        return BCEWithLogitsLoss(pos_weight=pos_weight)

    def init_model(self):
        return ViTModel(self.args.model_cfg).\
            to(self.args.dtype).\
            to(self.args.device)

    def compute_loss(self, logit, target):
        return self.criterion(logit, target)
    
    def create_checkpoint_dir(self):
        # create a dir w/ current date-time inside the base dir.
        chkpt_name = datetime.now().strftime("%y%m%d-%H%M")
        chkpt_name = Path(self.args.chkpt_base_dir) / chkpt_name
        os.makedirs(chkpt_name, exist_ok=True)
        # create a plots dir inside the base dir
        plots_dir = chkpt_name / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        return chkpt_name
    
    def create_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(self.checkpoint_path / 'logs.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger

    def write_tensorboard(self, report: dict):
        for key in report.keys():
            self.writer.add_scalar(key, report[key], self.args.curr_iter)
            
            
def main(    
        dataset_path,
        checkpoint_path = None,
        train = True,
        eval = True,
        device = 'cuda',
        dtype = 'fp32', # fp32, bf16,
        chkpt_base_dir = 'runs/',
        train_batch_sz = 384,
        test_batch_sz = 3072,
        max_iters = 5001,
        log_interval = 10,
        eval_iterval = 150,
        plot_interval = 150
):
    if dtype == 'fp32':
        dtype = torch.float32
    elif dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError('Expected type to be one of `fp32` or `bf16`.')
    
    checkpoint_path = checkpoint_path
    if checkpoint_path is not None:
        override_args=dict(
            dataset_path = dataset_path,
            device = torch.device(device),
            dtype = dtype,
            chkpt_base_dir = Path(chkpt_base_dir),
            train_batch_sz = train_batch_sz,
            test_batch_sz = test_batch_sz,
            max_iters = max_iters,
            log_interval = log_interval,
            eval_iterval = eval_iterval,
            plot_interval = plot_interval
            )
        trainer = Trainer(checkpoint_path=checkpoint_path,
                          override_args=override_args)
    else:
        args = TrainingArgs()
        args.dataset_path = Path(dataset_path)
        args.device = torch.device(device)
        args.dtype = dtype
        args.chkpt_base_dir = Path(chkpt_base_dir)
        args.train_batch_sz = train_batch_sz
        args.test_batch_sz = test_batch_sz
        args.max_iters = max_iters
        args.log_interval = log_interval
        args.eval_iterval = eval_iterval
        args.plot_interval = plot_interval

        trainer = Trainer(args)
    
    if train:
        trainer.train()
    if eval:
        trainer.evaluate()


if __name__ == '__main__':
    fire.Fire(main)