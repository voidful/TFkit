"""Training utilities for TFKit."""

import time
from datetime import timedelta
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

from tfkit.utility.constants import (
    WARMUP_RATIO, 
    MONITORING_STEP_INTERVAL, 
    CACHE_STEP_INTERVAL
)
from tfkit.utility.logger import Logger
from tfkit.utility.model import save_model


class TrainingManager:
    """Manages the training process for TFKit models.
    
    Provides functionality for:
    - Model and optimizer preparation
    - Training loop management
    - Evaluation coordination
    - Progress monitoring and logging
    """
    
    def __init__(self, accelerator: Accelerator, logger: Logger) -> None:
        """Initialize the training manager.
        
        Args:
            accelerator: Accelerator instance for distributed training
            logger: Logger instance for tracking progress
        """
        self.accelerator = accelerator
        self.logger = logger
        
    def create_optimizer(self, model: torch.nn.Module, lr: float, 
                        total_steps: int) -> Tuple[Optimizer, LambdaLR]:
        """Create optimizer and scheduler for training.
        
        Args:
            model: The model to optimize
            lr: Learning rate
            total_steps: Total number of training steps
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * WARMUP_RATIO),
            num_training_steps=total_steps
        )
        return optimizer, scheduler
        
    def prepare_models_and_optimizers(self, models_list, dataloaders, input_arg):
        """Prepare models and optimizers for training."""
        optims_schs = []
        models = []
        data_iters = []
        
        total_iter_length = len(dataloaders[0])
        
        for i, (model, dataloader) in enumerate(zip(models_list, dataloaders)):
            # Prepare model
            if not self.accelerator.state.backend:
                model = torch.nn.DataParallel(model)
            model.train()
            
            # Create optimizer
            lr = (input_arg.get('lr')[i] if i < len(input_arg.get('lr')) 
                  else input_arg.get('lr')[0])
            optimizer, scheduler = self.create_optimizer(model, lr, total_iter_length)
            
            # Prepare with accelerator
            model, (optimizer, scheduler), dataloader = self.accelerator.prepare(
                model, (optimizer, scheduler), dataloader
            )
            
            optims_schs.append((optimizer, scheduler))
            models.append(model)
            data_iters.append(iter(dataloader))
            
        return models, optims_schs, data_iters, total_iter_length
        
    def train_epoch(self, models, optims_schs, data_iters, models_tag, 
                   input_arg, epoch, fname, add_tokens, total_iter_length):
        """Train models for one epoch."""
        total_iter = 0
        t_loss = 0
        end = False
        
        pbar = tqdm(total=total_iter_length)
        
        while not end:
            for (model, optim_sch, mtag, train_batch) in zip(
                models, optims_schs, models_tag, data_iters
            ):
                optimizer, scheduler = optim_sch
                train_batch = next(train_batch, None)
                
                if train_batch is not None:
                    loss = self._process_batch(
                        model, optimizer, scheduler, train_batch, 
                        input_arg, total_iter, epoch, mtag
                    )
                    t_loss += loss
                    
                    # Monitoring
                    if total_iter % MONITORING_STEP_INTERVAL == 0 and total_iter != 0:
                        self._log_progress(epoch, mtag, model, total_iter, 
                                         t_loss, total_iter_length)
                        
                    # Caching
                    if total_iter % CACHE_STEP_INTERVAL == 0 and total_iter != 0:
                        save_model(
                            models, input_arg, models_tag, epoch,
                            f"{fname}_epoch_{epoch}_iter_{total_iter}", 
                            self.logger, add_tokens=add_tokens,
                            accelerator=self.accelerator
                        )
                else:
                    end = True
                    
            pbar.update(1)
            total_iter += 1
            
        pbar.close()
        
        # Final logging
        avg_loss = t_loss / total_iter if total_iter > 0 else 0
        self.logger.write_log(
            f"epoch: {epoch}, step: {total_iter}, loss: {avg_loss}, total: {total_iter}"
        )
        
        return avg_loss
        
    def _process_batch(self, model, optimizer, scheduler, train_batch, 
                      input_arg, total_iter, epoch, mtag):
        """Process a single training batch."""
        loss = model(train_batch)
        loss = loss / input_arg.get('grad_accum')
        
        self.accelerator.backward(loss.mean())
        
        if (total_iter + 1) % input_arg.get('grad_accum') == 0:
            optimizer.step()
            model.zero_grad()
            scheduler.step()
            
        loss_value = loss.mean().detach()
        self.logger.write_metric("loss/step", loss_value, epoch)
        
        return loss_value
        
    def _log_progress(self, epoch, mtag, model, total_iter, t_loss, total_iter_length):
        """Log training progress."""
        avg_loss = t_loss / total_iter if total_iter > 0 else 0
        self.logger.write_log(
            f"epoch: {epoch}, tag: {mtag}, task: {model.__class__.__name__}, "
            f"step: {total_iter}, loss: {avg_loss}, total: {total_iter_length}"
        )
        
    def evaluate_models(self, models, dataloaders, fname, input_arg, epoch):
        """Evaluate models on test data."""
        t_loss = 0
        t_length = 0
        
        for model in models:
            model.eval()
            
        with torch.no_grad():
            total_iter_length = len(dataloaders[0])
            iters = [iter(self.accelerator.prepare(ds)) for ds in dataloaders]
            end = False
            pbar = tqdm(total=total_iter_length)
            
            while not end:
                for model, batch in zip(models, iters):
                    test_batch = next(batch, None)
                    if test_batch is not None:
                        loss = model(test_batch)
                        loss = loss / input_arg.get('grad_accum')
                        t_loss += loss.mean().detach()
                        t_length += 1
                        pbar.update(1)
                    else:
                        end = True
                        
            pbar.close()
            
        avg_t_loss = t_loss / t_length if t_length > 0 else 0
        self.logger.write_log(f"task: {fname}, Total Loss: {avg_t_loss}")
        self.logger.write_metric("eval_loss/step", avg_t_loss, epoch)
        
        return avg_t_loss 