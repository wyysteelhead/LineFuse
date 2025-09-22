import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, 
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 mixed_precision: bool = True):
        
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (clean_images, blur_images) in enumerate(progress_bar):
            clean_images = clean_images.to(self.device)
            blur_images = blur_images.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    if hasattr(self.model, 'forward'):
                        outputs = self.model(clean_images, blur_images)
                        if isinstance(outputs, tuple):
                            noise_pred, noise_target = outputs
                            loss = self.loss_fn(noise_pred, noise_target)
                        else:
                            loss = self.loss_fn(outputs, clean_images)
                    else:
                        outputs = self.model(blur_images)
                        loss = self.loss_fn(outputs, clean_images)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if hasattr(self.model, 'forward'):
                    outputs = self.model(clean_images, blur_images)
                    if isinstance(outputs, tuple):
                        noise_pred, noise_target = outputs
                        loss = self.loss_fn(noise_pred, noise_target)
                    else:
                        loss = self.loss_fn(outputs, clean_images)
                else:
                    outputs = self.model(blur_images)
                    loss = self.loss_fn(outputs, clean_images)
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for clean_images, blur_images in tqdm(val_loader, desc="Validation"):
                clean_images = clean_images.to(self.device)
                blur_images = blur_images.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        if hasattr(self.model, 'inference'):
                            outputs = self.model.inference(blur_images)
                            loss = self.loss_fn(outputs, clean_images)
                        else:
                            outputs = self.model(blur_images)
                            loss = self.loss_fn(outputs, clean_images)
                else:
                    if hasattr(self.model, 'inference'):
                        outputs = self.model.inference(blur_images)
                        loss = self.loss_fn(outputs, clean_images)
                    else:
                        outputs = self.model(blur_images)
                        loss = self.loss_fn(outputs, clean_images)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              save_dir: Path,
              save_every: int = 10) -> Dict[str, Any]:
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            if self.scheduler:
                self.scheduler.step()
            
            logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_dir / 'best_model.pth', epoch, val_loss)
                logging.info(f"New best model saved with val loss: {val_loss:.4f}")
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir / f'epoch_{epoch+1}.pth', epoch, val_loss)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def save_checkpoint(self, filepath: Path, epoch: int, val_loss: float) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logging.info(f"Checkpoint loaded: {filepath}")
        
        return {
            'epoch': checkpoint['epoch'],
            'val_loss': checkpoint['val_loss']
        }