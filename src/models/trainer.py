import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error
import cv2

class DeblurDataset(Dataset):
    """Dataset class for loading deblurring image pairs"""

    def __init__(self, clean_dir: str, blur_dir: str, transform=None):
        self.clean_dir = Path(clean_dir)
        self.blur_dir = Path(blur_dir)
        self.transform = transform

        # Get all clean images
        self.clean_files = sorted(list(self.clean_dir.glob("*.png")))

        # Match clean images with corresponding blur images
        self.image_pairs = []
        for clean_file in self.clean_files:
            # Find all blur images that correspond to this clean image
            blur_pattern = f"{clean_file.stem}_*.png"
            blur_files = list(self.blur_dir.glob(blur_pattern))

            # Add all pairs for this clean image
            for blur_file in blur_files:
                self.image_pairs.append((clean_file, blur_file))

        print(f"Found {len(self.image_pairs)} image pairs")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        clean_path, blur_path = self.image_pairs[idx]

        # Load images
        clean_image = Image.open(clean_path).convert('RGB')
        blur_image = Image.open(blur_path).convert('RGB')

        if self.transform:
            clean_image = self.transform(clean_image)
            blur_image = self.transform(blur_image)

        return clean_image, blur_image


def get_default_transforms(image_size: int = 512):
    """Get default image transforms for training and validation"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = nn.MSELoss()(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index"""
    from skimage.metrics import structural_similarity as ssim

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2

    return ssim(img1_gray, img2_gray, data_range=1.0)


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
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (clean_images, blur_images) in enumerate(progress_bar):
            clean_images = clean_images.to(self.device)
            blur_images = blur_images.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    # 检查是否是扩散模型
                    if hasattr(self.model, 'scheduler') and hasattr(self.model, 'unet'):
                        # 扩散模型：返回(predicted_noise, target_noise)
                        noise_pred, noise_target = self.model(clean_images, blur_images)
                        loss = self.loss_fn(noise_pred, noise_target)
                    else:
                        # 普通模型：返回预测图像
                        outputs = self.model(blur_images)
                        loss = self.loss_fn(outputs, clean_images)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 检查是否是扩散模型
                if hasattr(self.model, 'scheduler') and hasattr(self.model, 'unet'):
                    # 扩散模型：返回(predicted_noise, target_noise)
                    noise_pred, noise_target = self.model(clean_images, blur_images)
                    loss = self.loss_fn(noise_pred, noise_target)
                else:
                    # 普通模型：返回预测图像
                    outputs = self.model(blur_images)
                    loss = self.loss_fn(outputs, clean_images)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Calculate PSNR for this batch
            with torch.no_grad():
                # 检查是否是扩散模型
                if hasattr(self.model, 'scheduler') and hasattr(self.model, 'unet'):
                    # 扩散模型训练时跳过PSNR计算（输出是噪声，不是图像）
                    batch_psnr = 0.0  # 占位符，实际验证时会计算真正的PSNR
                else:
                    # 普通模型：计算预测图像的PSNR
                    # Convert from [-1,1] to [0,1] for PSNR calculation
                    pred_01 = (outputs + 1) / 2
                    clean_01 = (clean_images + 1) / 2
                    batch_psnr = calculate_psnr(pred_01, clean_01)

            total_loss += loss.item()
            total_psnr += batch_psnr

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PSNR': f'{batch_psnr:.2f}dB'
            })

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        self.train_losses.append(avg_loss)

        return {
            'loss': avg_loss,
            'psnr': avg_psnr
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(val_loader)

        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for clean_images, blur_images in progress_bar:
                clean_images = clean_images.to(self.device)
                blur_images = blur_images.to(self.device)

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        # 检查是否是扩散模型
                        if hasattr(self.model, 'scheduler') and hasattr(self.model, 'unet'):
                            # 扩散模型：使用推理模式生成清晰图像
                            outputs = self.model.inference(blur_images, num_inference_steps=20)
                            # 计算噪声损失用于验证损失记录
                            noise_pred, noise_target = self.model(clean_images, blur_images)
                            loss = self.loss_fn(noise_pred, noise_target)
                        else:
                            # 普通模型：直接预测图像
                            outputs = self.model(blur_images)
                            loss = self.loss_fn(outputs, clean_images)
                else:
                    # 检查是否是扩散模型
                    if hasattr(self.model, 'scheduler') and hasattr(self.model, 'unet'):
                        # 扩散模型：使用推理模式生成清晰图像
                        outputs = self.model.inference(blur_images, num_inference_steps=20)
                        # 计算噪声损失用于验证损失记录
                        noise_pred, noise_target = self.model(clean_images, blur_images)
                        loss = self.loss_fn(noise_pred, noise_target)
                    else:
                        # 普通模型：直接预测图像
                        outputs = self.model(blur_images)
                        loss = self.loss_fn(outputs, clean_images)

                # Convert from [-1,1] to [0,1] for metrics calculation
                pred_01 = (outputs + 1) / 2
                clean_01 = (clean_images + 1) / 2

                # Calculate PSNR
                batch_psnr = calculate_psnr(pred_01, clean_01)

                # Calculate SSIM for first image in batch (to save time)
                pred_np = pred_01[0].cpu().numpy().transpose(1, 2, 0)
                clean_np = clean_01[0].cpu().numpy().transpose(1, 2, 0)
                batch_ssim = calculate_ssim(pred_np, clean_np)

                total_loss += loss.item()
                total_psnr += batch_psnr
                total_ssim += batch_ssim

                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{batch_psnr:.2f}dB',
                    'SSIM': f'{batch_ssim:.3f}'
                })

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        self.val_losses.append(avg_loss)

        return {
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              save_dir: Path,
              save_every: int = 10) -> Dict[str, Any]:
        """Complete training loop with validation and checkpointing"""

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        best_val_psnr = 0.0
        history = {
            'train_loss': [], 'train_psnr': [],
            'val_loss': [], 'val_psnr': [], 'val_ssim': []
        }

        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Model: {self.model.__class__.__name__}")
        logging.info(f"Device: {self.device}")

        for epoch in range(num_epochs):
            epoch_start_time = torch.cuda.Event(enable_timing=True)
            epoch_end_time = torch.cuda.Event(enable_timing=True)
            epoch_start_time.record()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Training
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_psnr'].append(train_metrics['psnr'])

            # Validation
            val_metrics = self.validate_epoch(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_psnr'].append(val_metrics['psnr'])
            history['val_ssim'].append(val_metrics['ssim'])

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            epoch_end_time.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0

            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train PSNR: {train_metrics['psnr']:.2f}dB")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val PSNR: {val_metrics['psnr']:.2f}dB, Val SSIM: {val_metrics['ssim']:.3f}")
            print(f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

            # Save best model based on validation PSNR
            if val_metrics['psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['psnr']
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(
                    save_dir / 'best_model.pth',
                    epoch, val_metrics, is_best=True
                )
                print(f"✓ New best model saved! PSNR: {val_metrics['psnr']:.2f}dB")

            # Regular checkpointing
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    save_dir / f'epoch_{epoch + 1}.pth',
                    epoch, val_metrics, is_best=False
                )

        final_results = {
            'history': history,
            'best_val_loss': best_val_loss,
            'best_val_psnr': best_val_psnr,
            'total_epochs': num_epochs
        }

        logging.info(f"Training completed!")
        logging.info(f"Best validation PSNR: {best_val_psnr:.2f}dB")

        return final_results
    
    def save_checkpoint(self, filepath: Path, epoch: int,
                       val_metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint with complete training state"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'is_best': is_best,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """Load model checkpoint and restore training state"""

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        logging.info(f"Checkpoint loaded: {filepath}")
        print(f"Resumed from epoch {checkpoint['epoch']} with val PSNR: {checkpoint['val_metrics']['psnr']:.2f}dB")

        return {
            'epoch': checkpoint['epoch'],
            'val_metrics': checkpoint['val_metrics'],
            'is_best': checkpoint.get('is_best', False)
        }


def create_loss_function(loss_type: str = 'combined') -> nn.Module:
    """Create loss function for deblurring"""

    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'l2':
        return nn.MSELoss()
    elif loss_type == 'combined':
        # Combine L1 and L2 losses
        class CombinedLoss(nn.Module):
            def __init__(self, l1_weight=0.7, l2_weight=0.3):
                super().__init__()
                self.l1_weight = l1_weight
                self.l2_weight = l2_weight
                self.l1_loss = nn.L1Loss()
                self.l2_loss = nn.MSELoss()

            def forward(self, pred, target):
                l1 = self.l1_loss(pred, target)
                l2 = self.l2_loss(pred, target)
                return self.l1_weight * l1 + self.l2_weight * l2

        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_optimizer(model: nn.Module, opt_type: str = 'adamw',
                    lr: float = 1e-4, weight_decay: float = 1e-2) -> optim.Optimizer:
    """Create optimizer for training"""

    if opt_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def create_scheduler(optimizer: optim.Optimizer, scheduler_type: str = 'cosine',
                    num_epochs: int = 100, min_lr: float = 1e-6) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""

    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                   patience=10, min_lr=min_lr)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")