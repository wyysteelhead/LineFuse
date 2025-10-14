import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, TYPE_CHECKING, Union
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error
import cv2
import time
import json
import hashlib
from contextlib import nullcontext

if TYPE_CHECKING:
    from .lora_adapter import HybridLoRAStrategy

def monitor_gpu_usage() -> Dict[str, Any]:
    """Monitor GPU memory usage and utilization"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    try:
        # Get GPU memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        # Try to get GPU utilization if nvidia-ml-py is available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_util = utilization.memory
        except ImportError:
            gpu_util = memory_util = -1  # Not available

        return {
            "gpu_available": True,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "memory_total_gb": memory_total,
            "memory_usage_percent": (memory_allocated / memory_total) * 100,
            "gpu_utilization_percent": gpu_util,
            "memory_utilization_percent": memory_util
        }
    except Exception as e:
        return {"gpu_available": True, "error": str(e)}


def log_gpu_info(prefix: str = ""):
    """Log GPU information for debugging"""
    gpu_info = monitor_gpu_usage()
    if gpu_info.get("gpu_available", False):
        if "error" not in gpu_info:
            logging.info(f"{prefix}GPU Memory: {gpu_info['memory_allocated_gb']:.2f}GB/{gpu_info['memory_total_gb']:.2f}GB "
                        f"({gpu_info['memory_usage_percent']:.1f}%)")
            if gpu_info['gpu_utilization_percent'] >= 0:
                logging.info(f"{prefix}GPU Utilization: {gpu_info['gpu_utilization_percent']}%")
        else:
            logging.warning(f"{prefix}GPU monitoring error: {gpu_info['error']}")
    else:
        logging.warning(f"{prefix}GPU not available")


class DeblurDataset(Dataset):
    """Dataset class for loading deblurring image pairs"""

    def __init__(self, clean_dir: str, blur_dir: str, transform=None):
        self.clean_dir = Path(clean_dir)
        self.blur_dir = Path(blur_dir)
        self.transform = transform

        # Get all clean images
        self.clean_files = sorted(list(self.clean_dir.glob("*.png")))

        # 🚀 持久化索引系统：建立一次，永久使用
        index_file = self.blur_dir / ".dataset_index.json"
        blur_index = self._load_or_build_index(index_file)

        # Match clean images with corresponding blur images
        self.image_pairs = []
        for clean_file in self.clean_files:
            clean_stem = clean_file.stem
            if clean_stem in blur_index:
                # Add all pairs for this clean image
                for blur_file_str in blur_index[clean_stem]:
                    blur_file = Path(blur_file_str)
                    if blur_file.exists():  # 确保文件仍然存在
                        self.image_pairs.append((clean_file, blur_file))

        print(f"Found {len(self.image_pairs)} image pairs")

    def _load_or_build_index(self, index_file: Path) -> Dict[str, list]:
        """加载或构建数据集索引"""

        # 计算目录内容的哈希值来检测变化
        blur_files = sorted(list(self.blur_dir.glob("*.png")))
        content_hash = self._calculate_dir_hash(blur_files)

        # 尝试加载现有索引
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)

                # 检查索引是否仍然有效
                if index_data.get('content_hash') == content_hash:
                    print(f"📁 Using cached dataset index ({len(index_data['index'])} entries)")
                    return index_data['index']
                else:
                    print("📁 Directory content changed, rebuilding index...")
            except (json.JSONDecodeError, KeyError):
                print("📁 Invalid index file, rebuilding...")

        # 构建新索引
        print("📁 Building dataset index...")
        blur_index = {}

        for blur_file in blur_files:
            # 从blur文件名中提取对应的clean文件名
            # 假设格式: spectrum_X_difficulty_effect.png -> spectrum_X
            parts = blur_file.stem.split('_')
            if len(parts) >= 2:
                clean_stem = '_'.join(parts[:2])  # spectrum_X
                if clean_stem not in blur_index:
                    blur_index[clean_stem] = []
                blur_index[clean_stem].append(str(blur_file))

        # 保存索引
        index_data = {
            'content_hash': content_hash,
            'index': blur_index,
            'created_at': time.time(),
            'total_blur_files': len(blur_files),
            'total_clean_stems': len(blur_index)
        }

        try:
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            print(f"💾 Saved dataset index to {index_file}")
        except Exception as e:
            print(f"⚠️  Failed to save index: {e}")

        print(f"📁 Built index: {len(blur_index)} clean stems, {len(blur_files)} blur files")
        return blur_index

    def _calculate_dir_hash(self, file_list: list) -> str:
        """计算目录内容的哈希值"""
        # 使用文件名和修改时间来生成哈希
        content_str = ""
        for file_path in file_list:
            try:
                mtime = file_path.stat().st_mtime
                content_str += f"{file_path.name}:{mtime}:"
            except OSError:
                content_str += f"{file_path.name}:0:"

        return hashlib.md5(content_str.encode()).hexdigest()

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
                 device: Union[str, torch.device] = 'cuda',
                 mixed_precision: bool = True,
                 precision_dtype: str = 'float16',  # 🆕 'float16' or 'bfloat16'
                 use_multi_gpu: bool = True,
                 gradient_accumulation_steps: int = 1,  # 🆕 梯度累积步数
                 max_grad_norm: float = 1.0,  # 🆕 梯度裁剪阈值
                 hybrid_lora_strategy: Optional["HybridLoRAStrategy"] = None,
                 metrics_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
                 validation_inference_steps: int = 20):

        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.use_multi_gpu = use_multi_gpu
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm if max_grad_norm and max_grad_norm > 0 else None
        self.hybrid_lora_strategy = hybrid_lora_strategy
        self.metrics_logger = metrics_logger
        self.validation_inference_steps = validation_inference_steps

        # 🚀 多GPU支持：自动检测并使用可用GPU
        if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs for training!")
            self.model = nn.DataParallel(model).to(self.device)
            self.is_multi_gpu = True
        else:
            self.model = model.to(self.device)
            self.is_multi_gpu = False

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision

        # 🆕 Enhanced mixed precision support with bfloat16
        if mixed_precision:
            # Determine precision dtype
            if precision_dtype.lower() == 'bfloat16':
                if not torch.cuda.is_bf16_supported():
                    print("⚠️ bfloat16 not supported on this GPU, falling back to float16")
                    self.precision_dtype = torch.float16
                else:
                    self.precision_dtype = torch.bfloat16
                    print("✅ Using bfloat16 mixed precision")
            else:
                self.precision_dtype = torch.float16
                print("✅ Using float16 mixed precision")

            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.precision_dtype == torch.float16))
        else:
            self.precision_dtype = torch.float32
            self.scaler = None

        self.train_losses = []
        self.val_losses = []

    def _get_base_model(self) -> nn.Module:
        """Return the underlying model (unwrapped from DataParallel if needed)."""
        return self.model.module if getattr(self, "is_multi_gpu", False) else self.model

    def _autocast(self):
        """Return appropriate autocast context manager based on precision settings."""
        device_str = str(self.device)
        if not self.mixed_precision or 'cuda' not in device_str:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.precision_dtype)

    def _apply_hybrid_strategy(self, epoch_idx: int) -> None:
        """Apply Hybrid LoRA strategy unfreezing logic when configured."""
        if not self.hybrid_lora_strategy:
            return
        base_model = self._get_base_model()
        self.hybrid_lora_strategy.update_frozen_layers(base_model, epoch_idx)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = max(1, len(train_loader))

        base_model = self._get_base_model()
        is_diffusion_model = hasattr(base_model, 'scheduler') and hasattr(base_model, 'unet')

        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (clean_images, blur_images) in enumerate(progress_bar):
            clean_images = clean_images.to(self.device, non_blocking=True)
            blur_images = blur_images.to(self.device, non_blocking=True)

            outputs = None
            with self._autocast():
                if is_diffusion_model:
                    noise_pred, noise_target = self.model(clean_images, blur_images)
                    loss = self.loss_fn(noise_pred, noise_target)
                else:
                    outputs = self.model(blur_images)
                    loss = self.loss_fn(outputs, clean_images)

            loss_value = loss.detach().item()
            total_loss += loss_value

            loss_to_backward = loss / max(1, self.gradient_accumulation_steps)

            if self.mixed_precision:
                self.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            should_step = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0 or
                (batch_idx + 1) == num_batches
            )

            if should_step:
                if self.mixed_precision and self.max_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if is_diffusion_model or outputs is None:
                    batch_psnr = 0.0
                else:
                    pred_01 = (outputs + 1) / 2
                    clean_01 = (clean_images + 1) / 2
                    batch_psnr = calculate_psnr(pred_01, clean_01)

            total_psnr += batch_psnr

            progress_bar.set_postfix({
                'Loss': f'{loss_value:.4f}',
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
        num_batches = max(1, len(val_loader))

        base_model = self._get_base_model()
        is_diffusion_model = hasattr(base_model, 'scheduler') and hasattr(base_model, 'unet')

        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for clean_images, blur_images in progress_bar:
                clean_images = clean_images.to(self.device, non_blocking=True)
                blur_images = blur_images.to(self.device, non_blocking=True)

                with self._autocast():
                    if is_diffusion_model:
                        outputs = self.model.inference(
                            blur_images,
                            num_inference_steps=self.validation_inference_steps
                        )
                        noise_pred, noise_target = self.model(clean_images, blur_images)
                        loss = self.loss_fn(noise_pred, noise_target)
                    else:
                        outputs = self.model(blur_images)
                        loss = self.loss_fn(outputs, clean_images)

                loss_value = loss.detach().item()

                pred_01 = (outputs + 1) / 2
                clean_01 = (clean_images + 1) / 2

                batch_psnr = calculate_psnr(pred_01, clean_01)

                try:
                    from torchmetrics import StructuralSimilarityIndexMeasure
                    if not hasattr(self, '_ssim_metric'):
                        self._ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                    batch_ssim = self._ssim_metric(pred_01[:1], clean_01[:1]).item()
                except ImportError:
                    pred_np = pred_01[0].cpu().numpy().transpose(1, 2, 0)
                    clean_np = clean_01[0].cpu().numpy().transpose(1, 2, 0)
                    batch_ssim = calculate_ssim(pred_np, clean_np)

                total_loss += loss_value
                total_psnr += batch_psnr
                total_ssim += batch_ssim

                progress_bar.set_postfix({
                    'Loss': f'{loss_value:.4f}',
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
            'val_loss': [], 'val_psnr': [], 'val_ssim': [],
            'epoch_time': [], 'samples_per_second': []
        }

        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Model: {self.model.__class__.__name__}")
        logging.info(f"Device: {self.device}")

        # Initial GPU monitoring
        log_gpu_info("Initial ")

        # Check if GPU utilization monitoring is available
        gpu_info = monitor_gpu_usage()
        if gpu_info.get("gpu_utilization_percent", -1) < 0:
            logging.warning("GPU utilization monitoring not available. Install nvidia-ml-py for detailed monitoring: pip install nvidia-ml-py")

        use_cuda_timing = torch.cuda.is_available() and 'cuda' in str(self.device)

        for epoch in range(num_epochs):
            if use_cuda_timing:
                epoch_start_event = torch.cuda.Event(enable_timing=True)
                epoch_end_event = torch.cuda.Event(enable_timing=True)
                epoch_start_event.record()
            else:
                epoch_start_event = epoch_end_event = None
                wall_time_start = time.time()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Log GPU status at epoch start
            log_gpu_info(f"Epoch {epoch + 1} start - ")

            # Update hybrid LoRA freezing schedule if configured
            self._apply_hybrid_strategy(epoch + 1)

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

            if use_cuda_timing:
                epoch_end_event.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start_event.elapsed_time(epoch_end_event) / 1000.0
            else:
                epoch_time = time.time() - wall_time_start

            if hasattr(train_loader, 'dataset'):
                try:
                    epoch_samples = len(train_loader.dataset)
                except TypeError:
                    epoch_samples = max(1, getattr(train_loader, 'batch_size', 1)) * len(train_loader)
            else:
                epoch_samples = max(1, getattr(train_loader, 'batch_size', 1)) * len(train_loader)

            samples_per_second = epoch_samples / epoch_time if epoch_time > 0 else 0.0

            history['epoch_time'].append(epoch_time)
            history['samples_per_second'].append(samples_per_second)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train PSNR: {train_metrics['psnr']:.2f}dB")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val PSNR: {val_metrics['psnr']:.2f}dB, Val SSIM: {val_metrics['ssim']:.3f}")
            print(f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s, Samples/s: {samples_per_second:.1f}")

            if self.metrics_logger:
                self.metrics_logger({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_psnr': train_metrics['psnr'],
                    'val_loss': val_metrics['loss'],
                    'val_psnr': val_metrics['psnr'],
                    'val_ssim': val_metrics['ssim'],
                    'lr': current_lr,
                    'epoch_time': epoch_time,
                    'samples_per_second': samples_per_second
                })

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

        # 🚀 多GPU兼容性：保存原始模型权重而非DataParallel包装的权重
        if self.is_multi_gpu:
            model_state_dict = self.model.module.state_dict()
            model_info = self.model.module.get_model_info() if hasattr(self.model.module, 'get_model_info') else {}
        else:
            model_state_dict = self.model.state_dict()
            model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'is_best': is_best,
            'model_info': model_info,
            'is_multi_gpu': self.is_multi_gpu  # 记录是否使用多GPU训练
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


def diagnose_gpu_performance(model: nn.Module, sample_batch_size: int = 4, device: str = 'cuda') -> None:
    """Diagnose potential GPU performance issues"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print("\n" + "="*50)
    print("🔍 GPU PERFORMANCE DIAGNOSIS")
    print("="*50)

    # Check device and memory
    gpu_info = monitor_gpu_usage()
    print(f"📊 GPU Info:")
    print(f"  - Total Memory: {gpu_info.get('memory_total_gb', 'Unknown'):.2f} GB")
    print(f"  - Current Usage: {gpu_info.get('memory_allocated_gb', 0):.2f} GB ({gpu_info.get('memory_usage_percent', 0):.1f}%)")

    # Test model forward pass
    model.eval()
    print(f"\n🧪 Testing model forward pass (batch_size={sample_batch_size})...")

    try:
        # Create dummy inputs
        if hasattr(model, 'scheduler') and hasattr(model, 'unet'):
            # Diffusion model
            clean_dummy = torch.randn(sample_batch_size, 3, 512, 512, device=device)
            blur_dummy = torch.randn(sample_batch_size, 3, 512, 512, device=device)

            start_time = time.time()
            torch.cuda.synchronize()

            with torch.no_grad():
                _ = model(clean_dummy, blur_dummy)

            torch.cuda.synchronize()
            forward_time = time.time() - start_time

            print(f"  ✅ Forward pass successful: {forward_time:.3f}s")

            # Test inference
            print(f"🔄 Testing inference (5 denoising steps)...")
            start_time = time.time()
            torch.cuda.synchronize()

            with torch.no_grad():
                _ = model.inference(blur_dummy, num_inference_steps=5)

            torch.cuda.synchronize()
            inference_time = time.time() - start_time

            print(f"  ✅ Inference successful: {inference_time:.3f}s")

        else:
            # Regular model
            dummy_input = torch.randn(sample_batch_size, 3, 512, 512, device=device)

            start_time = time.time()
            torch.cuda.synchronize()

            with torch.no_grad():
                _ = model(dummy_input)

            torch.cuda.synchronize()
            forward_time = time.time() - start_time

            print(f"  ✅ Forward pass successful: {forward_time:.3f}s")

        # Check GPU memory after test
        gpu_info_after = monitor_gpu_usage()
        print(f"  📈 GPU Memory after test: {gpu_info_after.get('memory_allocated_gb', 0):.2f} GB")

    except Exception as e:
        print(f"  ❌ Error during testing: {e}")

    print("\n💡 Performance Tips:")
    print("  - Ensure batch_size fully utilizes GPU memory")
    print("  - Use mixed precision (amp) for better performance")
    print("  - Monitor nvidia-smi during training for real-time GPU usage")
    print("  - Consider using pin_memory=True in DataLoader")
    print("="*50)
