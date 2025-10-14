import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from .lora_adapter import LoRAAdapter, create_diffusion_lora_config, HybridLoRAStrategy


class LoRAConditionalDiffusionModel(nn.Module):
    """LoRA-enhanced Conditional Diffusion Model for efficient image deblurring"""

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 sample_size: int = 512,
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
                 layers_per_block: int = 2,
                 num_train_timesteps: int = 1000,
                 scheduler_type: str = "ddim",  # "ddpm" or "ddim"
                 lora_config: Optional[Dict[str, Any]] = None,
                 use_lora: bool = True,
                 enable_gradient_checkpointing: bool = True):

        super(LoRAConditionalDiffusionModel, self).__init__()

        # Create base UNet
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels + 3,  # +3 for conditioning image
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=8
        )

        # Enable gradient checkpointing for memory efficiency
        if enable_gradient_checkpointing and hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()

        # Initialize scheduler
        if scheduler_type.lower() == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False
            )
        else:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule="linear",
                prediction_type="epsilon"
            )

        # Apply LoRA if specified
        self.use_lora = use_lora
        if use_lora:
            if lora_config is None:
                lora_config = create_diffusion_lora_config()

            self.lora_adapter = LoRAAdapter(
                rank=lora_config.get('rank', 16),
                alpha=lora_config.get('alpha', 32.0),
                dropout=lora_config.get('dropout', 0.1),
                target_modules=lora_config.get('target_modules', ['to_q', 'to_v', 'to_k', 'to_out'])
            )

            # Apply LoRA to key blocks only for efficiency
            for module_filter in lora_config.get('module_filters', ['mid_block', 'up_blocks']):
                self.unet = self.lora_adapter.apply_lora_to_model(self.unet, module_filter)
        else:
            self.lora_adapter = None

        self.num_train_timesteps = num_train_timesteps
        self.sample_size = sample_size
        self.scheduler_type = scheduler_type

        # Print parameter statistics
        self._print_parameter_stats()

    def _print_parameter_stats(self):
        """Print parameter statistics including LoRA breakdown"""
        if self.use_lora:
            param_stats = self.lora_adapter.get_trainable_parameter_count(self)
            print(f"📊 Model Parameter Statistics:")
            print(f"  Total parameters: {param_stats['total']:,}")
            print(f"  LoRA parameters: {param_stats['lora']:,}")
            print(f"  Frozen parameters: {param_stats['frozen']:,}")
            print(f"  Trainable ratio: {param_stats['lora'] / param_stats['total'] * 100:.1f}%")
        else:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"📊 Model Parameter Statistics:")
            print(f"  Total parameters: {total:,}")
            print(f"  Trainable parameters: {trainable:,}")

    def forward(self,
                clean_images: torch.Tensor,
                blur_images: torch.Tensor,
                timesteps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = clean_images.shape[0]
        device = clean_images.device

        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_train_timesteps, (batch_size,), device=device
            ).long()

        noise = torch.randn_like(clean_images)

        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)

        model_input = torch.cat([noisy_images, blur_images], dim=1)

        noise_pred = self.unet(
            model_input,
            timesteps,
            return_dict=False
        )[0]

        return noise_pred, noise

    def inference(self,
                  blur_images: torch.Tensor,
                  num_inference_steps: int = 25,  # Reduced from 50 for faster inference
                  guidance_scale: float = 1.0,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:

        device = blur_images.device
        batch_size = blur_images.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Initialize with noise
        denoised_images = torch.randn(
            (batch_size, 3, blur_images.shape[2], blur_images.shape[3]),
            device=device,
            generator=generator
        )

        # Denoising loop
        for timestep in self.scheduler.timesteps:
            timestep_batch = timestep.repeat(batch_size)

            model_input = torch.cat([denoised_images, blur_images], dim=1)

            with torch.no_grad():
                noise_pred = self.unet(
                    model_input,
                    timestep_batch,
                    return_dict=False
                )[0]

            # Scheduler step
            denoised_images = self.scheduler.step(
                noise_pred, timestep, denoised_images, return_dict=False
            )[0]

        return torch.clamp(denoised_images, -1, 1)

    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Get LoRA parameters for optimization"""
        if self.use_lora and self.lora_adapter:
            return self.lora_adapter.get_lora_parameters(self)
        return []

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters (LoRA + any unfrozen layers)"""
        return [p for p in self.parameters() if p.requires_grad]

    def save_lora_weights(self, filepath: str) -> None:
        """Save only LoRA weights"""
        if self.use_lora and self.lora_adapter:
            self.lora_adapter.save_lora_weights(self, filepath)
        else:
            print("⚠️ LoRA not enabled - cannot save LoRA weights")

    def load_lora_weights(self, filepath: str) -> None:
        """Load LoRA weights"""
        if self.use_lora and self.lora_adapter:
            self.lora_adapter.load_lora_weights(self, filepath)
        else:
            print("⚠️ LoRA not enabled - cannot load LoRA weights")

    def get_model_size(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        info = {
            'name': 'LoRA-ConditionalDiffusion',
            'sample_size': self.sample_size,
            'num_train_timesteps': self.num_train_timesteps,
            'scheduler_type': self.scheduler_type,
            'total_params': self.get_model_size(),
            'use_lora': self.use_lora
        }

        if self.use_lora and self.lora_adapter:
            param_stats = self.lora_adapter.get_trainable_parameter_count(self)
            info.update({
                'lora_params': param_stats['lora'],
                'frozen_params': param_stats['frozen'],
                'lora_rank': self.lora_adapter.rank,
                'lora_alpha': self.lora_adapter.alpha
            })
        else:
            info['trainable_params'] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return info


class MultiDomainLoRAModel(nn.Module):
    """Multi-domain LoRA model for different blur types"""

    def __init__(self,
                 base_model_config: Dict[str, Any],
                 blur_domains: List[str],
                 lora_config: Optional[Dict[str, Any]] = None):
        super(MultiDomainLoRAModel, self).__init__()

        self.blur_domains = blur_domains
        self.domain_models = nn.ModuleDict()

        # Create separate LoRA adapters for each blur domain
        base_lora_config = lora_config or create_diffusion_lora_config()

        for domain in blur_domains:
            # Create model with domain-specific LoRA
            domain_config = base_lora_config.copy()
            domain_config['rank'] = base_lora_config.get('rank', 8)  # Can vary by domain

            model = LoRAConditionalDiffusionModel(
                lora_config=domain_config,
                **base_model_config
            )
            self.domain_models[domain] = model

    def forward(self, clean_images: torch.Tensor, blur_images: torch.Tensor,
                domain: str, timesteps: Optional[torch.Tensor] = None):
        """Forward pass for specific domain"""
        if domain not in self.domain_models:
            raise ValueError(f"Unknown domain: {domain}. Available: {self.blur_domains}")

        return self.domain_models[domain](clean_images, blur_images, timesteps)

    def inference(self, blur_images: torch.Tensor, domain: str, **kwargs):
        """Inference for specific domain"""
        if domain not in self.domain_models:
            raise ValueError(f"Unknown domain: {domain}. Available: {self.blur_domains}")

        return self.domain_models[domain].inference(blur_images, **kwargs)

    def save_domain_weights(self, domain: str, filepath: str):
        """Save LoRA weights for specific domain"""
        if domain in self.domain_models:
            self.domain_models[domain].save_lora_weights(filepath)

    def load_domain_weights(self, domain: str, filepath: str):
        """Load LoRA weights for specific domain"""
        if domain in self.domain_models:
            self.domain_models[domain].load_lora_weights(filepath)


class EnhancedDiffusionLoss(nn.Module):
    """Enhanced loss function with Charbonnier loss option"""

    def __init__(self,
                 l1_weight: float = 1.0,
                 l2_weight: float = 0.5,
                 charbonnier_weight: float = 0.0,
                 charbonnier_eps: float = 1e-6,
                 perceptual_weight: float = 0.1):
        super(EnhancedDiffusionLoss, self).__init__()

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.charbonnier_weight = charbonnier_weight
        self.charbonnier_eps = charbonnier_eps
        self.perceptual_weight = perceptual_weight

        # Load VGG for perceptual loss if needed
        if perceptual_weight > 0:
            try:
                from torchvision.models import vgg16
                vgg = vgg16(pretrained=True).features[:16]
                for param in vgg.parameters():
                    param.requires_grad = False
                self.vgg = vgg
            except ImportError:
                print("⚠️ torchvision not available - disabling perceptual loss")
                self.perceptual_weight = 0.0

    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Charbonnier loss - more robust to outliers than L2"""
        diff = pred - target
        return torch.sqrt(diff * diff + self.charbonnier_eps).mean()

    def forward(self,
                noise_pred: torch.Tensor,
                noise_target: torch.Tensor,
                denoised_pred: Optional[torch.Tensor] = None,
                clean_target: Optional[torch.Tensor] = None) -> torch.Tensor:

        loss = 0.0

        # Primary noise prediction loss
        if self.l2_weight > 0:
            mse_loss = F.mse_loss(noise_pred, noise_target)
            loss += self.l2_weight * mse_loss

        if self.l1_weight > 0:
            l1_loss = F.l1_loss(noise_pred, noise_target)
            loss += self.l1_weight * l1_loss

        if self.charbonnier_weight > 0:
            charbonnier_loss = self.charbonnier_loss(noise_pred, noise_target)
            loss += self.charbonnier_weight * charbonnier_loss

        # Perceptual loss on denoised images if available
        if (self.perceptual_weight > 0 and
            denoised_pred is not None and
            clean_target is not None and
            hasattr(self, 'vgg')):

            if denoised_pred.shape[1] == 3:  # RGB images
                pred_features = self.vgg(denoised_pred)
                target_features = self.vgg(clean_target)
                perceptual_loss = F.mse_loss(pred_features, target_features)
                loss += self.perceptual_weight * perceptual_loss

        return loss


def create_lora_diffusion_model(config: Dict[str, Any]) -> LoRAConditionalDiffusionModel:
    """Factory function to create LoRA diffusion model"""

    lora_config = config.get('lora', create_diffusion_lora_config())

    model = LoRAConditionalDiffusionModel(
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 3),
        sample_size=config.get('sample_size', 512),
        block_out_channels=tuple(config.get('block_out_channels', [128, 256, 512, 1024])),
        layers_per_block=config.get('layers_per_block', 2),
        num_train_timesteps=config.get('num_train_timesteps', 1000),
        scheduler_type=config.get('scheduler_type', 'ddim'),
        lora_config=lora_config,
        use_lora=config.get('use_lora', True),
        enable_gradient_checkpointing=config.get('enable_gradient_checkpointing', True)
    )

    return model


def create_enhanced_loss(config: Dict[str, Any]) -> EnhancedDiffusionLoss:
    """Factory function to create enhanced loss"""

    return EnhancedDiffusionLoss(
        l1_weight=config.get('l1_weight', 1.0),
        l2_weight=config.get('l2_weight', 0.5),
        charbonnier_weight=config.get('charbonnier_weight', 0.5),
        charbonnier_eps=config.get('charbonnier_eps', 1e-6),
        perceptual_weight=config.get('perceptual_weight', 0.1)
    )