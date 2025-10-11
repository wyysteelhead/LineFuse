import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

try:
    from diffusers import UNet2DConditionModel, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  diffusers库未安装，将使用简化版扩散模型")
    print("完整功能请运行: pip install diffusers")

class SimpleDDPMScheduler:
    """Simplified DDPM scheduler when diffusers is not available"""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        """Add noise to samples according to the noise schedule"""
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Set the timesteps used for sampling"""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        self.timesteps = torch.flip(timesteps, [0]).to(device)


class SimpleUNet(nn.Module):
    """Simplified U-Net for diffusion when diffusers is not available"""

    def __init__(self, in_channels: int = 6, out_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels + 512 // (64 * 64), base_channels * 2, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1, stride=2)
        self.enc4 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1, stride=2)

        # Middle
        self.mid = nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1)

        # Decoder
        self.dec4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_channels * 4, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(base_channels * 2, out_channels, 3, padding=1)

    def positional_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding for timesteps"""
        half_dim = 64
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs):
        # Time embedding
        t_emb = self.positional_encoding(timesteps)
        t_emb = self.time_embed(t_emb)

        # Encoder
        e1 = F.relu(self.enc1(x))

        # Add time embedding (simplified approach)
        t_spatial = t_emb.view(t_emb.shape[0], -1, 1, 1).expand(-1, -1, e1.shape[2], e1.shape[3])
        e1_with_time = torch.cat([e1, t_spatial[:, :e1.shape[2]*e1.shape[3]//64//64, :, :]], dim=1)

        e2 = F.relu(self.enc2(e1_with_time))
        e3 = F.relu(self.enc3(e2))
        e4 = F.relu(self.enc4(e3))

        # Middle
        m = F.relu(self.mid(e4))

        # Decoder with skip connections
        d4 = F.relu(self.dec4(m))
        d3 = F.relu(self.dec3(torch.cat([d4, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return (d1,)  # Return as tuple to match diffusers interface


class ConditionalDiffusionModel(nn.Module):
    """Conditional Diffusion Model for image deblurring"""

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 sample_size: int = 512,
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
                 layers_per_block: int = 2,
                 num_train_timesteps: int = 1000):

        super(ConditionalDiffusionModel, self).__init__()

        if DIFFUSERS_AVAILABLE:
            self.unet = UNet2DConditionModel(
                sample_size=sample_size,
                in_channels=in_channels + 3,  # +3 for conditioning image
                out_channels=out_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                cross_attention_dim=768,
                encoder_hid_dim=8
            )

            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule="linear",
                prediction_type="epsilon"
            )
        else:
            self.unet = SimpleUNet(in_channels=in_channels + 3, out_channels=out_channels)
            self.scheduler = SimpleDDPMScheduler(num_train_timesteps=num_train_timesteps)

        self.num_train_timesteps = num_train_timesteps
        self.sample_size = sample_size
        
    def forward(self, 
                clean_images: torch.Tensor,
                blur_images: torch.Tensor,
                timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        
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
            encoder_hidden_states=None,  # 图像条件任务不需要文本编码
            return_dict=False
        )[0]
        
        return noise_pred, noise
    
    def inference(self, 
                  blur_images: torch.Tensor,
                  num_inference_steps: int = 50,
                  guidance_scale: float = 1.0) -> torch.Tensor:
        
        device = blur_images.device
        batch_size = blur_images.shape[0]
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        denoised_images = torch.randn(
            (batch_size, 3, blur_images.shape[2], blur_images.shape[3]),
            device=device
        )
        
        for timestep in self.scheduler.timesteps:
            timestep_batch = timestep.repeat(batch_size)
            
            model_input = torch.cat([denoised_images, blur_images], dim=1)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    model_input,
                    timestep_batch,
                    encoder_hidden_states=None,  # 图像条件任务不需要文本编码
                    return_dict=False
                )[0]
            
            denoised_images = self.scheduler.step(
                noise_pred, timestep, denoised_images, return_dict=False
            )[0]
        
        return torch.clamp(denoised_images, -1, 1)  # Match tanh output range

    def get_model_size(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        return {
            'name': 'ConditionalDiffusion',
            'sample_size': self.sample_size,
            'num_train_timesteps': self.num_train_timesteps,
            'diffusers_available': DIFFUSERS_AVAILABLE,
            'total_params': self.get_model_size(),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class DiffusionLoss(nn.Module):
    def __init__(self, 
                 l1_weight: float = 1.0,
                 l2_weight: float = 0.5,
                 perceptual_weight: float = 0.1):
        super(DiffusionLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        
        if perceptual_weight > 0:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features[:16]
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
    
    def forward(self, 
                noise_pred: torch.Tensor,
                noise_target: torch.Tensor,
                denoised_pred: Optional[torch.Tensor] = None,
                clean_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        loss = 0.0
        
        mse_loss = F.mse_loss(noise_pred, noise_target)
        loss += self.l2_weight * mse_loss
        
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(noise_pred, noise_target)
            loss += self.l1_weight * l1_loss
        
        if (self.perceptual_weight > 0 and 
            denoised_pred is not None and 
            clean_target is not None):
            
            if denoised_pred.shape[1] == 3:
                pred_features = self.vgg(denoised_pred)
                target_features = self.vgg(clean_target)
                perceptual_loss = F.mse_loss(pred_features, target_features)
                loss += self.perceptual_weight * perceptual_loss
        
        return loss