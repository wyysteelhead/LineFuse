import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from typing import Optional, Tuple

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,
                 sample_size: int = 512,
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
                 layers_per_block: int = 2,
                 num_train_timesteps: int = 1000):
        
        super(ConditionalDiffusionModel, self).__init__()
        
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels + 3,  # +3 for conditioning image
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=None,
            encoder_hid_dim=None
        )
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.num_train_timesteps = num_train_timesteps
        
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
                    return_dict=False
                )[0]
            
            denoised_images = self.scheduler.step(
                noise_pred, timestep, denoised_images, return_dict=False
            )[0]
        
        return torch.clamp(denoised_images, 0, 1)

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