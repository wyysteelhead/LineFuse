import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import math


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for efficient fine-tuning"""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super(LoRALayer, self).__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layers"""
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(self,
                 linear_layer: nn.Linear,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super(LoRALinear, self).__init__()

        # Freeze the original linear layer
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # Add LoRA adaptation
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original layer and LoRA"""
        return self.linear(x) + self.lora(x)


class LoRAConv2d(nn.Module):
    """2D Convolution layer with LoRA adaptation"""

    def __init__(self,
                 conv_layer: nn.Conv2d,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super(LoRAConv2d, self).__init__()

        # Freeze the original conv layer
        self.conv = conv_layer
        for param in self.conv.parameters():
            param.requires_grad = False

        # LoRA adaptation for conv layers using 1x1 convolutions
        self.lora_A = nn.Conv2d(
            conv_layer.in_channels,
            rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.lora_B = nn.Conv2d(
            rank,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=False
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha / rank

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original conv and LoRA"""
        original_out = self.conv(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return original_out + lora_out


class LoRAAdapter:
    """Manager class for applying LoRA to models"""

    def __init__(self,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0,
                 target_modules: Optional[List[str]] = None):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["to_q", "to_v", "to_k", "to_out"]
        self.lora_layers: Dict[str, Union[LoRALinear, LoRAConv2d]] = {}

    def apply_lora_to_model(self, model: nn.Module, module_filter: Optional[str] = None) -> nn.Module:
        """Apply LoRA to specified modules in a model

        Args:
            model: The model to adapt
            module_filter: Filter to apply LoRA only to specific modules (e.g., "mid_block", "up_blocks")
        """
        lora_applied_count = 0

        for name, module in model.named_modules():
            # Apply module filter if specified
            if module_filter and module_filter not in name:
                continue

            # Check if this module should have LoRA applied
            should_apply = False
            for target in self.target_modules:
                if target in name:
                    should_apply = True
                    break

            if not should_apply:
                continue

            # Apply LoRA based on layer type
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout
                )

                # Replace the module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model

                setattr(parent, child_name, lora_layer)
                self.lora_layers[name] = lora_layer
                lora_applied_count += 1

            elif isinstance(module, nn.Conv2d):
                lora_layer = LoRAConv2d(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout
                )

                # Replace the module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model

                setattr(parent, child_name, lora_layer)
                self.lora_layers[name] = lora_layer
                lora_applied_count += 1

        print(f"✅ Applied LoRA to {lora_applied_count} modules")
        return model

    def get_lora_parameters(self, model: nn.Module) -> List[torch.nn.Parameter]:
        """Get only LoRA parameters for optimization"""
        lora_params = []
        for name, module in model.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv2d)):
                lora_params.extend(module.lora.parameters() if hasattr(module, 'lora') else
                                 list(module.lora_A.parameters()) + list(module.lora_B.parameters()))
        return lora_params

    def get_trainable_parameter_count(self, model: nn.Module) -> Dict[str, int]:
        """Get count of trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for p in self.get_lora_parameters(model))

        return {
            'total': total_params,
            'trainable': trainable_params,
            'lora': lora_params,
            'frozen': total_params - trainable_params
        }

    def save_lora_weights(self, model: nn.Module, filepath: str) -> None:
        """Save only LoRA weights"""
        lora_state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"{name}.lora.lora_A.weight"] = module.lora.lora_A.weight
                lora_state_dict[f"{name}.lora.lora_B.weight"] = module.lora.lora_B.weight
            elif isinstance(module, LoRAConv2d):
                lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
                lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight

        torch.save({
            'lora_state_dict': lora_state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'target_modules': self.target_modules
        }, filepath)
        print(f"💾 LoRA weights saved to {filepath}")

    def load_lora_weights(self, model: nn.Module, filepath: str) -> None:
        """Load LoRA weights"""
        checkpoint = torch.load(filepath, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']

        for name, param in lora_state_dict.items():
            # Navigate to the parameter in the model
            module_path = name.split('.')
            current_module = model

            for path_part in module_path[:-1]:
                current_module = getattr(current_module, path_part)

            # Set the parameter
            if hasattr(current_module, module_path[-1]):
                getattr(current_module, module_path[-1]).data.copy_(param)

        print(f"✅ LoRA weights loaded from {filepath}")


def create_diffusion_lora_config() -> Dict[str, Any]:
    """Create configuration for diffusion model LoRA adaptation"""
    return {
        'rank': 16,
        'alpha': 32.0,
        'dropout': 0.1,
        'target_modules': [
            # UNet attention layers
            'to_q', 'to_v', 'to_k', 'to_out.0',
            # UNet projection layers
            'proj_in', 'proj_out',
            # Time embedding layers
            'time_proj', 'time_embedding',
            # Mid block layers
            'mid_block',
            # Up block layers (last 2 blocks for efficiency)
            'up_blocks.2', 'up_blocks.3'
        ],
        'module_filters': ['mid_block', 'up_blocks.2', 'up_blocks.3']  # Focus on key blocks
    }


class HybridLoRAStrategy:
    """Hybrid LoRA fine-tuning strategy for progressive unfreezing"""

    def __init__(self, total_epochs: int, unfreeze_schedule: Dict[int, List[str]]):
        """
        Args:
            total_epochs: Total training epochs
            unfreeze_schedule: Dict mapping epoch -> list of modules to unfreeze
                Example: {10: ['mid_block'], 20: ['up_blocks.2', 'up_blocks.3']}
        """
        self.total_epochs = total_epochs
        self.unfreeze_schedule = unfreeze_schedule

    def update_frozen_layers(self, model: nn.Module, current_epoch: int) -> None:
        """Update which layers are frozen based on current epoch"""
        if current_epoch not in self.unfreeze_schedule:
            return

        modules_to_unfreeze = self.unfreeze_schedule[current_epoch]
        unfrozen_count = 0

        for name, module in model.named_modules():
            for unfreeze_pattern in modules_to_unfreeze:
                if unfreeze_pattern in name:
                    for param in module.parameters():
                        if param.requires_grad == False:
                            param.requires_grad = True
                            unfrozen_count += 1

        print(f"🔓 Epoch {current_epoch}: Unfroze {unfrozen_count} parameters")