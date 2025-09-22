import torch
import cv2
import numpy as np
import base64
import io
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
import psutil

from ..models.unet_baseline import UNetBaseline
from ..models.diffusion_model import ConditionalDiffusionModel
from .models import ModelType, DeblurResponse, BatchDeblurResponse, ModelInfo

class DeblurService:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_paths: Dict[str, str] = {}
        
        logging.info(f"Initialized DeblurService on device: {device}")
    
    def load_model(self, model_type: ModelType, model_path: str) -> bool:
        try:
            if model_type == ModelType.UNET_BASELINE:
                model = UNetBaseline(n_channels=3, n_classes=3)
            elif model_type == ModelType.DIFFUSION:
                model = ConditionalDiffusionModel()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.models[model_type.value] = model
            self.model_paths[model_type.value] = model_path
            
            logging.info(f"Successfully loaded {model_type.value} model from {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model {model_type.value}: {e}")
            return False
    
    def unload_model(self, model_type: ModelType) -> bool:
        try:
            if model_type.value in self.models:
                del self.models[model_type.value]
                if model_type.value in self.model_paths:
                    del self.model_paths[model_type.value]
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logging.info(f"Successfully unloaded {model_type.value} model")
                return True
            else:
                logging.warning(f"Model {model_type.value} is not loaded")
                return False
                
        except Exception as e:
            logging.error(f"Failed to unload model {model_type.value}: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = image.astype(np.float32) / 255.0
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
        return tensor
    
    def base64_to_image(self, base64_str: str) -> np.ndarray:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)
    
    def image_to_base64(self, image: np.ndarray) -> str:
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def deblur_image(self, 
                    image: np.ndarray,
                    model_type: ModelType,
                    num_inference_steps: int = 50,
                    guidance_scale: float = 1.0) -> np.ndarray:
        
        if model_type.value not in self.models:
            raise ValueError(f"Model {model_type.value} is not loaded")
        
        model = self.models[model_type.value]
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            if model_type == ModelType.DIFFUSION:
                if hasattr(model, 'inference'):
                    output_tensor = model.inference(
                        input_tensor, 
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    )
                else:
                    output_tensor = model(input_tensor, input_tensor)[0]
            else:
                output_tensor = model(input_tensor)
        
        return self.postprocess_image(output_tensor)
    
    def process_single_image(self, 
                           image_data: str,
                           model_type: ModelType,
                           num_inference_steps: int = 50,
                           guidance_scale: float = 1.0) -> DeblurResponse:
        
        start_time = time.time()
        
        try:
            image = self.base64_to_image(image_data)
            
            deblurred_image = self.deblur_image(
                image, model_type, num_inference_steps, guidance_scale
            )
            
            result_base64 = self.image_to_base64(deblurred_image)
            
            processing_time = time.time() - start_time
            
            return DeblurResponse(
                success=True,
                message="Image deblurred successfully",
                deblurred_image=result_base64,
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return DeblurResponse(
                success=False,
                message=f"Error processing image: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def process_batch_images(self,
                           image_paths: List[str],
                           output_dir: str,
                           model_type: ModelType,
                           num_inference_steps: int = 50) -> BatchDeblurResponse:
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        output_paths = []
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    failed_count += 1
                    continue
                
                deblurred_image = self.deblur_image(
                    image, model_type, num_inference_steps
                )
                
                output_file = output_path / f"deblurred_{Path(image_path).name}"
                cv2.imwrite(str(output_file), cv2.cvtColor(deblurred_image, cv2.COLOR_RGB2BGR))
                
                output_paths.append(str(output_file))
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Failed to process {image_path}: {e}")
                failed_count += 1
        
        return BatchDeblurResponse(
            success=processed_count > 0,
            message=f"Processed {processed_count} images, {failed_count} failed",
            processed_count=processed_count,
            failed_count=failed_count,
            output_paths=output_paths
        )
    
    def get_model_info(self) -> List[ModelInfo]:
        info_list = []
        
        for model_type in ModelType:
            info = ModelInfo(
                model_type=model_type,
                model_path=self.model_paths.get(model_type.value, ""),
                is_loaded=model_type.value in self.models,
                device=self.device
            )
            info_list.append(info)
        
        return info_list
    
    def get_memory_usage(self) -> Dict[str, Any]:
        memory_info = {
            "system_memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved()
            }
        
        return memory_info