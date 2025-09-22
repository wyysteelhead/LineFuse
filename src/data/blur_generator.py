import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Union, List, Dict, Any
import random
import logging

class BlurGenerator:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def gaussian_blur(self, image: np.ndarray, kernel_size_range: tuple = (5, 15), 
                     sigma_range: tuple = (1.0, 3.0)) -> np.ndarray:
        kernel_size = random.randrange(kernel_size_range[0], kernel_size_range[1], 2)
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def motion_blur(self, image: np.ndarray, kernel_size_range: tuple = (5, 20), 
                   angle_range: tuple = (0, 180)) -> np.ndarray:
        kernel_size = random.randint(kernel_size_range[0], kernel_size_range[1])
        angle = random.randint(angle_range[0], angle_range[1])
        
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        angle_rad = np.deg2rad(angle)
        rotation_matrix = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        return cv2.filter2D(image, -1, kernel)
    
    def compression_artifacts(self, image: np.ndarray, quality_range: tuple = (10, 50)) -> np.ndarray:
        quality = random.randint(quality_range[0], quality_range[1])
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return decoded_img
    
    def print_scan_simulation(self, image: np.ndarray) -> np.ndarray:
        transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=2,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=0.8
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.8
            )
        ])
        
        augmented = transform(image=image)
        return augmented['image']
    
    def low_resolution_upscale(self, image: np.ndarray, 
                              downscale_factor_range: tuple = (4, 8)) -> np.ndarray:
        h, w = image.shape[:2]
        factor = random.randint(downscale_factor_range[0], downscale_factor_range[1])
        
        low_res = cv2.resize(image, (w//factor, h//factor), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return upscaled
    
    def add_text_interference(self, image: np.ndarray, 
                            num_texts_range: tuple = (3, 8)) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.copy()
        
        num_texts = random.randint(num_texts_range[0], num_texts_range[1])
        
        for _ in range(num_texts):
            text = random.choice(['A', 'B', 'C', '1', '2', '3', 'X', 'Y'])
            font_scale = random.uniform(0.3, 0.8)
            thickness = random.randint(1, 2)
            
            x = random.randint(50, w-50)
            y = random.randint(50, h-50)
            
            cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (128, 128, 128), thickness)
        
        return result
    
    def add_line_interference(self, image: np.ndarray, 
                            num_lines_range: tuple = (2, 6)) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.copy()
        
        num_lines = random.randint(num_lines_range[0], num_lines_range[1])
        
        for _ in range(num_lines):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            thickness = random.randint(1, 2)
            
            cv2.line(result, pt1, pt2, (200, 200, 200), thickness)
        
        return result
    
    def apply_random_blur(self, image: np.ndarray, blur_types: List[str] = None) -> Dict[str, Any]:
        if blur_types is None:
            blur_types = ['gaussian', 'motion', 'compression', 'scan', 'lowres', 'text', 'lines']
        
        blur_type = random.choice(blur_types)
        result_image = image.copy()
        
        if blur_type == 'gaussian':
            result_image = self.gaussian_blur(result_image)
        elif blur_type == 'motion':
            result_image = self.motion_blur(result_image)
        elif blur_type == 'compression':
            result_image = self.compression_artifacts(result_image)
        elif blur_type == 'scan':
            result_image = self.print_scan_simulation(result_image)
        elif blur_type == 'lowres':
            result_image = self.low_resolution_upscale(result_image)
        elif blur_type == 'text':
            result_image = self.add_text_interference(result_image)
        elif blur_type == 'lines':
            result_image = self.add_line_interference(result_image)
        
        return {
            'image': result_image,
            'blur_type': blur_type
        }
    
    def batch_generate_blur(self, 
                          input_dir: Union[str, Path],
                          output_dir: Union[str, Path],
                          num_variants_per_image: int = 5) -> None:
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                logging.warning(f"Could not load image: {img_file}")
                continue
                
            for i in range(num_variants_per_image):
                result = self.apply_random_blur(image)
                blur_image = result['image']
                blur_type = result['blur_type']
                
                output_file = output_path / f"{img_file.stem}_{blur_type}_{i}.png"
                cv2.imwrite(str(output_file), blur_image)
                
                logging.info(f"Generated blur variant: {output_file}")