import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Union, List, Dict, Any
import random
import logging
from .difficulty_config import get_difficulty_config, get_random_value_in_range, get_random_range_in_ranges

class BlurGenerator:
    def __init__(self, random_seed: int = 42, difficulty: str = 'easy'):
        self.random_seed = random_seed
        self.difficulty = difficulty
        self.difficulty_config = get_difficulty_config(difficulty)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def gaussian_blur(self, image: np.ndarray, kernel_size_range: tuple = (5, 15),
                     sigma_range: tuple = (1.0, 3.0),
                     kernel_size: int = None) -> np.ndarray:
        # 兼容性处理：如果提供了单个kernel_size，使用它；否则从范围中随机选择
        if kernel_size is not None:
            selected_kernel_size = kernel_size
        else:
            # 确保kernel_size为奇数且在有效范围内
            min_size = kernel_size_range[0]
            max_size = kernel_size_range[1]

            # 确保最小值为奇数
            if min_size % 2 == 0:
                min_size += 1
            # 确保最大值为奇数
            if max_size % 2 == 0:
                max_size -= 1

            # 确保有效范围
            if min_size > max_size:
                min_size = max_size

            # 生成奇数kernel大小
            if min_size == max_size:
                selected_kernel_size = min_size
            else:
                selected_kernel_size = random.randrange(min_size, max_size + 1, 2)

        # 确保selected_kernel_size是奇数
        if selected_kernel_size % 2 == 0:
            selected_kernel_size += 1

        sigma = random.uniform(sigma_range[0], sigma_range[1])
        return cv2.GaussianBlur(image, (selected_kernel_size, selected_kernel_size), sigma)
    
    def motion_blur(self, image: np.ndarray, kernel_size_range: tuple = (5, 20),
                   angle_range: tuple = (0, 180),
                   kernel_size: int = None) -> np.ndarray:
        # 兼容性处理：如果提供了单个kernel_size，使用它；否则从范围中随机选择
        if kernel_size is not None:
            selected_kernel_size = kernel_size
        else:
            selected_kernel_size = random.randint(kernel_size_range[0], kernel_size_range[1])
        angle = random.randint(angle_range[0], angle_range[1])

        kernel = np.zeros((selected_kernel_size, selected_kernel_size))
        kernel[int((selected_kernel_size-1)/2), :] = np.ones(selected_kernel_size)
        kernel = kernel / selected_kernel_size

        angle_rad = np.deg2rad(angle)
        rotation_matrix = cv2.getRotationMatrix2D((selected_kernel_size//2, selected_kernel_size//2), angle, 1)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (selected_kernel_size, selected_kernel_size))
        
        return cv2.filter2D(image, -1, kernel)
    
    def compression_artifacts(self, image: np.ndarray, quality_range: tuple = (30, 70)) -> np.ndarray:
        quality = random.randint(quality_range[0], quality_range[1])
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return decoded_img
    
    def print_scan_simulation(self, image: np.ndarray,
                             enable_geometric_distortion: bool = False,
                             noise_intensity: float = 0.3,
                             brightness_contrast_intensity: float = 0.1) -> np.ndarray:
        """
        打印扫描模拟效果 - 可配置强度参数

        Args:
            noise_intensity: 噪声强度 (0-1)
            brightness_contrast_intensity: 亮度对比度变化强度 (0-1)
        """
        transforms = []

        # 可选的几何变形（旋转、平移、缩放）
        if enable_geometric_distortion:
            transforms.append(A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-2, 2),
                p=0.8
            ))

        # 可配置强度的效果
        # 噪声强度: 根据difficulty调整
        noise_var_max = 10.0 + 40.0 * noise_intensity  # 10-50的范围
        transforms.append(A.GaussNoise(
            var_limit=(5.0, noise_var_max),
            mean=0,
            p=0.7
        ))

        # 亮度对比度变化: 根据difficulty调整
        bc_limit = 0.05 + 0.15 * brightness_contrast_intensity  # 0.05-0.2的范围
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=bc_limit,
            contrast_limit=bc_limit,
            p=0.8
        ))

        transform = A.Compose(transforms)
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
            blur_types = [
                'gaussian', 'motion', 'compression', 'scan', 'lowres',
                'text', 'lines',
                'print_scan', 'localblur', 'threshold', 'print_noise',
                'scan_lines', 'composite',
                'line_discontinuity', 'regional_thinning', 'spectral_degradation'
            ]

        blur_type = random.choice(blur_types)
        result_image = image.copy()

        if blur_type == 'gaussian':
            result_image = self.gaussian_blur(result_image)
        elif blur_type == 'motion':
            result_image = self.motion_blur(result_image)
        elif blur_type == 'compression':
            result_image = self.compression_artifacts(result_image)
        elif blur_type == 'scan':
            result_image = self.print_scan_simulation(result_image, enable_geometric_distortion=False)
        elif blur_type == 'lowres':
            result_image = self.low_resolution_upscale(result_image)
        elif blur_type == 'text':
            result_image = self.add_text_interference(result_image)
        elif blur_type == 'lines':
            result_image = self.add_line_interference(result_image)
        elif blur_type == 'print_scan':
            result_image = self.print_scan_blur(result_image)
        elif blur_type == 'localblur':
            result_image = self.local_blur(result_image)
        elif blur_type == 'threshold':
            result_image = self.threshold_artifacts(result_image)
        elif blur_type == 'print_noise':
            result_image = self.add_print_noise(result_image)
        elif blur_type == 'scan_lines':
            result_image = self.add_scan_lines(result_image)
        elif blur_type == 'line_discontinuity':
            result_image = self.line_discontinuity_blur(result_image)
        elif blur_type == 'regional_thinning':
            result_image = self.regional_line_thinning(result_image)
        elif blur_type == 'spectral_degradation':
            result_image = self.spectral_line_degradation(result_image)
        elif blur_type == 'composite':
            return self.apply_composite_blur(result_image, num_ops=random.randint(2, 3))

        return {
            'image': result_image,
            'blur_type': blur_type
        }

    def print_scan_blur(self, image: np.ndarray,
                        edge_blur_sigma: float = 1.2,
                        contrast_reduction: float = 0.9) -> np.ndarray:
        """模拟真实的打印扫描效果：线条变细+边缘模糊+对比度降低"""

        # 1. 强烈的高斯模糊，模拟打印的羽化效果
        heavily_blurred = cv2.GaussianBlur(image, (0, 0), edge_blur_sigma)

        # 2. 轻度模糊，保持一些线条结构
        lightly_blurred = cv2.GaussianBlur(image, (0, 0), edge_blur_sigma * 0.3)

        # 3. 混合得到羽化但还能看到的线条 - 降低重度模糊比例
        result = cv2.addWeighted(lightly_blurred, 0.5, heavily_blurred, 0.5, 0)

        # 4. 降低对比度，模拟真实打印的灰度效果
        result = result.astype(np.float32)
        # 将黑色(0)变成深灰色，白色(255)保持
        result = result * contrast_reduction + (255 * (1 - contrast_reduction))

        # 5. 添加细微的背景噪声，模拟纸张纹理
        paper_texture = np.random.normal(0, 8, result.shape)
        result += paper_texture

        return np.clip(result, 0, 255).astype(np.uint8)

    def local_blur(self, image: np.ndarray,
                   num_patches_range: tuple = (3, 8),
                   patch_size_range: tuple = (20, 60)) -> np.ndarray:
        """Apply blur to random local patches to simulate partial degradation"""
        h, w = image.shape[:2]
        result = image.copy()

        num_patches = random.randint(num_patches_range[0], num_patches_range[1])

        for _ in range(num_patches):
            # Random patch location and size
            patch_size = random.randint(patch_size_range[0], patch_size_range[1])
            x = random.randint(0, max(1, w - patch_size))
            y = random.randint(0, max(1, h - patch_size))

            # Extract patch
            patch = result[y:y+patch_size, x:x+patch_size].copy()

            # Apply strong blur to patch
            kernel_size = random.randrange(7, 15, 2)
            sigma = random.uniform(2.0, 4.0)
            blurred_patch = cv2.GaussianBlur(patch, (kernel_size, kernel_size), sigma)

            # Put back the blurred patch
            result[y:y+patch_size, x:x+patch_size] = blurred_patch

        return result

    def threshold_artifacts(self, image: np.ndarray,
                           threshold_range: tuple = (80, 120)) -> np.ndarray:
        """Apply threshold artifacts to simulate binary conversion effects"""
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False

        # Apply random threshold
        threshold_val = random.randint(threshold_range[0], threshold_range[1])
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

        # Convert back to original format
        if is_color:
            # Convert back to BGR
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            result = binary

        return result

    def add_print_noise(self, image: np.ndarray,
                       noise_intensity: float = 0.15,
                       intensity: float = None) -> np.ndarray:
        """添加真实的打印纸张噪点：灰色斑点、纸张纹理"""
        # 兼容性处理：如果使用了intensity参数，则覆盖noise_intensity
        if intensity is not None:
            noise_intensity = intensity

        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]

        # 1. 生成纸张纹理噪声（主要是灰色值）
        paper_noise = np.random.normal(0, noise_intensity * 20, (h, w))

        # 2. 生成细微的打印斑点（不是纯黑白，而是灰色）
        spot_mask = np.random.random((h, w)) < (noise_intensity * 0.02)  # 2%的像素有斑点
        spot_values = np.random.uniform(180, 220, np.sum(spot_mask))  # 灰色斑点，不是纯白

        # 3. 应用纸张纹理
        if len(result.shape) == 3:
            for c in range(3):
                result[:, :, c] += paper_noise
        else:
            result += paper_noise

        # 4. 添加灰色斑点
        if len(result.shape) == 3:
            # 对于彩色图像，为每个通道设置相同的灰色值
            result[spot_mask, 0] = spot_values
            result[spot_mask, 1] = spot_values
            result[spot_mask, 2] = spot_values
        else:
            result[spot_mask] = spot_values

        # 5. 确保值在合理范围内
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def add_scan_lines(self, image: np.ndarray,
                      line_intensity: float = 0.1,
                      line_spacing: int = 3) -> np.ndarray:
        """添加扫描条纹效果"""
        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]

        # 创建水平扫描线效果
        for y in range(0, h, line_spacing):
            # 随机强度的扫描线
            line_strength = np.random.uniform(0.5, 1.0) * line_intensity * 30
            if len(result.shape) == 3:
                result[y, :] = result[y, :] * (1 - line_strength) + 240 * line_strength
            else:
                result[y, :] = result[y, :] * (1 - line_strength) + 240 * line_strength

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_composite_blur(self, image: np.ndarray, num_ops: int = 2) -> Dict[str, Any]:
        """Apply multiple blur operations in sequence for more realistic degradation"""
        available_ops = [
            'gaussian', 'motion', 'compression', 'print_scan',
            'localblur', 'threshold', 'print_noise', 'scan_lines'
        ]

        # Randomly select operations
        selected_ops = random.sample(available_ops, min(num_ops, len(available_ops)))

        result_image = image.copy()
        applied_operations = []

        for op in selected_ops:
            if op == 'gaussian':
                result_image = self.gaussian_blur(result_image)
            elif op == 'motion':
                result_image = self.motion_blur(result_image)
            elif op == 'compression':
                result_image = self.compression_artifacts(result_image)
            elif op == 'morphology':
                result_image = self.random_morphology(result_image)
            elif op == 'localblur':
                result_image = self.local_blur(result_image)
            elif op == 'threshold':
                result_image = self.threshold_artifacts(result_image)
            elif op == 'saltpepper':
                result_image = self.add_salt_pepper(result_image)

            applied_operations.append(op)

        return {
            'image': result_image,
            'blur_type': f"composite_{'_'.join(applied_operations)}"
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

    def generate_blur(self, input_path: Union[str, Path], output_path: Union[str, Path],
                     blur_type: str = None) -> None:
        """Generate a single blurred image"""
        # Load the image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")

        # Apply blur effect
        if blur_type:
            result = self.apply_random_blur(image, blur_types=[blur_type])
        else:
            result = self.apply_random_blur(image)

        # Save result
        cv2.imwrite(str(output_path), result['image'])

    def generate_blur_with_difficulty(self, input_path: Union[str, Path],
                                    output_path: Union[str, Path],
                                    blur_type: str, difficulty_config: dict) -> None:
        """根据难度配置生成模糊图像"""
        # Load the image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")

        # 根据模糊类型和难度配置应用效果
        if blur_type == 'print_scan':
            result = self.print_scan_blur_with_config(image, difficulty_config)
        elif blur_type == 'print_noise':
            result = self.add_print_noise_with_config(image, difficulty_config)
        elif blur_type == 'scan_lines':
            result = self.add_scan_lines_with_config(image, difficulty_config)
        else:
            # 默认使用原有方法
            blur_result = self.apply_random_blur(image, blur_types=[blur_type])
            result = blur_result['image']

        # Save result
        cv2.imwrite(str(output_path), result)

    def print_scan_blur_with_config(self, image, difficulty_config):
        """基于难度配置的打印扫描模糊效果"""
        blur_strength = difficulty_config['blur_strength']
        contrast_reduction = difficulty_config['contrast_reduction']
        print ("blur_strength", blur_strength, "contrast_reduction", contrast_reduction)

        # 根据难度调整模糊强度
        edge_blur_sigma = 1.5 * blur_strength

        # 线条细化处理（更高难度->更细的线）
        kernel_size = max(1, int(3 - blur_strength))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)

        # 多层模糊叠加
        heavily_blurred = cv2.GaussianBlur(image, (0, 0), edge_blur_sigma)
        lightly_blurred = cv2.GaussianBlur(image, (0, 0), edge_blur_sigma * 0.3)

        # 混合模糊效果
        result = cv2.addWeighted(lightly_blurred, 0.4, heavily_blurred, 0.6, 0)

        # 对比度降低
        result = result * contrast_reduction + (255 * (1 - contrast_reduction))

        # 添加轻微的纸张纹理
        noise_intensity = 0.02 * blur_strength
        noise = np.random.normal(0, noise_intensity * 255, image.shape).astype(np.float32)
        result = np.clip(result + noise, 0, 255)

        return result.astype(np.uint8)

    def add_print_noise_with_config(self, image, difficulty_config):
        """基于难度配置的打印噪点效果"""
        blur_strength = difficulty_config['blur_strength']
        contrast_reduction = difficulty_config['contrast_reduction']

        # 首先应用细化处理
        kernel_size = max(1, int(3 - blur_strength))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)

        result = image.copy().astype(np.float32)

        # 根据难度调整噪点密度和强度
        noise_density = 0.001 * (blur_strength ** 2)  # 难度越高噪点越多
        noise_intensity = 30 * blur_strength  # 难度越高噪点越明显

        # 生成噪点位置
        h, w = result.shape[:2]
        num_spots = int(h * w * noise_density)

        if num_spots > 0:
            spot_coords = np.random.randint(0, [h, w], size=(num_spots, 2))

            # 生成灰色噪点（不是纯黑白）
            spot_values = np.random.uniform(80, 180, num_spots).astype(np.float32)  # 灰色范围

            # 创建掩码
            spot_mask = (spot_coords[:, 0], spot_coords[:, 1])

            # 应用噪点到所有颜色通道
            if len(result.shape) == 3:
                result[spot_mask[0], spot_mask[1], 0] = spot_values
                result[spot_mask[0], spot_mask[1], 1] = spot_values
                result[spot_mask[0], spot_mask[1], 2] = spot_values
            else:
                result[spot_mask] = spot_values

        # 应用对比度降低
        result = result * contrast_reduction + (255 * (1 - contrast_reduction))

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_scan_lines_with_config(self, image, difficulty_config):
        """基于难度配置的扫描线条效果"""
        blur_strength = difficulty_config['blur_strength']
        contrast_reduction = difficulty_config['contrast_reduction']

        # 首先应用细化处理
        kernel_size = max(1, int(3 - blur_strength))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)

        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]

        # 根据难度调整扫描线密度和强度
        line_spacing = max(2, int(8 - blur_strength * 2))  # 难度越高线条越密
        line_opacity = 0.1 + 0.1 * blur_strength  # 难度越高线条越明显

        # 添加水平扫描线
        for y in range(0, h, line_spacing):
            if len(result.shape) == 3:
                result[y, :, :] *= (1 - line_opacity)
            else:
                result[y, :] *= (1 - line_opacity)

        # 添加轻微的垂直扫描线（密度更低）
        vertical_spacing = line_spacing * 3
        for x in range(0, w, vertical_spacing):
            if len(result.shape) == 3:
                result[:, x, :] *= (1 - line_opacity * 0.5)
            else:
                result[:, x] *= (1 - line_opacity * 0.5)

        # 应用对比度降低
        result = result * contrast_reduction + (255 * (1 - contrast_reduction))

        # 添加轻微的整体模糊
        result = cv2.GaussianBlur(result.astype(np.uint8), (3, 3), 0.8 * blur_strength)

        return result.astype(np.uint8)

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load an image from file"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image

    def line_discontinuity_blur(self, image: np.ndarray,
                               gap_density: float = 0.4,
                               gap_size_range: tuple = (2, 5)) -> np.ndarray:
        """
        Create dashed-line effects with many small gaps (not too disconnected)
        创建虚线效果 - 小间隙但数量多，像虚线一样
        """
        result = image.copy()
        h, w = result.shape[:2]

        # Convert to grayscale for line detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. 基于Hough变换的线段断续处理
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15,
                               minLineLength=8, maxLineGap=3)

        gaps_created = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                # Process lines to create dashed-line effect
                if line_length > 10:  # Process shorter lines too
                    # 创建虚线效果 - 更多但更小的间隙
                    num_gaps = max(3, int(line_length * gap_density / 6))  # More gaps

                    for _ in range(num_gaps):
                        # Random gap position along the line
                        t = random.uniform(0.1, 0.9)
                        gap_center_x = int(x1 + t * (x2 - x1))
                        gap_center_y = int(y1 + t * (y2 - y1))

                        # 小间隙尺寸 - 像虚线的短划
                        gap_size = random.randint(gap_size_range[0], gap_size_range[1])

                        # Create small circular gaps for dashed effect
                        cv2.circle(result, (gap_center_x, gap_center_y), gap_size//2,
                                 (255, 255, 255) if len(result.shape) == 3 else 255, -1)
                        gaps_created += 1

        # 2. 基于像素密度的智能间隙生成
        # 找到所有线条像素
        line_mask = gray < 180  # 更宽松的阈值以捕获更多线条
        line_coords = np.where(line_mask)

        if len(line_coords[0]) > 0:
            # 计算虚线效果的小间隙
            num_pixel_gaps = int(len(line_coords[0]) * gap_density * 0.008)  # 更多小间隙

            if num_pixel_gaps > 0:
                # 随机选择线条像素位置创建小间隙
                indices = random.sample(range(len(line_coords[0])),
                                      min(num_pixel_gaps, len(line_coords[0])))

                for idx in indices:
                    gap_y, gap_x = line_coords[0][idx], line_coords[1][idx]
                    gap_size = random.randint(gap_size_range[0], gap_size_range[1])

                    # 只创建小圆形间隙，保持虚线效果简单一致
                    cv2.circle(result, (gap_x, gap_y), gap_size//2,
                             (255, 255, 255) if len(result.shape) == 3 else 255, -1)
                    gaps_created += 1

        # 3. 轻微的线条边缘处理，保持虚线效果自然
        # 减少边缘腐蚀强度，避免过度断开
        if len(result.shape) == 3:
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            result_gray = result.copy()

        # 非常轻微的腐蚀，只是稍微软化边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        eroded = cv2.erode(result_gray, kernel, iterations=1)

        # 非常轻微混合，保持绝大部分原图结构
        if len(result.shape) == 3:
            eroded_3ch = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(result, 0.95, eroded_3ch, 0.05, 0)  # 极轻的混合
        else:
            result = cv2.addWeighted(result, 0.95, eroded, 0.05, 0)

        # print(f"  虚线效果处理: 创建了 {gaps_created} 个小间隙")
        return result

    def regional_line_variations(self, image: np.ndarray,
                                   thinning_strength: float = 0.3,
                                   fading_strength: float = 0.3,
                                   num_regions: int = 2) -> np.ndarray:
        """
        Regional line thinning and fading effects
        区域性线条变细和变淡效果 - 只影响少数区域，保持主体线条正常

        Args:
            thinning_strength: 线条变细强度 (0-1)
            fading_strength: 线条变淡强度 (0-1)
            num_regions: 影响的区域数量
        """
        result = image.copy()
        h, w = result.shape[:2]

        # 如果区域数量为0，直接返回原图
        if num_regions == 0:
            return result

        for i in range(num_regions):
            # 创建较小的随机区域 - 只影响图像的一小部分
            region_w = random.randint(60, 150)  # 较小的区域
            region_h = random.randint(60, 150)
            region_x = random.randint(0, max(1, w - region_w))
            region_y = random.randint(0, max(1, h - region_h))

            # 提取区域
            region = result[region_y:region_y+region_h, region_x:region_x+region_w].copy()

            # 随机选择应用变细或变淡（不是都应用）
            effect_type = random.choice(['thinning', 'fading'])

            if effect_type == 'thinning' and thinning_strength > 0:
                # 轻微的线条变细
                kernel_size = max(1, min(3, int(1 + 2 * thinning_strength)))  # 1-3像素
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                thinned_region = cv2.erode(region, kernel, iterations=1)  # 只1次迭代

                # 轻微混合，保持大部分原始线条
                alpha = 0.3 + 0.3 * thinning_strength  # 最多0.6的混合比例
                processed_region = cv2.addWeighted(region, 1-alpha, thinned_region, alpha, 0)

            elif effect_type == 'fading' and fading_strength > 0:
                # 轻微的线条变淡
                processed_region = region.copy().astype(np.float32)

                # 找到线条区域
                if len(region.shape) == 3:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                else:
                    gray = region.copy()

                line_mask = gray < 200

                if np.any(line_mask):
                    # 轻微的变淡效果
                    fade_amount = int(20 + 30 * fading_strength)  # 20-50的轻微变淡

                    if len(processed_region.shape) == 3:
                        for c in range(3):
                            processed_region[:, :, c][line_mask] += fade_amount
                    else:
                        processed_region[line_mask] += fade_amount

                processed_region = np.clip(processed_region, 0, 255).astype(np.uint8)
            else:
                processed_region = region

            # 将处理后的区域放回原图
            result[region_y:region_y+region_h, region_x:region_x+region_w] = processed_region

        return result

    def apply_single_blur_effect(self, image: np.ndarray, effect_type: str,
                               intensity: float = 0.5) -> np.ndarray:
        """
        应用单个模糊效果 - 用于演示目的
        Args:
            intensity: 效果强度 (0-1)，用于区分easy/medium/hard
        """
        result = image.copy()

        if effect_type == 'gaussian':
            # 根据强度调整高斯模糊参数 - 降低强度
            kernel_size = max(3, int(3 + 4 * intensity))  # 3-7 instead of 3-9
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = 0.3 + 1.2 * intensity  # 0.3-1.5 instead of 0.5-2.5
            result = self.gaussian_blur(result, kernel_size=kernel_size, sigma_range=(sigma, sigma))
        elif effect_type == 'motion':
            # 根据强度调整运动模糊 - 降低强度
            kernel_size = max(3, int(3 + 5 * intensity))  # 3-8 instead of 3-11
            result = self.motion_blur(result, kernel_size=kernel_size)
        elif effect_type == 'compression':
            # 根据强度调整压缩质量 - 提高最低质量
            quality = int(70 - 40 * intensity)  # 70->30 instead of 80->20
            result = self.compression_blur(result, quality=quality)
        elif effect_type == 'scan':
            # 基础扫描效果 - 更简单的噪声
            result = self.print_scan_simulation(result,
                                              noise_intensity=0.2 * intensity,
                                              brightness_contrast_intensity=0.1 * intensity)
        elif effect_type == 'lowres':
            # 根据强度调整下采样因子 - 降低强度
            factor = int(2 + 2 * intensity)  # 2->4 instead of 2->6
            result = self.low_resolution_blur(result, downscale_factor=factor)
        elif effect_type == 'text':
            result = self.add_text_interference(result)
        elif effect_type == 'lines':
            result = self.add_line_interference(result)
        elif effect_type == 'print_scan':
            # 高级打印扫描效果 - 更强的噪声和对比度变化
            result = self.print_scan_simulation(result,
                                              noise_intensity=0.5 + 0.5 * intensity,
                                              brightness_contrast_intensity=0.2 + 0.3 * intensity)
        elif effect_type == 'localblur':
            result = self.local_blur_degradation(result)
        elif effect_type == 'scan_lines':
            result = self.add_scan_lines(result)
        elif effect_type == 'spectral_degradation':
            # 使用配置化参数 - 降低强度
            strength = 0.1 + 0.4 * intensity  # 0.1-0.5 instead of 0.2-0.8
            range_pct = 0.2 + 0.3 * intensity  # 0.2-0.5 instead of 0.3-0.6
            result = self.spectral_line_degradation(result,
                                                  degradation_strength=strength,
                                                  range_percentage=range_pct)
        else:
            print(f"未知效果类型: {effect_type}")

        return result

    def spectral_line_degradation(self, image: np.ndarray,
                                x_range: tuple = None,
                                degradation_strength: float = 0.3,
                                range_percentage: float = 0.4,
                                degradation_type: str = 'both') -> np.ndarray:
        """
        Apply configurable degradation effects to spectral line ranges
        专门针对光谱线的特定x轴范围进行可配置强度的退化处理

        Args:
            degradation_strength: 退化强度 (0-1)
            range_percentage: 影响范围百分比 (0-1)
        """
        if x_range is None:
            # Use configurable range percentage
            w = image.shape[1]
            range_width = int(w * range_percentage)
            x_start = random.randint(int(w * 0.1), int(w * 0.6))
            x_range = (x_start, min(x_start + range_width, w))

        result = image.copy()
        x_start, x_end = x_range

        # Extract the region of interest
        roi = result[:, x_start:x_end].copy()

        # 1. 可配置线条细化
        if degradation_type in ['thinning', 'both']:
            # 根据强度调整kernel大小和迭代次数
            kernel_size = max(1, int(1 + 2 * degradation_strength))  # 1-3
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            iterations = max(1, int(degradation_strength * 2))  # 0-2次迭代
            if iterations > 0:
                roi = cv2.erode(roi, kernel, iterations=iterations)

            # 可配置模糊强度
            blur_sigma = 0.3 + 1.2 * degradation_strength  # 0.3-1.5
            roi = cv2.GaussianBlur(roi, (3, 3), blur_sigma)

        # 2. 可配置断续效果
        if degradation_type in ['discontinuity', 'both']:
            gap_density = 0.05 + 0.2 * degradation_strength  # 0.05-0.25
            gap_size_max = int(2 + 6 * degradation_strength)  # 2-8
            roi = self.line_discontinuity_blur(roi, gap_density=gap_density,
                                             gap_size_range=(1, gap_size_max))

        # 3. 可配置运动模糊
        motion_kernel_size = max(3, int(3 + 4 * degradation_strength))  # 3-7
        motion_kernel = np.zeros((1, motion_kernel_size))
        motion_kernel[0, :] = 1/motion_kernel_size
        roi = cv2.filter2D(roi, -1, motion_kernel)

        # 4. 可配置对比度降低
        contrast_factor = 1.0 - 0.3 * degradation_strength  # 1.0-0.7
        if len(roi.shape) == 3:
            roi = roi.astype(np.float32)
            roi = roi * contrast_factor + 255 * (1 - contrast_factor)
            roi = np.clip(roi, 0, 255).astype(np.uint8)

        # Put back the processed region
        result[:, x_start:x_end] = roi

        return result

    def background_color_variation(self, image: np.ndarray,
                                 variation_type: str = 'random',
                                 intensity: float = 0.3) -> np.ndarray:
        """
        Apply background color variation to simulate different lighting/printing conditions
        应用背景颜色变化 - 模拟不同的光线或打印条件

        Args:
            variation_type: 'global' (整体变化), 'local' (局部变化), 'random' (随机选择)
            intensity: 变化强度 0.0-1.0
        """
        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]

        if variation_type == 'random':
            variation_type = random.choice(['global', 'local', 'gradient'])

        if variation_type == 'global':
            # 整体颜色偏移 - 模拟整体光线变化
            if len(result.shape) == 3:
                # 生成整体色偏
                color_shift = np.array([
                    random.uniform(-30*intensity, 30*intensity),  # R
                    random.uniform(-25*intensity, 25*intensity),  # G
                    random.uniform(-35*intensity, 35*intensity)   # B
                ])

                # 应用到背景区域（亮像素）
                for c in range(3):
                    channel = result[:, :, c]
                    bg_mask = channel > 200  # 背景像素
                    channel[bg_mask] += color_shift[c]

            else:
                # 灰度图的亮度变化
                brightness_shift = random.uniform(-20*intensity, 20*intensity)
                bg_mask = result > 200
                result[bg_mask] += brightness_shift

        elif variation_type == 'local':
            # 局部颜色变化 - 模拟扫描时的不均匀光照
            num_patches = random.randint(3, 8)

            for _ in range(num_patches):
                # 随机补丁位置和大小
                patch_w = random.randint(int(w*0.2), int(w*0.6))
                patch_h = random.randint(int(h*0.2), int(h*0.6))
                patch_x = random.randint(0, max(1, w - patch_w))
                patch_y = random.randint(0, max(1, h - patch_h))

                # 创建渐变遮罩让变化更自然
                mask = np.zeros((patch_h, patch_w), dtype=np.float32)
                center_x, center_y = patch_w//2, patch_h//2
                max_dist = min(patch_w, patch_h) // 2

                for y in range(patch_h):
                    for x in range(patch_w):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        mask[y, x] = max(0, 1 - dist/max_dist)

                if len(result.shape) == 3:
                    # 彩色图的局部色偏
                    local_color_shift = np.array([
                        random.uniform(-25*intensity, 25*intensity),
                        random.uniform(-20*intensity, 20*intensity),
                        random.uniform(-30*intensity, 30*intensity)
                    ])

                    for c in range(3):
                        patch = result[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w, c]
                        bg_mask = patch > 180  # 背景区域
                        patch[bg_mask] += mask[bg_mask] * local_color_shift[c]
                else:
                    # 灰度图的局部亮度变化
                    brightness_shift = random.uniform(-15*intensity, 15*intensity)
                    patch = result[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
                    bg_mask = patch > 180
                    patch[bg_mask] += mask[bg_mask] * brightness_shift

        elif variation_type == 'gradient':
            # 渐变变化 - 模拟扫描仪光源不均
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])

            if direction == 'horizontal':
                gradient = np.linspace(-20*intensity, 20*intensity, w)
                gradient_map = np.tile(gradient, (h, 1))
            elif direction == 'vertical':
                gradient = np.linspace(-20*intensity, 20*intensity, h)
                gradient_map = np.tile(gradient.reshape(-1, 1), (1, w))
            else:  # diagonal
                x_grad = np.linspace(-15*intensity, 15*intensity, w)
                y_grad = np.linspace(-15*intensity, 15*intensity, h)
                gradient_map = np.add.outer(y_grad, x_grad) / 2

            if len(result.shape) == 3:
                # 随机选择主要影响的颜色通道
                primary_channel = random.randint(0, 2)
                for c in range(3):
                    channel = result[:, :, c]
                    bg_mask = channel > 180
                    if c == primary_channel:
                        channel[bg_mask] += gradient_map[bg_mask]
                    else:
                        channel[bg_mask] += gradient_map[bg_mask] * 0.3
            else:
                bg_mask = result > 180
                result[bg_mask] += gradient_map[bg_mask]

        # 确保值在有效范围内
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def apply_base_degradation(self, image: np.ndarray) -> tuple:
        """
        Apply base degradation effects that should be present in every image:
        - Background color variation (lighting/printing conditions)
        - Line thickness inconsistency (regional_thinning with color variation)
        - Line discontinuity (dashed line effect)
        - Print noise artifacts
        对每张图片都应用的基础退化效果组合 - 使用配置化的强度范围

        Returns:
            tuple: (degraded_image, effects_log)
        """
        result = image.copy()
        config = self.difficulty_config
        applied_effects = []

        try:
            # 1. 底色变化 (模拟光线/打印条件差异) - 使用配置化强度
            bg_intensity = get_random_value_in_range(config['background_variation']['intensity'])
            effect_log = f"background_variation(intensity={bg_intensity:.3f})"
            applied_effects.append(effect_log)
            result = self.background_color_variation(result, intensity=bg_intensity)

            # 2. 线条变细和变淡 - 区域性轻微变化
            line_config = config['line_thinning_fading']
            thinning_strength = get_random_value_in_range(line_config['thinning_strength'])
            fading_strength = get_random_value_in_range(line_config['fading_strength'])
            num_regions = get_random_value_in_range(line_config['num_regions'], is_int=True)
            effect_log = f"line_variations(thin={thinning_strength:.3f}, fade={fading_strength:.3f}, regions={num_regions})"
            applied_effects.append(effect_log)
            result = self.regional_line_variations(result,
                                                 thinning_strength=thinning_strength,
                                                 fading_strength=fading_strength,
                                                 num_regions=num_regions)

            # 3. 线段断断续续 - 使用配置化参数
            discontinuity_config = config['line_discontinuity']
            gap_density = get_random_value_in_range(discontinuity_config['gap_density'])
            gap_size_range = discontinuity_config['gap_size_range'][0]  # 使用第一个范围

            # 确保gap_size_range有效
            if gap_size_range[0] > gap_size_range[1]:
                gap_size_range = (gap_size_range[1], gap_size_range[0])

            effect_log = f"line_discontinuity(density={gap_density:.3f}, gap_range={gap_size_range})"
            applied_effects.append(effect_log)
            result = self.line_discontinuity_blur(result,
                                                gap_density=gap_density,
                                                gap_size_range=gap_size_range)

            # 4. 打印噪点效果 - 使用配置化参数
            noise_intensity = get_random_value_in_range(config['print_noise']['noise_intensity'])
            effect_log = f"print_noise(intensity={noise_intensity:.3f})"
            applied_effects.append(effect_log)
            result = self.add_print_noise(result, noise_intensity=noise_intensity)

        except Exception as e:
            print(f"🚨 BASE DEGRADATION ERROR:")
            print(f"   Difficulty: {self.difficulty}")
            print(f"   Applied effects so far: {applied_effects}")
            print(f"   Error: {str(e)}")
            print(f"   Config: {config}")
            raise e

        return result, applied_effects

    def apply_random_additional_blur(self, image: np.ndarray, num_effects: int = None) -> Dict[str, Any]:
        """
        Apply random additional blur effects on top of base degradation using difficulty-based parameters
        在基础退化的基础上随机添加额外的模糊效果（使用配置化的难度参数）
        """
        config = self.difficulty_config
        if num_effects is None:
            num_effects = get_random_value_in_range(config['additional_effects_count'], is_int=True)

        # 可选的额外模糊效果（不包括基础必加效果）
        additional_effects = [
            'gaussian', 'motion', 'compression', 'scan', 'lowres',
            'text', 'lines', 'print_scan', 'localblur',  # 暂时移除threshold
            'scan_lines', 'spectral_degradation'
        ]

        # 随机选择effects
        selected_effects = random.sample(additional_effects,
                                       min(num_effects, len(additional_effects)))

        result = image.copy()
        applied_effects = []
        effect_details = []

        print(f"🔧 ADDITIONAL BLUR: {self.difficulty} difficulty, applying {num_effects} effects: {selected_effects}")

        for effect in selected_effects:
            try:
                if effect == 'gaussian':
                    # 使用配置化的高斯模糊参数
                    gaussian_config = config['gaussian_blur']
                    kernel_range = get_random_range_in_ranges(gaussian_config['kernel_size_range'], is_int=True)
                    sigma_range = get_random_range_in_ranges(gaussian_config['sigma_range'])
                    effect_details.append(f"gaussian(kernel={kernel_range}, sigma={sigma_range})")
                    result = self.gaussian_blur(result, kernel_size_range=kernel_range, sigma_range=sigma_range)
                elif effect == 'motion':
                    # 使用配置化的运动模糊参数
                    motion_config = config['motion_blur']
                    kernel_range = get_random_range_in_ranges(motion_config['kernel_size_range'], is_int=True)
                    effect_details.append(f"motion(kernel={kernel_range})")
                    result = self.motion_blur(result, kernel_size_range=kernel_range)
                elif effect == 'compression':
                    # 使用配置化的压缩参数
                    comp_config = config['compression']
                    quality_range = get_random_range_in_ranges(comp_config['quality_range'], is_int=True)
                    effect_details.append(f"compression(quality={quality_range})")
                    result = self.compression_artifacts(result, quality_range=quality_range)
                elif effect == 'scan':
                    effect_details.append("print_scan_simulation")
                    result = self.print_scan_simulation(result, enable_geometric_distortion=False)
                elif effect == 'lowres':
                    # 使用配置化的低分辨率参数
                    lowres_config = config['lowres']
                    factor_range = get_random_range_in_ranges(lowres_config['downscale_factor_range'], is_int=True)
                    effect_details.append(f"lowres(factor={factor_range})")
                    result = self.low_resolution_upscale(result, downscale_factor_range=factor_range)
                elif effect == 'text':
                    # 使用配置化的文本干扰参数 (固定范围)
                    effect_details.append("text_interference(1-3)")
                    result = self.add_text_interference(result, num_texts_range=(1, 3))
                elif effect == 'lines':
                    # 使用配置化的线条干扰参数 (固定范围)
                    effect_details.append("line_interference(1-3)")
                    result = self.add_line_interference(result, num_lines_range=(1, 3))
                elif effect == 'print_scan':
                    effect_details.append("print_scan_blur")
                    result = self.print_scan_blur(result)
                elif effect == 'localblur':
                    effect_details.append("local_blur")
                    result = self.local_blur(result)
                elif effect == 'threshold':
                    # 使用配置化的阈值参数
                    threshold_config = config['threshold']
                    threshold_range = get_random_range_in_ranges(threshold_config['threshold_range'], is_int=True)
                    effect_details.append(f"threshold(range={threshold_range})")
                    result = self.threshold_artifacts(result, threshold_range=threshold_range)
                elif effect == 'scan_lines':
                    effect_details.append("scan_lines")
                    result = self.add_scan_lines(result)
                elif effect == 'spectral_degradation':
                    # 使用配置化的光谱退化参数
                    spectral_config = config['spectral_degradation']
                    degradation_strength = get_random_value_in_range(spectral_config['degradation_strength'])
                    range_percentage = get_random_value_in_range(spectral_config['range_percentage'])
                    effect_details.append(f"spectral_degradation(strength={degradation_strength:.3f}, range={range_percentage:.3f})")
                    result = self.spectral_line_degradation(result)

                applied_effects.append(effect)
            except Exception as e:
                print(f"🚨 ADDITIONAL BLUR ERROR in {effect}:")
                print(f"   Difficulty: {self.difficulty}")
                print(f"   Applied effects so far: {effect_details}")
                print(f"   Current effect: {effect}")
                print(f"   Error: {str(e)}")
                raise e

        return {
            'image': result,
            'blur_type': f"base_plus_{'_'.join(applied_effects)}",
            'additional_effects': applied_effects,
            'additional_effects_details': effect_details,
            'num_additional': len(applied_effects)
        }

    def simple_line_thinning_and_fading(self, image: np.ndarray,
                                      thinning_strength: float = 0.3,
                                      fading_strength: float = 0.3) -> np.ndarray:
        """
        简单的线条细化和变淡效果

        Args:
            image: 输入图像
            thinning_strength: 细化强度 (0-1)
            fading_strength: 变淡强度 (0-1)

        Returns:
            处理后的图像
        """
        result = image.copy()

        # 检测线条区域 (暗色区域)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result.copy()
        line_mask = gray < 180  # 线条通常是暗色的

        # 1. 线条细化 - 使用形态学腐蚀
        if thinning_strength > 0:
            kernel_size = max(1, int(3 * thinning_strength))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # 仅在线条区域应用腐蚀
            thinned_gray = cv2.erode(gray, kernel, iterations=1)
            # 混合原图和细化结果
            alpha = 1 - thinning_strength
            gray = cv2.addWeighted(gray, alpha, thinned_gray, thinning_strength, 0)

        # 2. 线条变淡 - 增加线条区域的亮度
        if fading_strength > 0:
            fade_amount = int(50 * fading_strength)  # 最多增加50的亮度
            gray[line_mask] = np.clip(gray[line_mask] + fade_amount, 0, 255)

        # 转换回原始色彩空间
        if len(result.shape) == 3:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            result = gray

        return result

    def local_blur_degradation(self, image: np.ndarray,
                              num_regions: int = 3,
                              blur_intensity: float = 0.5) -> np.ndarray:
        """
        局部模糊退化效果 - 在图像的随机区域应用模糊

        Args:
            image: 输入图像
            num_regions: 模糊区域数量
            blur_intensity: 模糊强度 (0-1)

        Returns:
            处理后的图像
        """
        result = image.copy()
        h, w = result.shape[:2]

        for i in range(num_regions):
            # 随机选择区域位置和大小
            region_w = random.randint(w//8, w//3)  # 区域宽度
            region_h = random.randint(h//8, h//3)  # 区域高度
            x = random.randint(0, w - region_w)
            y = random.randint(0, h - region_h)

            # 提取区域
            region = result[y:y+region_h, x:x+region_w].copy()

            # 应用高斯模糊
            kernel_size = max(3, int(7 * blur_intensity))
            if kernel_size % 2 == 0:  # 确保是奇数
                kernel_size += 1
            sigma = 1.0 + 2.0 * blur_intensity
            blurred_region = cv2.GaussianBlur(region, (kernel_size, kernel_size), sigma)

            # 创建渐变混合蒙版，避免硬边缘
            mask = np.zeros((region_h, region_w), dtype=np.float32)
            center_x, center_y = region_w // 2, region_h // 2
            max_dist = min(region_w, region_h) // 2

            for py in range(region_h):
                for px in range(region_w):
                    dist = np.sqrt((px - center_x)**2 + (py - center_y)**2)
                    if dist < max_dist:
                        mask[py, px] = 1.0 - (dist / max_dist)

            # 应用混合蒙版
            if len(result.shape) == 3:
                mask = np.stack([mask] * 3, axis=-1)

            # 混合原始区域和模糊区域
            mixed_region = region * (1 - mask * blur_intensity) + blurred_region * (mask * blur_intensity)
            result[y:y+region_h, x:x+region_w] = mixed_region.astype(np.uint8)

        return result

    def compression_blur(self, image: np.ndarray, quality: int = None, quality_range: tuple = (30, 70)) -> np.ndarray:
        """
        压缩模糊效果的别名方法 - 兼容main.py中的调用

        Args:
            image: 输入图像
            quality: 压缩质量 (1-100，越小压缩越强)
            quality_range: 质量范围，当quality为None时使用

        Returns:
            压缩后的图像
        """
        if quality is not None:
            # 单个质量值
            return self.compression_artifacts(image, quality_range=(quality, quality))
        else:
            # 使用范围
            return self.compression_artifacts(image, quality_range=quality_range)

    def low_resolution_blur(self, image: np.ndarray, downscale_factor: int = None, downscale_factor_range: tuple = (4, 8)) -> np.ndarray:
        """
        低分辨率模糊效果的别名方法 - 兼容main.py中的调用

        Args:
            image: 输入图像
            downscale_factor: 下采样因子
            downscale_factor_range: 下采样因子范围，当downscale_factor为None时使用

        Returns:
            低分辨率处理后的图像
        """
        if downscale_factor is not None:
            # 单个因子值
            return self.low_resolution_upscale(image, downscale_factor_range=(downscale_factor, downscale_factor))
        else:
            # 使用范围
            return self.low_resolution_upscale(image, downscale_factor_range=downscale_factor_range)