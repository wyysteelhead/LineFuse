import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import signal
from typing import Dict, Any, Tuple, List
import logging

class ImageMetrics:
    @staticmethod
    def calculate_psnr(gt_image: np.ndarray, pred_image: np.ndarray) -> float:
        try:
            if gt_image.shape != pred_image.shape:
                pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))
            
            psnr = peak_signal_noise_ratio(gt_image, pred_image, data_range=255)
            return float(psnr)
        except Exception as e:
            logging.error(f"Error calculating PSNR: {e}")
            return 0.0
    
    @staticmethod
    def calculate_ssim(gt_image: np.ndarray, pred_image: np.ndarray) -> float:
        try:
            if gt_image.shape != pred_image.shape:
                pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))
            
            if len(gt_image.shape) == 3:
                ssim = structural_similarity(gt_image, pred_image, multichannel=True, channel_axis=2)
            else:
                ssim = structural_similarity(gt_image, pred_image)
            
            return float(ssim)
        except Exception as e:
            logging.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    @staticmethod
    def calculate_mse(gt_image: np.ndarray, pred_image: np.ndarray) -> float:
        try:
            if gt_image.shape != pred_image.shape:
                pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))
            
            mse = np.mean((gt_image.astype(np.float32) - pred_image.astype(np.float32)) ** 2)
            return float(mse)
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_mae(gt_image: np.ndarray, pred_image: np.ndarray) -> float:
        try:
            if gt_image.shape != pred_image.shape:
                pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))
            
            mae = np.mean(np.abs(gt_image.astype(np.float32) - pred_image.astype(np.float32)))
            return float(mae)
        except Exception as e:
            logging.error(f"Error calculating MAE: {e}")
            return float('inf')

class SpectralMetrics:
    @staticmethod
    def extract_line_profile(image: np.ndarray, axis: int = 1) -> np.ndarray:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        if axis == 1:  # horizontal profile
            profile = np.mean(gray_image, axis=0)
        else:  # vertical profile
            profile = np.mean(gray_image, axis=1)
        
        return profile
    
    @staticmethod
    def find_peaks(profile: np.ndarray, 
                   height: float = None,
                   distance: int = 10,
                   prominence: float = None) -> Tuple[np.ndarray, Dict]:
        if height is None:
            height = np.mean(profile) + np.std(profile)
        
        if prominence is None:
            prominence = np.std(profile) * 0.5
        
        peaks, properties = signal.find_peaks(
            profile,
            height=height,
            distance=distance,
            prominence=prominence
        )
        
        return peaks, properties
    
    @staticmethod
    def calculate_peak_shift(gt_profile: np.ndarray, 
                           pred_profile: np.ndarray,
                           **peak_kwargs) -> Dict[str, Any]:
        try:
            gt_peaks, gt_props = SpectralMetrics.find_peaks(gt_profile, **peak_kwargs)
            pred_peaks, pred_props = SpectralMetrics.find_peaks(pred_profile, **peak_kwargs)
            
            if len(gt_peaks) == 0 or len(pred_peaks) == 0:
                return {
                    'mean_shift': float('inf'),
                    'max_shift': float('inf'),
                    'num_gt_peaks': len(gt_peaks),
                    'num_pred_peaks': len(pred_peaks)
                }
            
            shifts = []
            for gt_peak in gt_peaks:
                closest_pred_peak = pred_peaks[np.argmin(np.abs(pred_peaks - gt_peak))]
                shift = abs(gt_peak - closest_pred_peak)
                shifts.append(shift)
            
            return {
                'mean_shift': float(np.mean(shifts)),
                'max_shift': float(np.max(shifts)),
                'std_shift': float(np.std(shifts)),
                'num_gt_peaks': len(gt_peaks),
                'num_pred_peaks': len(pred_peaks),
                'shifts': shifts
            }
            
        except Exception as e:
            logging.error(f"Error calculating peak shift: {e}")
            return {
                'mean_shift': float('inf'),
                'max_shift': float('inf'),
                'num_gt_peaks': 0,
                'num_pred_peaks': 0
            }
    
    @staticmethod
    def calculate_integral_error(gt_profile: np.ndarray, 
                               pred_profile: np.ndarray,
                               peak_width: int = 20) -> Dict[str, Any]:
        try:
            gt_peaks, _ = SpectralMetrics.find_peaks(gt_profile)
            pred_peaks, _ = SpectralMetrics.find_peaks(pred_profile)
            
            if len(gt_peaks) == 0:
                return {'error': float('inf'), 'relative_error': float('inf')}
            
            integral_errors = []
            relative_errors = []
            
            for gt_peak in gt_peaks:
                start_idx = max(0, gt_peak - peak_width)
                end_idx = min(len(gt_profile), gt_peak + peak_width)
                
                gt_integral = np.trapz(gt_profile[start_idx:end_idx])
                pred_integral = np.trapz(pred_profile[start_idx:end_idx])
                
                error = abs(gt_integral - pred_integral)
                relative_error = error / max(abs(gt_integral), 1e-8)
                
                integral_errors.append(error)
                relative_errors.append(relative_error)
            
            return {
                'mean_error': float(np.mean(integral_errors)),
                'mean_relative_error': float(np.mean(relative_errors)),
                'max_error': float(np.max(integral_errors)),
                'max_relative_error': float(np.max(relative_errors))
            }
            
        except Exception as e:
            logging.error(f"Error calculating integral error: {e}")
            return {'error': float('inf'), 'relative_error': float('inf')}

class MetricsCalculator:
    def __init__(self):
        self.image_metrics = ImageMetrics()
        self.spectral_metrics = SpectralMetrics()
    
    def calculate_all_metrics(self, 
                            gt_image: np.ndarray, 
                            pred_image: np.ndarray,
                            blur_image: np.ndarray = None) -> Dict[str, Any]:
        
        metrics = {}
        
        metrics['psnr'] = self.image_metrics.calculate_psnr(gt_image, pred_image)
        metrics['ssim'] = self.image_metrics.calculate_ssim(gt_image, pred_image)
        metrics['mse'] = self.image_metrics.calculate_mse(gt_image, pred_image)
        metrics['mae'] = self.image_metrics.calculate_mae(gt_image, pred_image)
        
        if blur_image is not None:
            metrics['psnr_improvement'] = (
                self.image_metrics.calculate_psnr(gt_image, pred_image) -
                self.image_metrics.calculate_psnr(gt_image, blur_image)
            )
            metrics['ssim_improvement'] = (
                self.image_metrics.calculate_ssim(gt_image, pred_image) -
                self.image_metrics.calculate_ssim(gt_image, blur_image)
            )
        
        gt_profile = self.spectral_metrics.extract_line_profile(gt_image)
        pred_profile = self.spectral_metrics.extract_line_profile(pred_image)
        
        peak_shift_metrics = self.spectral_metrics.calculate_peak_shift(gt_profile, pred_profile)
        metrics.update({f'peak_{k}': v for k, v in peak_shift_metrics.items()})
        
        integral_metrics = self.spectral_metrics.calculate_integral_error(gt_profile, pred_profile)
        metrics.update({f'integral_{k}': v for k, v in integral_metrics.items()})
        
        return metrics
    
    def evaluate_batch(self, 
                      gt_images: List[np.ndarray],
                      pred_images: List[np.ndarray],
                      blur_images: List[np.ndarray] = None) -> Dict[str, Any]:
        
        if len(gt_images) != len(pred_images):
            raise ValueError("Number of ground truth and predicted images must match")
        
        if blur_images is not None and len(blur_images) != len(gt_images):
            raise ValueError("Number of blur images must match ground truth images")
        
        all_metrics = []
        
        for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)):
            blur_img = blur_images[i] if blur_images is not None else None
            metrics = self.calculate_all_metrics(gt_img, pred_img, blur_img)
            all_metrics.append(metrics)
        
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isinf(m[key])]
            if values:
                aggregated_metrics[f'{key}_mean'] = np.mean(values)
                aggregated_metrics[f'{key}_std'] = np.std(values)
                aggregated_metrics[f'{key}_min'] = np.min(values)
                aggregated_metrics[f'{key}_max'] = np.max(values)
            else:
                aggregated_metrics[f'{key}_mean'] = float('inf')
                aggregated_metrics[f'{key}_std'] = 0.0
        
        aggregated_metrics['num_samples'] = len(all_metrics)
        
        return aggregated_metrics