import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns

class VisualizationUtils:
    @staticmethod
    def setup_matplotlib():
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['figure.dpi'] = 150
        plt.style.use('default')
    
    @staticmethod
    def compare_images(images: List[np.ndarray], 
                      titles: List[str],
                      figsize: Tuple[int, int] = (15, 5),
                      save_path: Optional[Path] = None) -> None:
        
        VisualizationUtils.setup_matplotlib()
        
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            if len(img.shape) == 3:
                axes[i].imshow(img)
            else:
                axes[i].imshow(img, cmap='gray')
            
            axes[i].set_title(title, fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def plot_line_profiles(gt_profile: np.ndarray,
                          pred_profile: np.ndarray,
                          blur_profile: Optional[np.ndarray] = None,
                          title: str = "Line Profile Comparison",
                          save_path: Optional[Path] = None) -> None:
        
        VisualizationUtils.setup_matplotlib()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(gt_profile))
        
        ax.plot(x, gt_profile, label='Ground Truth', color='green', linewidth=2)
        ax.plot(x, pred_profile, label='Deblurred', color='blue', linewidth=2, alpha=0.8)
        
        if blur_profile is not None:
            ax.plot(x, blur_profile, label='Blurred Input', color='red', linewidth=1, alpha=0.6)
        
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, Any],
                              title: str = "Metrics Comparison",
                              save_path: Optional[Path] = None) -> None:
        
        VisualizationUtils.setup_matplotlib()
        
        metric_names = []
        values = []
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not np.isinf(value):
                metric_names.append(key.replace('_', ' ').title())
                values.append(value)
        
        if not values:
            print("No valid metrics to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metric_names, values, color='skyblue', alpha=0.7)
        
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def plot_training_curves(train_losses: List[float],
                           val_losses: List[float],
                           title: str = "Training Curves",
                           save_path: Optional[Path] = None) -> None:
        
        VisualizationUtils.setup_matplotlib()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
        ax.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def create_result_grid(clean_images: List[np.ndarray],
                          blur_images: List[np.ndarray],
                          deblurred_images: List[np.ndarray],
                          num_samples: int = 4,
                          figsize: Tuple[int, int] = (15, 12),
                          save_path: Optional[Path] = None) -> None:
        
        VisualizationUtils.setup_matplotlib()
        
        num_samples = min(num_samples, len(clean_images))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        column_titles = ['Blurred Input', 'Deblurred Output', 'Ground Truth']
        
        for j, title in enumerate(column_titles):
            axes[0, j].set_title(title, fontsize=14, fontweight='bold')
        
        for i in range(num_samples):
            # Blurred input
            if len(blur_images[i].shape) == 3:
                axes[i, 0].imshow(blur_images[i])
            else:
                axes[i, 0].imshow(blur_images[i], cmap='gray')
            axes[i, 0].axis('off')
            
            # Deblurred output
            if len(deblurred_images[i].shape) == 3:
                axes[i, 1].imshow(deblurred_images[i])
            else:
                axes[i, 1].imshow(deblurred_images[i], cmap='gray')
            axes[i, 1].axis('off')
            
            # Ground truth
            if len(clean_images[i].shape) == 3:
                axes[i, 2].imshow(clean_images[i])
            else:
                axes[i, 2].imshow(clean_images[i], cmap='gray')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def plot_metrics_distribution(metrics_list: List[Dict[str, Any]],
                                metric_name: str,
                                title: Optional[str] = None,
                                save_path: Optional[Path] = None) -> None:
        
        VisualizationUtils.setup_matplotlib()
        
        values = []
        for metrics in metrics_list:
            if metric_name in metrics and not np.isinf(metrics[metric_name]):
                values.append(metrics[metric_name])
        
        if not values:
            print(f"No valid values found for metric: {metric_name}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel(metric_name.replace('_', ' ').title())
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of {metric_name.replace("_", " ").title()}')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(values)
        ax2.set_ylabel(metric_name.replace('_', ' ').title())
        ax2.set_title(f'Box Plot of {metric_name.replace("_", " ").title()}')
        ax2.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def save_comparison_report(results: Dict[str, Any],
                             output_dir: Path,
                             filename: str = "evaluation_report.html") -> None:
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LineFuse Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ color: #2E7D32; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>LineFuse Evaluation Report</h1>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
        """
        
        for key, value in results.items():
            if isinstance(value, (int, float)) and not np.isinf(value):
                html_content += f"<tr><td>{key}</td><td>{value:.4f}</td><td>-</td><td>-</td><td>-</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Generated at</h2>
            <p>{}</p>
            
        </body>
        </html>
        """.format(np.datetime64('now'))
        
        with open(output_path / filename, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to: {output_path / filename}")