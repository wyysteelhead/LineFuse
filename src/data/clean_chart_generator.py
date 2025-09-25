import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
import random

class CleanChartGenerator:
    def __init__(self,
                 figure_size: tuple = (1024, 1024),  # 提升到1024分辨率
                 dpi: int = 150,
                 background_color: str = 'white',
                 line_color: str = 'black',
                 font_family: str = 'DejaVu Sans',
                 line_width: float = 0.8,  # 减细线条
                 enable_style_diversity: bool = True,
                 style_diversity_level: float = 1.0,
                 target_style: Optional[str] = None):
        """
        Initialize CleanChartGenerator with optional style diversity

        Args:
            enable_style_diversity: Whether to use diverse chart styles
            style_diversity_level: 0.0-1.0, controls diversity amount
            target_style: Specific style template ('scan_document', etc.)
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.background_color = background_color
        self.line_color = line_color
        self.font_family = font_family
        self.line_width = line_width
        self.enable_style_diversity = enable_style_diversity
        self.style_diversity_level = style_diversity_level
        self.target_style = target_style

        plt.rcParams['font.family'] = self.font_family

        # Initialize style manager if diversity is enabled
        self.style_manager = None
        if self.enable_style_diversity:
            try:
                from .chart_styles import ChartStyleManager
                self.style_manager = ChartStyleManager()
            except ImportError as e:
                logging.warning(f"Style diversity not available: {e}")
                self.enable_style_diversity = False
        
    def load_csv_data(self, csv_path: Union[str, Path]) -> Dict:
        try:
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            # Read header
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
            return {'data': data, 'columns': header}
        except Exception as e:
            logging.error(f"Error loading CSV file {csv_path}: {e}")
            raise
    
    def generate_clean_chart(self,
                           x_data: np.ndarray,
                           y_data: np.ndarray,
                           output_path: Union[str, Path],
                           xlabel: str = "Wavelength",
                           ylabel: str = "Intensity",
                           title: Optional[str] = None,
                           pure_line_only: bool = False,
                           pixel_perfect: bool = True) -> None:
        """
        Generate a clean chart with optional style diversity

        Args:
            pixel_perfect: If True, ensures pixel-perfect alignment with blur images
                          by using fixed margins and no tight_layout()
        """
        fig_width = self.figure_size[0] / self.dpi
        fig_height = self.figure_size[1] / self.dpi

        # Create figure with fixed subplot parameters for pixel-perfect alignment
        if pixel_perfect:
            # Fixed margins to ensure consistent positioning
            left, bottom, right, top = 0.1, 0.1, 0.9, 0.9
            fig = plt.figure(figsize=(fig_width, fig_height))
            ax = fig.add_axes([left, bottom, right-left, top-bottom])
        else:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # If pure_line_only mode, generate minimal clean line with exact positioning
        if pure_line_only:
            # Pure white background, no grid, no labels, no titles
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Plot the line with exact positioning
            ax.plot(x_data, y_data, color='black', linewidth=self.line_width)

            # Set exact data limits to ensure consistent positioning
            x_margin = (x_data.max() - x_data.min()) * 0.02
            y_margin = (y_data.max() - y_data.min()) * 0.02
            ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
            ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)

            # Remove all axes elements but maintain positioning
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)

        # Apply diverse styling if enabled
        elif self.enable_style_diversity and self.style_manager:
            # Get style configuration
            style_config = self.style_manager.get_random_style_config(
                diversity_level=self.style_diversity_level,
                target_template=self.target_style
            )

            # Override generator colors with style colors
            line_color = style_config.line_color
            background_color = style_config.background_color

            # Set initial background
            fig.patch.set_facecolor(background_color)
            ax.set_facecolor(background_color)

            # Plot the data with consistent positioning
            ax.plot(x_data, y_data, color=line_color, linewidth=self.line_width)

            # Set consistent data limits
            x_margin = (x_data.max() - x_data.min()) * 0.02
            y_margin = (y_data.max() - y_data.min()) * 0.02
            ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
            ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)

            # Apply complete styling (but with fixed positioning)
            fig, ax = self.style_manager.apply_style_to_chart(fig, ax, style_config, x_data, y_data)

            # Log style info for debugging
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Applied style: {self.style_manager.get_style_summary(style_config)}")

        else:
            # Use traditional styling with consistent positioning
            fig.patch.set_facecolor(self.background_color)
            ax.set_facecolor(self.background_color)

            ax.plot(x_data, y_data, color=self.line_color, linewidth=self.line_width)

            # Set consistent data limits
            x_margin = (x_data.max() - x_data.min()) * 0.02
            y_margin = (y_data.max() - y_data.min()) * 0.02
            ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
            ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            if title:
                ax.set_title(title, fontsize=12)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        # CRITICAL: Do NOT use tight_layout() for pixel-perfect alignment
        # plt.tight_layout()  # Removed to ensure consistent positioning

        # Save with fixed bbox to maintain exact pixel positioning
        if pixel_perfect:
            plt.savefig(output_path,
                       dpi=self.dpi,
                       bbox_inches=None,  # Use full figure area
                       facecolor=fig.get_facecolor() if hasattr(fig, 'get_facecolor') else self.background_color,
                       edgecolor='none',
                       pad_inches=0)  # No additional padding
        else:
            # Legacy mode with tight bbox
            plt.savefig(output_path,
                       dpi=self.dpi,
                       bbox_inches='tight',
                       facecolor=fig.get_facecolor() if hasattr(fig, 'get_facecolor') else self.background_color,
                       edgecolor='none')

        plt.close()
        
    def process_csv_to_chart(self,
                           csv_path: Union[str, Path],
                           output_path: Union[str, Path],
                           x_column: str = None,
                           y_column: str = None,
                           pure_line_only: bool = False,
                           pixel_perfect: bool = True,
                           **kwargs) -> None:

        csv_data = self.load_csv_data(csv_path)
        data = csv_data['data']
        columns = csv_data['columns']

        if x_column is None:
            x_column_idx = 0
        else:
            x_column_idx = columns.index(x_column)

        if y_column is None:
            y_column_idx = 1
        else:
            y_column_idx = columns.index(y_column)

        x_data = data[:, x_column_idx]
        y_data = data[:, y_column_idx]

        self.generate_clean_chart(x_data, y_data, output_path, pure_line_only=pure_line_only, pixel_perfect=pixel_perfect, **kwargs)
        
    def batch_process(self, 
                     input_dir: Union[str, Path],
                     output_dir: Union[str, Path],
                     pattern: str = "*.csv") -> None:
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(input_path.glob(pattern))
        
        for csv_file in csv_files:
            output_file = output_path / f"{csv_file.stem}.png"
            try:
                self.process_csv_to_chart(csv_file, output_file)
                logging.info(f"Generated chart: {output_file}")
            except Exception as e:
                logging.error(f"Failed to process {csv_file}: {e}")