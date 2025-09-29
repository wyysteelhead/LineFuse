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
                 target_style: Optional[str] = None,
                 enable_line_variations: bool = False):
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
        self.enable_line_variations = enable_line_variations

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

    def _plot_line_with_variations(self, ax, x_data, y_data, base_color='black',
                                 base_linewidth=0.8, variation_params=None):
        """
        Plot line with optional variations (thinning, fading, dashing)
        Only used when enable_line_variations=True

        Args:
            variation_params: dict with keys like:
                - thinning_strength: 0.0-1.0
                - fading_strength: 0.0-1.0
                - dash_density: 0.0-1.0
        """
        if not self.enable_line_variations or variation_params is None:
            # Standard plotting
            ax.plot(x_data, y_data, color=base_color, linewidth=base_linewidth)
            return

        # Extract variation parameters
        thinning = variation_params.get('thinning_strength', 0.0)
        fading = variation_params.get('fading_strength', 0.0)
        dashing = variation_params.get('dash_density', 0.0)

        # 实现沿线条的粗细和颜色变化效果
        if thinning > 0 or fading > 0:
            # 分段绘制实现变化效果
            n_segments = max(40, len(x_data) // 8)  # 更多段数确保细致变化
            segment_indices = np.linspace(0, len(x_data)-1, n_segments+1, dtype=int)

            # 使用固定随机种子确保一致的变化模式
            import random
            random.seed(42)

            for i in range(len(segment_indices) - 1):
                start_idx = segment_indices[i]
                end_idx = segment_indices[i+1]

                # 创建更复杂的变化模式
                # 使用多个正弦波叠加 + 噪声模拟真实的线条退化
                position = i / max(1, len(segment_indices) - 2)  # 0 to 1

                # 多频率正弦波叠加
                wave1 = np.sin(position * 2 * np.pi * 3)  # 3个周期
                wave2 = np.sin(position * 2 * np.pi * 7) * 0.5  # 7个周期，幅度减半
                wave3 = np.sin(position * 2 * np.pi * 13) * 0.25  # 13个周期，幅度更小
                combined_wave = (wave1 + wave2 + wave3) / 1.75  # 归一化

                # 添加随机噪声
                noise = (random.random() - 0.5) * 0.6  # ±30% 噪声

                # 组合波形和噪声，转换到0-1范围
                variation_factor = (combined_wave + noise + 1) / 2
                variation_factor = max(0.1, min(0.9, variation_factor))  # 限制范围

                # 计算该段的线宽变化 - 大幅增强变化范围
                if thinning > 0:
                    # 变化范围：从极细到正常粗细 - 大幅增强对比
                    min_width = base_linewidth * (1.0 - 0.95 * thinning)  # 最多变细95%
                    max_width = base_linewidth * (1.0 + 0.5 * thinning)   # 最多变粗50%
                    segment_linewidth = min_width + (max_width - min_width) * variation_factor
                    segment_linewidth = max(0.02, segment_linewidth)  # 允许更细的线条
                else:
                    segment_linewidth = base_linewidth

                # 计算该段的颜色变化（褪色效果）- 大幅增强颜色对比
                if fading > 0 and base_color == 'black':
                    # 变化范围：从黑色到很浅的灰色 - 大幅增强对比
                    max_gray = fading * 0.85  # 最大灰度值增加到0.85（很浅）
                    gray_value = max_gray * (1.0 - variation_factor)  # 反向变化
                    segment_color = (gray_value, gray_value, gray_value)
                else:
                    segment_color = base_color

                # 绘制该段线条
                segment_x = x_data[start_idx:end_idx+1]
                segment_y = y_data[start_idx:end_idx+1]

                if len(segment_x) > 1:  # 确保有足够的点绘制
                    ax.plot(segment_x, segment_y,
                           color=segment_color,
                           linewidth=segment_linewidth,
                           linestyle='-',
                           solid_capstyle='round',  # 圆角端点
                           solid_joinstyle='round',  # 圆角连接
                           antialiased=True)  # 抗锯齿
        else:
            # 没有变化，正常绘制
            ax.plot(x_data, y_data, color=base_color, linewidth=base_linewidth)

        # 虚线效果（如果需要）
        if dashing > 0:
            dash_length = max(2, int(8 * (1 - dashing)))
            gap_length = max(1, int(4 * dashing))
            ax.plot(x_data, y_data, color=base_color, linewidth=base_linewidth * 0.8,
                   linestyle='--', dashes=[dash_length, gap_length], alpha=0.4)
        
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
                           pixel_perfect: bool = True,
                           line_variation_params: Optional[Dict] = None) -> None:
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

            # Plot the line with exact positioning (with optional variations)
            self._plot_line_with_variations(ax, x_data, y_data, 'black',
                                          self.line_width, line_variation_params)

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

            # Plot the data with consistent positioning (with optional variations)
            self._plot_line_with_variations(ax, x_data, y_data, line_color,
                                          self.line_width, line_variation_params)

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

            # Plot with traditional styling (with optional variations)
            self._plot_line_with_variations(ax, x_data, y_data, self.line_color,
                                          self.line_width, line_variation_params)

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