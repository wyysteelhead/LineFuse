import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple
from .style_manager import StyleConfig

class AxisStyleRenderer:
    """
    Handles different axis labeling and styling approaches

    Provides various axis styles from full scientific labeling to
    handwritten annotations matching real document styles.
    """

    def apply_axis_style(self, fig: Figure, ax: Axes, config: StyleConfig,
                        x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Apply axis styling based on configuration"""

        method_map = {
            'full_axis': self._apply_full_axis,
            'ticks_only': self._apply_ticks_only,
            'handwritten': self._apply_handwritten,
            'minimal': self._apply_minimal,
            'no_axis': self._apply_no_axis
        }

        if config.axis_type in method_map:
            return method_map[config.axis_type](fig, ax, config, x_data, y_data)
        else:
            return self._apply_full_axis(fig, ax, config, x_data, y_data)

    def _apply_full_axis(self, fig: Figure, ax: Axes, config: StyleConfig,
                        x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Full academic axis labels - complete scientific styling"""
        ax.set_xlabel('Wavelength (nm)', fontsize=10, color=config.text_color)
        ax.set_ylabel('Intensity', fontsize=10, color=config.text_color)
        ax.tick_params(labelsize=8, colors=config.text_color)
        return fig, ax

    def _apply_ticks_only(self, fig: Figure, ax: Axes, config: StyleConfig,
                         x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Only tick marks, no axis labels"""
        ax.set_xlabel('', fontsize=0)
        ax.set_ylabel('', fontsize=0)
        ax.tick_params(labelsize=8, colors=config.text_color)
        return fig, ax

    def _apply_handwritten(self, fig: Figure, ax: Axes, config: StyleConfig,
                          x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """
        Handwritten style annotations - key for image.png matching

        Places handwritten-style numbers at key positions, similar to
        the manual annotations seen in scanned documents.
        """
        # Remove default axis labels
        ax.set_xlabel('', fontsize=0)
        ax.set_ylabel('', fontsize=0)

        # Get data ranges for intelligent annotation placement
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()

        # Calculate meaningful annotation positions
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Key x-axis positions (wavelength markers)
        x_annotations = [
            (x_min + 0.2 * x_range, f"{int(x_min + 0.2 * x_range)}"),
            (x_min + 0.5 * x_range, f"{int(x_min + 0.5 * x_range)}"),
            (x_min + 0.8 * x_range, f"{int(x_min + 0.8 * x_range)}")
        ]

        # Key y-axis positions (intensity markers) - like image.png
        # Use round numbers similar to the 40, 80, 100, 120 pattern
        y_annotations = []
        y_step = max(20, int(y_range / 5 / 20) * 20)  # Round to nearest 20
        start_y = int(y_min / y_step) * y_step
        for i in range(4):  # 3-4 major markers
            y_val = start_y + (i + 1) * y_step
            if y_min <= y_val <= y_max:
                y_annotations.append((y_val, f"{int(y_val)}"))

        # Apply handwritten-style font and positioning
        handwritten_font = {
            'family': 'serif',  # More handwritten-like
            'style': 'italic',
            'weight': 'normal',
            'size': 9
        }

        # Place x-axis annotations (below the plot)
        for x_pos, label in x_annotations:
            ax.annotate(label, xy=(x_pos, y_min), xytext=(x_pos, y_min - 0.08 * y_range),
                       fontfamily=handwritten_font['family'], fontsize=handwritten_font['size'],
                       fontstyle=handwritten_font['style'], fontweight=handwritten_font['weight'],
                       color=config.text_color, ha='center', va='top', clip_on=False)

        # Place y-axis annotations (to the left of the plot) - like image.png
        for y_pos, label in y_annotations:
            ax.annotate(label, xy=(x_min, y_pos), xytext=(x_min - 0.08 * x_range, y_pos),
                       fontfamily=handwritten_font['family'], fontsize=handwritten_font['size'],
                       fontstyle=handwritten_font['style'], fontweight=handwritten_font['weight'],
                       color=config.text_color, ha='right', va='center', clip_on=False)

        # Remove tick marks for cleaner look
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        return fig, ax

    def _apply_minimal(self, fig: Figure, ax: Axes, config: StyleConfig,
                      x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Minimal axis styling - very clean"""
        ax.set_xlabel('', fontsize=0)
        ax.set_ylabel('', fontsize=0)

        # Only show a few key tick marks
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Set minimal ticks
        x_ticks = [xlim[0] + (xlim[1] - xlim[0]) * frac for frac in [0, 0.5, 1.0]]
        y_ticks = [ylim[0] + (ylim[1] - ylim[0]) * frac for frac in [0, 0.5, 1.0]]

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(labelsize=7, colors=config.text_color)

        return fig, ax

    def _apply_no_axis(self, fig: Figure, ax: Axes, config: StyleConfig,
                      x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """No axis labels or ticks - pure data visualization"""
        ax.set_xlabel('', fontsize=0)
        ax.set_ylabel('', fontsize=0)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Remove axis spines for ultra-clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        return fig, ax