import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Optional
from .style_manager import StyleConfig

class BackgroundStyleRenderer:
    """
    Handles different background styles for charts

    Supports various paper types and scanning effects to simulate
    real-world document variations.
    """

    def apply_background(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Apply background styling based on configuration"""

        method_map = {
            'clean_white': self._apply_clean_white,
            'graph_paper': self._apply_graph_paper,
            'aged_paper': self._apply_aged_paper,
            'scan_document': self._apply_scan_document,
            'lab_notebook': self._apply_lab_notebook
        }

        if config.background_type in method_map:
            return method_map[config.background_type](fig, ax, config)
        else:
            return self._apply_clean_white(fig, ax, config)

    def _apply_clean_white(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Clean white background - modern academic style"""
        fig.patch.set_facecolor(config.background_color)
        ax.set_facecolor(config.background_color)
        return fig, ax

    def _apply_graph_paper(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Graph paper background with regular grid pattern"""
        fig.patch.set_facecolor(config.background_color)
        ax.set_facecolor(config.background_color)

        # Add subtle grid pattern as background (finer than the main grid)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Calculate grid spacing
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Fine grid - many small squares like real graph paper
        fine_x_spacing = x_range / 50  # 50 divisions
        fine_y_spacing = y_range / 50

        # Draw fine grid lines
        x_fine = np.arange(xlim[0], xlim[1], fine_x_spacing)
        y_fine = np.arange(ylim[0], ylim[1], fine_y_spacing)

        for x in x_fine:
            ax.axvline(x, color=config.grid_color, linewidth=0.3, alpha=0.3, zorder=0)
        for y in y_fine:
            ax.axhline(y, color=config.grid_color, linewidth=0.3, alpha=0.3, zorder=0)

        return fig, ax

    def _apply_aged_paper(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Aged paper with yellowing and slight texture"""
        fig.patch.set_facecolor(config.background_color)
        ax.set_facecolor(config.background_color)

        # Add subtle texture/staining
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Create subtle color variations to simulate aging
        np.random.seed(42)  # For consistent aging pattern
        n_stains = 10

        for _ in range(n_stains):
            # Random circular stains
            center_x = np.random.uniform(xlim[0], xlim[1])
            center_y = np.random.uniform(ylim[0], ylim[1])
            radius_x = np.random.uniform(x_range * 0.05, x_range * 0.15)
            radius_y = np.random.uniform(y_range * 0.05, y_range * 0.15)

            circle = patches.Ellipse((center_x, center_y), radius_x, radius_y,
                                   alpha=0.02, facecolor='brown', zorder=0)
            ax.add_patch(circle)

        return fig, ax

    def _apply_scan_document(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """
        Scanned document style - key style for image.png matching

        This creates the specific look of scanned paper documents with:
        - Slightly off-white background
        - Subtle scanning artifacts
        - Paper texture effects
        """
        fig.patch.set_facecolor(config.background_color)
        ax.set_facecolor(config.background_color)

        # Get current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Add very subtle noise pattern to simulate scanning artifacts
        np.random.seed(123)  # For consistent pattern
        noise_density = 0.001  # Very subtle

        # Add tiny random spots (scanning dust/artifacts)
        n_spots = int(noise_density * x_range * y_range)
        for _ in range(n_spots):
            spot_x = np.random.uniform(xlim[0], xlim[1])
            spot_y = np.random.uniform(ylim[0], ylim[1])
            spot_size = np.random.uniform(0.001, 0.003) * min(x_range, y_range)

            circle = patches.Circle((spot_x, spot_y), spot_size,
                                  alpha=0.05, facecolor='gray', zorder=0)
            ax.add_patch(circle)

        # Add subtle paper fiber texture (very faint lines)
        for _ in range(20):
            start_x = np.random.uniform(xlim[0], xlim[1])
            start_y = np.random.uniform(ylim[0], ylim[1])
            length = np.random.uniform(x_range * 0.01, x_range * 0.05)
            angle = np.random.uniform(0, 2 * np.pi)

            end_x = start_x + length * np.cos(angle)
            end_y = start_y + length * np.sin(angle)

            ax.plot([start_x, end_x], [start_y, end_y],
                   color=config.grid_color, linewidth=0.2, alpha=0.1, zorder=0)

        return fig, ax

    def _apply_lab_notebook(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Laboratory notebook style with binding holes and lines"""
        fig.patch.set_facecolor(config.background_color)
        ax.set_facecolor(config.background_color)

        # Add notebook-style elements
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Left margin line (like in real notebooks)
        margin_x = xlim[0] + x_range * 0.08
        ax.axvline(margin_x, color='red', linewidth=0.8, alpha=0.6, zorder=0)

        # Horizontal lines (like ruled paper)
        n_lines = 15
        for i in range(n_lines):
            y_pos = ylim[0] + (i + 1) * y_range / (n_lines + 1)
            ax.axhline(y_pos, color='lightblue', linewidth=0.3, alpha=0.4, zorder=0)

        # Binding holes (circles on the left)
        hole_x = xlim[0] + x_range * 0.04
        for i in range(4):
            hole_y = ylim[0] + (i + 1) * y_range / 5
            hole = patches.Circle((hole_x, hole_y), x_range * 0.008,
                                facecolor='white', edgecolor='gray',
                                linewidth=0.5, zorder=0)
            ax.add_patch(hole)

        return fig, ax