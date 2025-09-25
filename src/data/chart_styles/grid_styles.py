import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple
from .style_manager import StyleConfig

class GridStyleRenderer:
    """
    Handles different grid styles for charts

    Provides various grid patterns from clean academic grids to
    realistic paper grid patterns.
    """

    def apply_grid(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Apply grid styling based on configuration"""

        method_map = {
            'major_minor': self._apply_major_minor_grid,
            'square_grid': self._apply_square_grid,
            'no_grid': self._apply_no_grid,
            'custom_grid': self._apply_custom_grid,
            'minimal_grid': self._apply_minimal_grid
        }

        if config.grid_type in method_map:
            return method_map[config.grid_type](fig, ax, config)
        else:
            return self._apply_major_minor_grid(fig, ax, config)

    def _apply_major_minor_grid(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Standard major and minor grid - academic style"""
        ax.grid(True, which='major', color=config.grid_color, linewidth=0.8, alpha=0.6)
        ax.grid(True, which='minor', color=config.grid_color, linewidth=0.4, alpha=0.3)
        ax.minorticks_on()
        return fig, ax

    def _apply_square_grid(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """
        Square grid pattern - key for image.png style

        Creates uniform square grid like graph paper, essential for
        matching the target document style.
        """
        # Get current data limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Calculate optimal grid spacing for square appearance
        # Aim for approximately square grid cells
        target_divisions = 20  # Number of major divisions
        x_spacing = x_range / target_divisions
        y_spacing = y_range / target_divisions

        # Create custom tick locations for perfect squares
        x_ticks = np.arange(xlim[0], xlim[1] + x_spacing, x_spacing)
        y_ticks = np.arange(ylim[0], ylim[1] + y_spacing, y_spacing)

        # Set custom ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Apply grid with consistent styling
        ax.grid(True, color=config.grid_color, linewidth=0.5, alpha=0.4, linestyle='-')

        # Add finer subdivision grid (optional, for detailed graph paper look)
        if config.diversity_level > 0.7:  # Only for high diversity
            fine_x_spacing = x_spacing / 5
            fine_y_spacing = y_spacing / 5

            fine_x_ticks = np.arange(xlim[0], xlim[1] + fine_x_spacing, fine_x_spacing)
            fine_y_ticks = np.arange(ylim[0], ylim[1] + fine_y_spacing, fine_y_spacing)

            # Draw fine grid manually
            for x in fine_x_ticks:
                if x not in x_ticks:  # Don't overlap with major grid
                    ax.axvline(x, color=config.grid_color, linewidth=0.2, alpha=0.2, zorder=0)
            for y in fine_y_ticks:
                if y not in y_ticks:  # Don't overlap with major grid
                    ax.axhline(y, color=config.grid_color, linewidth=0.2, alpha=0.2, zorder=0)

        return fig, ax

    def _apply_no_grid(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """No grid - clean minimal look"""
        ax.grid(False)
        return fig, ax

    def _apply_custom_grid(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Custom grid with irregular spacing - more organic look"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Create slightly irregular grid spacing
        np.random.seed(42)  # For consistent randomization
        base_x_divisions = 15
        base_y_divisions = 15

        # Add some randomness to grid line positions
        x_positions = []
        for i in range(base_x_divisions):
            base_pos = xlim[0] + (i / base_x_divisions) * x_range
            # Add small random offset (max 5% of spacing)
            offset = np.random.uniform(-0.05, 0.05) * (x_range / base_x_divisions)
            x_positions.append(base_pos + offset)

        y_positions = []
        for i in range(base_y_divisions):
            base_pos = ylim[0] + (i / base_y_divisions) * y_range
            offset = np.random.uniform(-0.05, 0.05) * (y_range / base_y_divisions)
            y_positions.append(base_pos + offset)

        # Draw custom grid
        for x in x_positions:
            ax.axvline(x, color=config.grid_color, linewidth=0.4, alpha=0.3, zorder=0)
        for y in y_positions:
            ax.axhline(y, color=config.grid_color, linewidth=0.4, alpha=0.3, zorder=0)

        # Turn off default grid
        ax.grid(False)
        return fig, ax

    def _apply_minimal_grid(self, fig: Figure, ax: Axes, config: StyleConfig) -> Tuple[Figure, Axes]:
        """Minimal grid - only a few guide lines"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Only 3-4 major grid lines in each direction
        x_lines = [xlim[0] + x_range * frac for frac in [0.25, 0.5, 0.75]]
        y_lines = [ylim[0] + y_range * frac for frac in [0.25, 0.5, 0.75]]

        # Draw minimal grid
        for x in x_lines:
            ax.axvline(x, color=config.grid_color, linewidth=0.6,
                      alpha=0.4, linestyle='--', zorder=0)
        for y in y_lines:
            ax.axhline(y, color=config.grid_color, linewidth=0.6,
                      alpha=0.4, linestyle='--', zorder=0)

        # Turn off default grid
        ax.grid(False)
        return fig, ax