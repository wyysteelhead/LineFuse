import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple
from .style_manager import StyleConfig

class AnnotationStyleRenderer:
    """
    Handles different annotation and text styling approaches

    Provides various annotation styles from typed labels to
    handwritten notes matching real document annotations.
    """

    def apply_annotations(self, fig: Figure, ax: Axes, config: StyleConfig,
                         x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Apply annotation styling based on configuration"""

        method_map = {
            'typed_labels': self._apply_typed_labels,
            'handwritten_notes': self._apply_handwritten_notes,
            'measurement_marks': self._apply_measurement_marks,
            'minimal_text': self._apply_minimal_text
        }

        if config.annotation_type in method_map:
            return method_map[config.annotation_type](fig, ax, config, x_data, y_data)
        else:
            return self._apply_typed_labels(fig, ax, config, x_data, y_data)

    def _apply_typed_labels(self, fig: Figure, ax: Axes, config: StyleConfig,
                           x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Standard typed labels - professional academic style"""
        # Add title if appropriate
        if config.diversity_level > 0.5:
            ax.set_title('Spectrum Analysis', fontsize=11, color=config.text_color, pad=15)

        # No additional annotations needed - handled by axis styling
        return fig, ax

    def _apply_handwritten_notes(self, fig: Figure, ax: Axes, config: StyleConfig,
                                x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """
        Handwritten notes style - key for image.png matching

        Adds handwritten-style annotations and notes that simulate
        manual markup on scanned documents.
        """
        # Get data characteristics for intelligent annotation
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Handwritten font style
        handwritten_style = {
            'family': 'serif',
            'style': 'italic',
            'weight': 'normal',
            'size': 8
        }

        # Find peaks for annotation (simple peak detection)
        if len(y_data) > 10:
            # Use simple local maxima detection
            peaks = []
            for i in range(2, len(y_data) - 2):
                if (y_data[i] > y_data[i-1] and y_data[i] > y_data[i+1] and
                    y_data[i] > y_min + 0.7 * y_range):  # Only high peaks
                    peaks.append((x_data[i], y_data[i]))

            # Limit to 2-3 most significant peaks
            peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:3]

            # Annotate peaks with handwritten-style labels
            for i, (x_peak, y_peak) in enumerate(peaks):
                if i < 2:  # Only annotate top 2 peaks
                    # Handwritten-style peak labels
                    label = f"Peak {i+1}"
                    offset_x = x_range * (0.02 + 0.01 * i)
                    offset_y = y_range * (0.05 + 0.02 * i)

                    ax.annotate(label,
                               xy=(x_peak, y_peak),
                               xytext=(x_peak + offset_x, y_peak + offset_y),
                               fontfamily=handwritten_style['family'],
                               fontsize=handwritten_style['size'],
                               fontstyle=handwritten_style['style'],
                               fontweight=handwritten_style['weight'],
                               color=config.text_color,
                               alpha=0.8,
                               arrowprops=dict(arrowstyle='->',
                                             color=config.text_color,
                                             alpha=0.6,
                                             lw=0.8))

        # Add some random handwritten-style measurement notes
        if config.diversity_level > 0.7:
            # Notes in margins (like real lab notebooks)
            notes = [
                ("Sample #1", (x_min + 0.85 * x_range, y_min + 0.9 * y_range)),
                ("280°C", (x_min + 0.1 * x_range, y_min + 0.85 * y_range)),
                ("pH 7.2", (x_min + 0.9 * x_range, y_min + 0.15 * y_range))
            ]

            for note_text, (note_x, note_y) in notes:
                # Only add note if it won't overlap with data
                if (note_y > y_max + 0.1 * y_range or
                    note_y < y_min - 0.1 * y_range or
                    note_x < x_min or note_x > x_max):

                    ax.annotate(note_text,
                               xy=(note_x, note_y),
                               fontfamily=handwritten_style['family'],
                               fontsize=handwritten_style['size'],
                               fontstyle=handwritten_style['style'],
                               fontweight=handwritten_style['weight'],
                               color=config.text_color,
                               alpha=0.7,
                               ha='center', va='center')

        return fig, ax

    def _apply_measurement_marks(self, fig: Figure, ax: Axes, config: StyleConfig,
                                x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Measurement marks and technical annotations"""
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Technical measurement font
        tech_style = {
            'family': 'monospace',
            'style': 'normal',
            'weight': 'normal',
            'size': 7
        }

        # Add measurement lines and values
        measurement_positions = [
            (x_min + 0.25 * x_range, "25%"),
            (x_min + 0.75 * x_range, "75%")
        ]

        for x_pos, label in measurement_positions:
            # Find corresponding y value
            idx = np.argmin(np.abs(x_data - x_pos))
            y_pos = y_data[idx]

            # Draw measurement line
            ax.plot([x_pos, x_pos], [y_min, y_pos],
                   color=config.text_color, linestyle='--',
                   linewidth=0.8, alpha=0.6)

            # Add label
            ax.annotate(f"{label}\n{y_pos:.1f}",
                       xy=(x_pos, y_pos),
                       xytext=(x_pos, y_pos + 0.15 * y_range),
                       fontfamily=tech_style['family'],
                       fontsize=tech_style['size'],
                       fontstyle=tech_style['style'],
                       fontweight=tech_style['weight'],
                       color=config.text_color,
                       ha='center', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.2",
                                facecolor='white',
                                edgecolor=config.text_color,
                                alpha=0.8, linewidth=0.5))

        return fig, ax

    def _apply_minimal_text(self, fig: Figure, ax: Axes, config: StyleConfig,
                           x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """Minimal text annotations - very clean presentation"""
        # Only add a small, discrete label if high diversity
        if config.diversity_level > 0.8:
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            y_range = y_max - y_min

            minimal_style = {
                'family': 'sans-serif',
                'style': 'normal',
                'weight': 'light',
                'size': 6
            }

            # Small discrete label in corner
            ax.annotate("Spectrum",
                       xy=(x_max, y_max),
                       xytext=(x_max - 0.05 * (x_max - x_min),
                              y_max - 0.05 * y_range),
                       fontfamily=minimal_style['family'],
                       fontsize=minimal_style['size'],
                       fontstyle=minimal_style['style'],
                       fontweight=minimal_style['weight'],
                       color=config.text_color,
                       alpha=0.6,
                       ha='right', va='top')

        return fig, ax