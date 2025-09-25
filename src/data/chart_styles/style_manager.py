import random
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

@dataclass
class StyleConfig:
    """Configuration for chart styling"""
    background_type: str = 'clean_white'
    grid_type: str = 'major_minor'
    axis_type: str = 'full_axis'
    annotation_type: str = 'typed_labels'
    border_type: str = 'clean_border'

    # Color and appearance
    background_color: str = 'white'
    line_color: str = 'black'
    grid_color: str = 'gray'
    text_color: str = 'black'

    # Probability weights (for random generation)
    diversity_level: float = 1.0  # 0.0 = no diversity, 1.0 = full diversity

class ChartStyleManager:
    """
    Central manager for chart style diversity

    Handles all aspects of chart styling to create realistic variations
    that match real-world document scanning scenarios.
    """

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Style probability distributions
        self.style_probabilities = {
            'background_type': {
                'clean_white': 0.25,
                'graph_paper': 0.25,    # Key for image.png style
                'aged_paper': 0.20,
                'scan_document': 0.20,  # Key for image.png style
                'lab_notebook': 0.10
            },
            'grid_type': {
                'major_minor': 0.25,
                'square_grid': 0.30,    # Key for image.png style
                'no_grid': 0.15,
                'custom_grid': 0.20,
                'minimal_grid': 0.10
            },
            'axis_type': {
                'full_axis': 0.30,
                'ticks_only': 0.25,
                'handwritten': 0.20,    # Key for image.png style
                'minimal': 0.15,
                'no_axis': 0.10
            },
            'annotation_type': {
                'typed_labels': 0.40,
                'handwritten_notes': 0.25,  # Key for image.png style
                'measurement_marks': 0.20,
                'minimal_text': 0.15
            },
            'border_type': {
                'clean_border': 0.40,
                'scan_edges': 0.30,     # Key for image.png style
                'notebook_style': 0.20,
                'photo_frame': 0.10
            }
        }

        # Predefined style templates
        self.style_templates = {
            'scan_document': {
                'background_type': 'graph_paper',
                'grid_type': 'square_grid',
                'axis_type': 'handwritten',
                'annotation_type': 'handwritten_notes',
                'border_type': 'scan_edges',
                'background_color': '#f8f6f0',  # Slightly aged white
                'line_color': '#2c2c2c',        # Dark gray, not pure black
                'grid_color': '#d0d0d0',        # Light gray grid
                'text_color': '#3c3c3c'         # Dark gray text
            },
            'academic_paper': {
                'background_type': 'clean_white',
                'grid_type': 'major_minor',
                'axis_type': 'full_axis',
                'annotation_type': 'typed_labels',
                'border_type': 'clean_border'
            },
            'lab_notebook': {
                'background_type': 'lab_notebook',
                'grid_type': 'square_grid',
                'axis_type': 'ticks_only',
                'annotation_type': 'handwritten_notes',
                'border_type': 'notebook_style'
            },
            'field_notes': {
                'background_type': 'aged_paper',
                'grid_type': 'minimal_grid',
                'axis_type': 'minimal',
                'annotation_type': 'handwritten_notes',
                'border_type': 'scan_edges'
            }
        }

    def get_random_style_config(self, diversity_level: float = 1.0,
                               target_template: Optional[str] = None) -> StyleConfig:
        """
        Generate a random style configuration

        Args:
            diversity_level: 0.0-1.0, controls how diverse the styles are
            target_template: Optional specific template to use

        Returns:
            StyleConfig with randomized or template-based settings
        """
        if target_template and target_template in self.style_templates:
            template = self.style_templates[target_template]
            config = StyleConfig(**template)
        else:
            # Generate random configuration based on probabilities
            config = StyleConfig()

            if diversity_level > 0:
                # Apply randomization based on diversity level
                config.background_type = self._sample_from_distribution(
                    self.style_probabilities['background_type'], diversity_level
                )
                config.grid_type = self._sample_from_distribution(
                    self.style_probabilities['grid_type'], diversity_level
                )
                config.axis_type = self._sample_from_distribution(
                    self.style_probabilities['axis_type'], diversity_level
                )
                config.annotation_type = self._sample_from_distribution(
                    self.style_probabilities['annotation_type'], diversity_level
                )
                config.border_type = self._sample_from_distribution(
                    self.style_probabilities['border_type'], diversity_level
                )

            # Apply consistent color schemes based on background
            config = self._apply_color_consistency(config)

        config.diversity_level = diversity_level
        return config

    def _sample_from_distribution(self, distribution: Dict[str, float],
                                 diversity_level: float) -> str:
        """Sample from probability distribution with diversity control"""
        if diversity_level <= 0:
            # Return most common option
            return max(distribution.items(), key=lambda x: x[1])[0]

        # Adjust probabilities based on diversity level
        adjusted_probs = {}
        for key, prob in distribution.items():
            if diversity_level < 1.0:
                # Favor more common options when diversity is low
                adjusted_probs[key] = prob ** (2 - diversity_level)
            else:
                adjusted_probs[key] = prob

        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        normalized_probs = {k: v/total_prob for k, v in adjusted_probs.items()}

        # Sample
        rand_val = random.random()
        cumsum = 0
        for key, prob in normalized_probs.items():
            cumsum += prob
            if rand_val <= cumsum:
                return key

        # Fallback
        return list(distribution.keys())[0]

    def _apply_color_consistency(self, config: StyleConfig) -> StyleConfig:
        """Apply consistent color schemes based on style choices"""

        color_schemes = {
            'clean_white': {
                'background_color': 'white',
                'line_color': 'black',
                'grid_color': 'gray',
                'text_color': 'black'
            },
            'graph_paper': {
                'background_color': '#f8f6f0',  # Slightly warm white
                'line_color': '#2c2c2c',
                'grid_color': '#b8b8b8',        # Medium gray for grid
                'text_color': '#3c3c3c'
            },
            'aged_paper': {
                'background_color': '#f5f2e8',  # Aged yellow-white
                'line_color': '#4a4a4a',
                'grid_color': '#c0b8a8',
                'text_color': '#5a5a5a'
            },
            'scan_document': {
                'background_color': '#f8f6f0',
                'line_color': '#2c2c2c',
                'grid_color': '#d0d0d0',
                'text_color': '#3c3c3c'
            },
            'lab_notebook': {
                'background_color': '#fbfbfb',  # Very light gray
                'line_color': '#1a1a1a',
                'grid_color': '#e0e0e0',
                'text_color': '#2a2a2a'
            }
        }

        if config.background_type in color_schemes:
            scheme = color_schemes[config.background_type]
            config.background_color = scheme['background_color']
            config.line_color = scheme['line_color']
            config.grid_color = scheme['grid_color']
            config.text_color = scheme['text_color']

        return config

    def apply_style_to_chart(self, fig: Figure, ax: Axes,
                           config: StyleConfig,
                           x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Figure, Axes]:
        """
        Apply the complete style configuration to a matplotlib chart

        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes
            config: Style configuration to apply
            x_data: X-axis data for contextual styling
            y_data: Y-axis data for contextual styling

        Returns:
            Styled figure and axes
        """
        # Import style renderers
        from .background_styles import BackgroundStyleRenderer
        from .grid_styles import GridStyleRenderer
        from .axis_styles import AxisStyleRenderer
        from .annotation_styles import AnnotationStyleRenderer

        # Create renderer instances
        bg_renderer = BackgroundStyleRenderer()
        grid_renderer = GridStyleRenderer()
        axis_renderer = AxisStyleRenderer()
        annotation_renderer = AnnotationStyleRenderer()

        # Apply styles in order
        fig, ax = bg_renderer.apply_background(fig, ax, config)
        fig, ax = grid_renderer.apply_grid(fig, ax, config)
        fig, ax = axis_renderer.apply_axis_style(fig, ax, config, x_data, y_data)
        fig, ax = annotation_renderer.apply_annotations(fig, ax, config, x_data, y_data)

        return fig, ax

    def get_style_summary(self, config: StyleConfig) -> str:
        """Get a human-readable summary of the style configuration"""
        return (f"Style: {config.background_type} background, "
                f"{config.grid_type} grid, "
                f"{config.axis_type} axes, "
                f"{config.annotation_type} annotations")