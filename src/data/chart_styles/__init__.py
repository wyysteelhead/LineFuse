"""
Chart Style Diversity System for LineFuse

This module provides comprehensive chart styling capabilities to generate
diverse training data that matches real-world document variations.
"""

from .style_manager import ChartStyleManager
from .background_styles import BackgroundStyleRenderer
from .grid_styles import GridStyleRenderer
from .axis_styles import AxisStyleRenderer
from .annotation_styles import AnnotationStyleRenderer

__all__ = [
    'ChartStyleManager',
    'BackgroundStyleRenderer',
    'GridStyleRenderer',
    'AxisStyleRenderer',
    'AnnotationStyleRenderer'
]

# Version info
__version__ = '1.0.0'
__author__ = 'LineFuse Team'