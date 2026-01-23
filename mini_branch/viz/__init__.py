# mini_branch/viz - Visualization Tools
"""
VIZ: Visualization for 2D and 3D Structures
============================================

This package provides visualization tools:
- viz2d: Existing 2D frame visualization (matplotlib)
- viz3d: New 3D structure visualization (Plotly)
"""

from .viz3d import plot_spaceframe_3d, create_spaceframe_figure

__all__ = ['plot_spaceframe_3d', 'create_spaceframe_figure']


