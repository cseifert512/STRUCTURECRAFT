# mini_branch/generative - Parametric Geometry Generators
"""
GENERATIVE: Parametric Structure Generators
============================================

This package generates 3D structural geometries from design parameters.
The key idea: turn a "design brief" into nodes, bars, and boundary conditions.

Available Generators:
---------------------
- canopy: Spaceframe canopy/pavilion structures over rectangular footprints

USAGE:
------
    from mini_branch.generative import generate_canopy, CanopyParams
    
    params = CanopyParams(
        width=10.0, depth=8.0,
        nx=5, ny=4,
        max_height=3.0,
        heightfield='paraboloid',
        topology='diagrid',
        support_layout='corners'
    )
    
    nodes, bars, fixed_dofs, loads = generate_canopy(params)
"""

from .canopy import generate_canopy, CanopyParams

__all__ = ['generate_canopy', 'CanopyParams']


