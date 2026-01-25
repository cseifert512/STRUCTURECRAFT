# mini_branch - Structural Analysis and Design Exploration
"""
MINI-BRANCH: A Structural Design Exploration Platform
======================================================

This package provides:
- 2D frame analysis (existing, production-ready)
- 3D truss analysis (new, via kernel)
- Design exploration with Pareto optimization
- ML-guided search (surrogate models)

ARCHITECTURE:
-------------
    kernel/         Dimension-agnostic core (DOF management, assembly, solve)
    model.py        2D model definitions (Node, Frame2D)
    elements.py     2D element stiffness matrices
    assembly.py     2D assembly (uses kernel internally)
    solve.py        Backward-compatible solve interface
    loads.py        2D equivalent nodal loads (UDL)
    explore.py      Parametric design generation & evaluation
    pareto.py       Multi-objective optimization
    viz.py          Visualization
    
    v3d/            3D-specific code (coming soon)
    generative/     Geometry generators (coming soon)
"""

# Re-export kernel components for convenience
from .kernel import DOFManager, solve_linear, MechanismError

# Version
__version__ = "0.3.0"  # Bumped for kernel refactor





