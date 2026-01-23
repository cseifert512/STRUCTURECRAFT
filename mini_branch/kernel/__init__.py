# mini_branch/kernel - Dimension-agnostic structural analysis core
"""
KERNEL: THE DIMENSION-AGNOSTIC FOUNDATION
==========================================

This package contains the core abstractions that work for ANY structural
analysis: 2D frames, 3D trusses, 3D frames, etc.

The key insight: assembly and solving don't care about dimensions.
They just need:
- A way to map (node_id, local_dof) → global_dof_index
- Element stiffness matrices (any size)
- Fixed DOF lists
- Load vectors

The ELEMENT implementations (Frame2D, Truss3D, etc.) are dimension-specific,
but the kernel plumbing is universal.
"""

from .dof import DOFManager
from .solve import solve_linear, MechanismError

__all__ = ['DOFManager', 'solve_linear', 'MechanismError']

