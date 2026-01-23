# mini_branch/v3d - 3D Structural Analysis Elements
"""
V3D: 3D STRUCTURAL ELEMENTS
===========================

This package provides 3D structural analysis capabilities:
- Truss3D: Axial-only bar elements (6×6 stiffness, 3 DOF/node)
- (Future) Frame3D: Full beam elements (12×12 stiffness, 6 DOF/node)

These elements work with the dimension-agnostic kernel for assembly and solving.

USAGE:
------
    from mini_branch.v3d import Node3D, Truss3D, truss3d_global_stiffness
    from mini_branch.kernel import DOFManager, solve_linear
    from mini_branch.kernel.assemble import assemble_global_K
    
    # Create nodes
    nodes = {
        0: Node3D(0, 0.0, 0.0, 0.0),
        1: Node3D(1, 1.0, 0.0, 0.0),
        ...
    }
    
    # Create elements
    bars = [Truss3D(id=0, ni=0, nj=1, E=210e9, A=0.001), ...]
    
    # Build stiffness
    dof = DOFManager(dof_per_node=3)
    contributions = [(dof.element_dof_map([b.ni, b.nj]), 
                      truss3d_global_stiffness(nodes, b)) for b in bars]
    K = assemble_global_K(dof.ndof(len(nodes)), contributions)
    
    # Solve
    d, R, free = solve_linear(K, F, fixed_dofs)
"""

from .model import Node3D, Truss3D
from .elements import truss3d_global_stiffness, element_geometry_3d

__all__ = ['Node3D', 'Truss3D', 'truss3d_global_stiffness', 'element_geometry_3d']


