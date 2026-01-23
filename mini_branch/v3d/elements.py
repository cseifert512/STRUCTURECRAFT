# mini_branch/v3d/elements.py
"""
3D TRUSS ELEMENT: Stiffness Matrix from Direction Cosines
=========================================================

PURPOSE:
--------
This module computes the 6×6 global stiffness matrix for a 3D truss element.
This is THE core engineering calculation that enables 3D structural analysis.

ENGINEERING DERIVATION:
-----------------------
A 3D truss bar has stiffness only along its axis. In LOCAL coordinates
(x' along the bar), the stiffness matrix is simple:

    k_local = (EA/L) × [ 1  -1 ]
                       [-1   1 ]

But we need this in GLOBAL coordinates (x, y, z). We use direction cosines
to transform from local to global:

    l = (xj - xi) / L    (cosine with x-axis)
    m = (yj - yi) / L    (cosine with y-axis)
    n = (zj - zi) / L    (cosine with z-axis)

The 6×6 global stiffness matrix becomes:

    ke_global = (EA/L) × [  l²    lm    ln   -l²   -lm   -ln  ]
                         [  lm    m²    mn   -lm   -m²   -mn  ]
                         [  ln    mn    n²   -ln   -mn   -n²  ]
                         [ -l²   -lm   -ln    l²    lm    ln  ]
                         [ -lm   -m²   -mn    lm    m²    mn  ]
                         [ -ln   -mn   -n²    ln    mn    n²  ]

This elegant formula comes from:
    ke_global = T^T × k_local_expanded × T

where T is the transformation matrix containing direction cosines.

WHY THIS MATTERS:
-----------------
This is the foundation for ALL 3D truss analysis:
- Space trusses
- Spaceframes
- Canopy structures
- Lattice towers
- Any structure with axial-only members in 3D
"""

import numpy as np
from typing import Dict, Tuple

from .model import Node3D, Truss3D


def element_geometry_3d(nodes: Dict[int, Node3D], element: Truss3D) -> Tuple[float, float, float, float]:
    """
    Compute length and direction cosines for a 3D truss element.
    
    Direction cosines (l, m, n) define the orientation of the element
    in 3D space. They are the cosines of angles between the element axis
    and the global x, y, z axes.
    
    Properties:
    - l² + m² + n² = 1 (unit vector)
    - l, m, n can be positive or negative depending on element direction
    
    Parameters:
    -----------
    nodes : Dict[int, Node3D]
        Dictionary mapping node IDs to Node3D objects
        
    element : Truss3D
        The truss element
    
    Returns:
    --------
    Tuple[float, float, float, float]
        (L, l, m, n) where:
        - L: Element length (m)
        - l: Direction cosine with x-axis (dx/L)
        - m: Direction cosine with y-axis (dy/L)
        - n: Direction cosine with z-axis (dz/L)
    
    Raises:
    -------
    ValueError
        If element has zero length (nodes at same location)
    
    Example:
    --------
    >>> nodes = {0: Node3D(0, 0, 0, 0), 1: Node3D(1, 1, 0, 0)}
    >>> bar = Truss3D(0, 0, 1, 210e9, 0.001)
    >>> L, l, m, n = element_geometry_3d(nodes, bar)
    >>> print(f"L={L}, direction=({l}, {m}, {n})")
    L=1.0, direction=(1.0, 0.0, 0.0)
    """
    ni = nodes[element.ni]
    nj = nodes[element.nj]
    
    # Displacement vector from node i to node j
    dx = nj.x - ni.x
    dy = nj.y - ni.y
    dz = nj.z - ni.z
    
    # Element length
    L = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if L <= 0.0:
        raise ValueError(
            f"Element {element.id} has zero length (nodes {element.ni} and {element.nj} "
            f"at same location: ({ni.x}, {ni.y}, {ni.z}))"
        )
    
    # Direction cosines
    l = dx / L
    m = dy / L
    n = dz / L
    
    return L, l, m, n


def truss3d_global_stiffness(nodes: Dict[int, Node3D], element: Truss3D) -> np.ndarray:
    """
    Compute the 6×6 global stiffness matrix for a 3D truss element.
    
    This is the heart of 3D truss analysis. The stiffness matrix relates
    nodal forces to nodal displacements in global coordinates:
    
        f = ke × d
    
    where:
        f = [fx_i, fy_i, fz_i, fx_j, fy_j, fz_j]^T  (nodal forces)
        d = [ux_i, uy_i, uz_i, ux_j, uy_j, uz_j]^T  (nodal displacements)
    
    The matrix is symmetric and positive semi-definite. It has rank 1
    (only one deformation mode: axial extension/compression).
    
    Parameters:
    -----------
    nodes : Dict[int, Node3D]
        Dictionary mapping node IDs to Node3D objects
        
    element : Truss3D
        The truss element with properties E (modulus) and A (area)
    
    Returns:
    --------
    np.ndarray
        6×6 global stiffness matrix
        DOF order: [ux_i, uy_i, uz_i, ux_j, uy_j, uz_j]
    
    Example:
    --------
    >>> nodes = {0: Node3D(0, 0, 0, 0), 1: Node3D(1, 1, 0, 0)}
    >>> bar = Truss3D(0, 0, 1, E=210e9, A=0.001)
    >>> ke = truss3d_global_stiffness(nodes, bar)
    >>> print(f"Axial stiffness: {ke[0,0]:.2e} N/m")
    
    Notes:
    ------
    The matrix has the structure:
    
        ke = (EA/L) × [ B   -B ]
                      [-B    B ]
    
    where B is the 3×3 direction cosine outer product:
    
        B = [ l²  lm  ln ]
            [ lm  m²  mn ]
            [ ln  mn  n² ]
    
    This comes from the transformation: ke = T^T × k_local × T
    """
    # Get geometry
    L, l, m, n = element_geometry_3d(nodes, element)
    
    # Axial stiffness coefficient
    EA_L = element.E * element.A / L
    
    # Build the 3×3 direction cosine matrix (outer product of [l, m, n])
    # B[i,j] = direction_cosine[i] × direction_cosine[j]
    B = np.array([
        [l*l, l*m, l*n],
        [m*l, m*m, m*n],
        [n*l, n*m, n*n],
    ], dtype=float)
    
    # Build the 6×6 global stiffness matrix
    # Structure: [  B  -B ]
    #            [ -B   B ]
    ke = np.zeros((6, 6), dtype=float)
    
    # Upper-left 3×3: B
    ke[0:3, 0:3] = B
    
    # Upper-right 3×3: -B
    ke[0:3, 3:6] = -B
    
    # Lower-left 3×3: -B
    ke[3:6, 0:3] = -B
    
    # Lower-right 3×3: B
    ke[3:6, 3:6] = B
    
    # Scale by EA/L
    ke *= EA_L
    
    return ke


def truss3d_axial_force(
    nodes: Dict[int, Node3D], 
    element: Truss3D, 
    d_global: np.ndarray,
    dof_per_node: int = 3
) -> float:
    """
    Compute the axial force in a 3D truss element from global displacements.
    
    Sign convention:
    - Positive = tension (element elongated)
    - Negative = compression (element shortened)
    
    Parameters:
    -----------
    nodes : Dict[int, Node3D]
        Dictionary mapping node IDs to Node3D objects
        
    element : Truss3D
        The truss element
        
    d_global : np.ndarray
        Global displacement vector from solver
        
    dof_per_node : int
        Number of DOFs per node (3 for 3D truss)
    
    Returns:
    --------
    float
        Axial force in the element (N)
        Positive = tension, Negative = compression
    
    Example:
    --------
    >>> N = truss3d_axial_force(nodes, bar, d)
    >>> print(f"Axial force: {N/1000:.1f} kN ({'tension' if N > 0 else 'compression'})")
    """
    # Get geometry
    L, l, m, n = element_geometry_3d(nodes, element)
    
    # Extract element displacements from global vector
    # Node i DOFs
    i_base = dof_per_node * element.ni
    ux_i = d_global[i_base + 0]
    uy_i = d_global[i_base + 1]
    uz_i = d_global[i_base + 2]
    
    # Node j DOFs
    j_base = dof_per_node * element.nj
    ux_j = d_global[j_base + 0]
    uy_j = d_global[j_base + 1]
    uz_j = d_global[j_base + 2]
    
    # Compute axial deformation (change in length)
    # This is the projection of relative displacement onto the element axis
    delta_ux = ux_j - ux_i
    delta_uy = uy_j - uy_i
    delta_uz = uz_j - uz_i
    
    # Axial deformation = dot product with element direction
    delta_L = l * delta_ux + m * delta_uy + n * delta_uz
    
    # Axial force = stiffness × deformation
    # N = (EA/L) × ΔL
    N = (element.E * element.A / L) * delta_L
    
    return N

