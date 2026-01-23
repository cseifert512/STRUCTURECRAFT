# mini_branch/v3d/elements.py
"""3D truss element stiffness matrix and force recovery."""

import numpy as np
from typing import Dict, Tuple
from .model import Node3D, Truss3D


def element_geometry_3d(nodes: Dict[int, Node3D], element: Truss3D) -> Tuple[float, float, float, float]:
    """
    Compute length and direction cosines for a 3D truss element.
    
    Args:
        nodes: Node dictionary {id: Node3D}
        element: Truss3D element
    
    Returns:
        (L, l, m, n): Length and direction cosines with x, y, z axes
    
    Raises:
        ValueError: If element has zero length
    """
    ni, nj = nodes[element.ni], nodes[element.nj]
    dx, dy, dz = nj.x - ni.x, nj.y - ni.y, nj.z - ni.z
    L = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if L <= 0.0:
        raise ValueError(f"Element {element.id} has zero length")
    
    return L, dx/L, dy/L, dz/L


def truss3d_global_stiffness(nodes: Dict[int, Node3D], element: Truss3D) -> np.ndarray:
    """
    Compute 6x6 global stiffness matrix for 3D truss element.
    
    DOF order: [ux_i, uy_i, uz_i, ux_j, uy_j, uz_j]
    
    Args:
        nodes: Node dictionary {id: Node3D}
        element: Truss3D element with E and A properties
    
    Returns:
        6x6 global stiffness matrix
    """
    L, l, m, n = element_geometry_3d(nodes, element)
    EA_L = element.E * element.A / L
    
    # Direction cosine outer product: B[i,j] = dir[i] * dir[j]
    B = np.array([
        [l*l, l*m, l*n],
        [m*l, m*m, m*n],
        [n*l, n*m, n*n],
    ], dtype=float)
    
    # 6x6 stiffness: [B, -B; -B, B] * EA/L
    ke = np.zeros((6, 6), dtype=float)
    ke[0:3, 0:3] = B
    ke[0:3, 3:6] = -B
    ke[3:6, 0:3] = -B
    ke[3:6, 3:6] = B
    ke *= EA_L
    
    return ke


def truss3d_axial_force(
    nodes: Dict[int, Node3D], 
    element: Truss3D, 
    d_global: np.ndarray,
    dof_per_node: int = 3
) -> float:
    """
    Compute axial force in 3D truss element from global displacements.
    
    Sign: positive = tension, negative = compression
    
    Args:
        nodes: Node dictionary
        element: Truss3D element
        d_global: Global displacement vector
        dof_per_node: DOFs per node (default 3)
    
    Returns:
        Axial force N (positive = tension)
    """
    L, l, m, n = element_geometry_3d(nodes, element)
    
    # Extract nodal displacements
    i_base = dof_per_node * element.ni
    j_base = dof_per_node * element.nj
    
    delta_u = d_global[j_base:j_base+3] - d_global[i_base:i_base+3]
    
    # Axial deformation = dot product with element direction
    delta_L = l * delta_u[0] + m * delta_u[1] + n * delta_u[2]
    
    # N = EA/L * ΔL
    return (element.E * element.A / L) * delta_L
