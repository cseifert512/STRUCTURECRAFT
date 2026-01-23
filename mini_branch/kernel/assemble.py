# mini_branch/kernel/assemble.py
"""
ASSEMBLY: Dimension-Agnostic Global Matrix Assembly
===================================================

PURPOSE:
--------
This module handles the assembly of element contributions into global matrices.
This is the scatter-add operation that builds K and F from element-level data.

The key insight: assembly doesn't care about element TYPE.
It just needs:
- Total number of DOFs
- For each element: its DOF map and its stiffness matrix

Whether the element is a 2D frame (6×6 ke) or 3D truss (6×6 ke) or 
3D frame (12×12 ke), the assembly logic is identical.

USAGE:
------
    # Compute element contributions (element-specific code does this)
    contributions = []
    for element in elements:
        dof_map = [...]  # Global DOF indices for this element
        ke = element.ke_global(nodes)  # Element stiffness in global coords
        contributions.append((dof_map, ke))
    
    # Assemble (dimension-agnostic)
    K = assemble_global_K(ndof, contributions)
"""

import numpy as np
from typing import List, Tuple


def assemble_global_K(
    ndof: int, 
    contributions: List[Tuple[List[int], np.ndarray]]
) -> np.ndarray:
    """
    Assemble global stiffness matrix from element contributions.
    
    This is the dimension-agnostic scatter-add operation:
    - For each element, add its stiffness matrix entries to global K
    - The DOF map tells us WHERE in K each entry goes
    
    Works for ANY element type (2D frame, 3D truss, 3D frame, etc.)
    as long as you provide the correct DOF maps and ke matrices.
    
    ALGORITHM:
    ----------
    K = zeros(ndof × ndof)
    for each element:
        for each (local_i, local_j) in element ke:
            global_i = dof_map[local_i]
            global_j = dof_map[local_j]
            K[global_i, global_j] += ke[local_i, local_j]
    
    Parameters:
    -----------
    ndof : int
        Total number of DOFs in the system
        - 2D frame: 3 × n_nodes
        - 3D truss: 3 × n_nodes
        - 3D frame: 6 × n_nodes
        
    contributions : List[Tuple[List[int], np.ndarray]]
        List of (dof_map, ke) tuples, one per element:
        - dof_map: List of global DOF indices for this element
          e.g., [0, 1, 2, 3, 4, 5] for a 2-node element with 3 DOF/node
        - ke: Element stiffness matrix in global coordinates
          Shape must be (len(dof_map), len(dof_map))
    
    Returns:
    --------
    np.ndarray
        Global stiffness matrix K, shape (ndof, ndof)
        Symmetric positive semi-definite (becomes PD after BCs applied)
    
    Example (2D Frame):
    -------------------
    >>> dof = DOFManager(dof_per_node=3)
    >>> ndof = dof.ndof(4)  # 4 nodes → 12 DOFs
    >>> 
    >>> contributions = []
    >>> for element in elements:
    ...     dof_map = dof.element_dof_map([element.ni, element.nj])
    ...     ke = frame2d_global_stiffness(nodes, element)
    ...     contributions.append((dof_map, ke))
    >>> 
    >>> K = assemble_global_K(ndof, contributions)
    
    Example (3D Truss):
    -------------------
    >>> dof = DOFManager(dof_per_node=3)  # Same as 2D! ux, uy, uz
    >>> ndof = dof.ndof(10)  # 10 nodes → 30 DOFs
    >>> 
    >>> contributions = []
    >>> for bar in bars:
    ...     dof_map = dof.element_dof_map([bar.ni, bar.nj])
    ...     ke = truss3d_global_stiffness(nodes, bar)
    ...     contributions.append((dof_map, ke))
    >>> 
    >>> K = assemble_global_K(ndof, contributions)
    """
    # Initialize global stiffness matrix to zeros
    K = np.zeros((ndof, ndof), dtype=float)
    
    # Scatter-add each element's contribution
    for dof_map, ke in contributions:
        n_element_dofs = len(dof_map)
        
        # Sanity check: ke must match dof_map size
        assert ke.shape == (n_element_dofs, n_element_dofs), \
            f"Element ke shape {ke.shape} doesn't match dof_map length {n_element_dofs}"
        
        # Add element stiffness to global positions
        for a in range(n_element_dofs):
            ia = dof_map[a]
            for b in range(n_element_dofs):
                ib = dof_map[b]
                K[ia, ib] += ke[a, b]
    
    return K


def assemble_global_F(
    ndof: int,
    contributions: List[Tuple[List[int], np.ndarray]]
) -> np.ndarray:
    """
    Assemble global load vector from element contributions.
    
    Same scatter-add logic as assemble_global_K, but for load vectors.
    Used for equivalent nodal loads from distributed loads.
    
    Parameters:
    -----------
    ndof : int
        Total number of DOFs in the system
        
    contributions : List[Tuple[List[int], np.ndarray]]
        List of (dof_map, fe) tuples, one per element with loads:
        - dof_map: List of global DOF indices
        - fe: Element load vector in global coordinates, shape (len(dof_map),)
    
    Returns:
    --------
    np.ndarray
        Global load vector F, shape (ndof,)
    
    Example:
    --------
    >>> contributions = []
    >>> for element in elements:
    ...     if element.has_udl:
    ...         dof_map = dof.element_dof_map([element.ni, element.nj])
    ...         fe = compute_equiv_nodal_load(element)  # in global coords
    ...         contributions.append((dof_map, fe))
    >>> 
    >>> F = assemble_global_F(ndof, contributions)
    """
    F = np.zeros(ndof, dtype=float)
    
    for dof_map, fe in contributions:
        n_element_dofs = len(dof_map)
        
        assert fe.shape == (n_element_dofs,), \
            f"Element fe shape {fe.shape} doesn't match dof_map length {n_element_dofs}"
        
        for a in range(n_element_dofs):
            ia = dof_map[a]
            F[ia] += fe[a]
    
    return F


def add_nodal_load(
    F: np.ndarray,
    node_id: int,
    load_vector: np.ndarray,
    dof_per_node: int
) -> None:
    """
    Add a nodal load to the global load vector (in-place).
    
    Convenience function for adding point loads at nodes.
    
    Parameters:
    -----------
    F : np.ndarray
        Global load vector (modified in-place)
        
    node_id : int
        Node to apply load to
        
    load_vector : np.ndarray
        Load components at the node, shape (dof_per_node,)
        - 2D frame: [Fx, Fy, Mz]
        - 3D truss: [Fx, Fy, Fz]
        - 3D frame: [Fx, Fy, Fz, Mx, My, Mz]
        
    dof_per_node : int
        Number of DOFs per node
    
    Example:
    --------
    >>> F = np.zeros(12)  # 4 nodes, 3 DOF each
    >>> add_nodal_load(F, node_id=1, load_vector=np.array([1000, 0, 0]), dof_per_node=3)
    >>> # Now F[3] = 1000 (horizontal force at node 1)
    """
    base_dof = dof_per_node * node_id
    for i, val in enumerate(load_vector):
        F[base_dof + i] += val

