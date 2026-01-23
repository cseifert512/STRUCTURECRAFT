# mini_branch/kernel/assemble.py
"""Dimension-agnostic global matrix assembly via scatter-add."""

import numpy as np
from typing import List, Tuple


def assemble_global_K(
    ndof: int, 
    contributions: List[Tuple[List[int], np.ndarray]]
) -> np.ndarray:
    """
    Assemble global stiffness matrix from element contributions.
    
    Args:
        ndof: Total DOFs in system
        contributions: List of (dof_map, ke) tuples per element
    
    Returns:
        Global stiffness matrix K, shape (ndof, ndof)
    """
    K = np.zeros((ndof, ndof), dtype=float)
    
    for dof_map, ke in contributions:
        n = len(dof_map)
        assert ke.shape == (n, n), f"ke shape {ke.shape} != dof_map length {n}"
        
        for a in range(n):
            for b in range(n):
                K[dof_map[a], dof_map[b]] += ke[a, b]
    
    return K


def assemble_global_F(
    ndof: int,
    contributions: List[Tuple[List[int], np.ndarray]]
) -> np.ndarray:
    """
    Assemble global load vector from element contributions.
    
    Args:
        ndof: Total DOFs in system
        contributions: List of (dof_map, fe) tuples per element
    
    Returns:
        Global load vector F, shape (ndof,)
    """
    F = np.zeros(ndof, dtype=float)
    
    for dof_map, fe in contributions:
        n = len(dof_map)
        assert fe.shape == (n,), f"fe shape {fe.shape} != dof_map length {n}"
        
        for a in range(n):
            F[dof_map[a]] += fe[a]
    
    return F


def add_nodal_load(
    F: np.ndarray,
    node_id: int,
    load_vector: np.ndarray,
    dof_per_node: int
) -> None:
    """
    Add nodal load to global load vector (in-place).
    
    Args:
        F: Global load vector (modified in-place)
        node_id: Node to apply load to
        load_vector: Load components, shape (dof_per_node,)
        dof_per_node: DOFs per node (3 for 2D frame/3D truss, 6 for 3D frame)
    """
    base = dof_per_node * node_id
    for i, val in enumerate(load_vector):
        F[base + i] += val
