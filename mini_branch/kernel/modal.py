# mini_branch/kernel/modal.py
"""Modal analysis: mass matrix assembly and natural frequency computation."""

import numpy as np
from scipy.linalg import eigh
from typing import List, Tuple, Dict, Any


def build_lumped_mass_matrix(
    nodes: dict,
    bars: list,
    density: float,
    dof_manager: Any,
    element_geometry_func: callable
) -> np.ndarray:
    """
    Build global lumped mass matrix from element contributions.
    
    Uses lumped mass formulation: half of each element's mass goes to each node.
    For truss: M_node = (ρ * A * L) / 2
    
    Args:
        nodes: Node dictionary {id: Node}
        bars: List of bar elements
        density: Material density (kg/m³), e.g. steel=7850, timber=500
        dof_manager: DOFManager instance
        element_geometry_func: Function(nodes, bar) -> (L, l, m, n)
    
    Returns:
        M: Global mass matrix (ndof x ndof), diagonal
    """
    n_nodes = len(nodes)
    ndof = dof_manager.ndof(n_nodes)
    dof_per_node = dof_manager.dof_per_node
    
    # Diagonal lumped mass matrix
    M = np.zeros((ndof, ndof))
    
    for bar in bars:
        L, _, _, _ = element_geometry_func(nodes, bar)
        
        # Element mass
        element_mass = density * bar.A * L
        node_mass = element_mass / 2.0  # Half to each node
        
        # Add mass to diagonal entries for both nodes
        for node_id in [bar.ni, bar.nj]:
            for d in range(dof_per_node):
                dof_idx = dof_manager.idx(node_id, d)
                M[dof_idx, dof_idx] += node_mass
    
    return M


def build_consistent_mass_matrix(
    nodes: dict,
    bars: list,
    density: float,
    dof_manager: Any,
    element_geometry_func: callable
) -> np.ndarray:
    """
    Build global consistent mass matrix from element contributions.
    
    Consistent mass matrix provides better accuracy than lumped for higher modes.
    For 3D truss element: M_e = (ρAL/6) * [[2, 1], [1, 2]] per DOF direction.
    
    Args:
        nodes: Node dictionary {id: Node}
        bars: List of bar elements
        density: Material density (kg/m³)
        dof_manager: DOFManager instance
        element_geometry_func: Function(nodes, bar) -> (L, l, m, n)
    
    Returns:
        M: Global consistent mass matrix (ndof x ndof)
    """
    n_nodes = len(nodes)
    ndof = dof_manager.ndof(n_nodes)
    dof_per_node = dof_manager.dof_per_node
    
    M = np.zeros((ndof, ndof))
    
    for bar in bars:
        L, _, _, _ = element_geometry_func(nodes, bar)
        
        # Mass per unit length
        rhoAL = density * bar.A * L
        
        # Element DOF map
        dof_map = dof_manager.element_dof_map([bar.ni, bar.nj])
        
        # Build consistent element mass matrix
        # For 3D truss with 6 DOFs per element (3 per node)
        me = np.zeros((2 * dof_per_node, 2 * dof_per_node))
        
        for d in range(dof_per_node):
            i1 = d           # Node i, direction d
            i2 = dof_per_node + d  # Node j, direction d
            
            me[i1, i1] = 2.0
            me[i2, i2] = 2.0
            me[i1, i2] = 1.0
            me[i2, i1] = 1.0
        
        me *= rhoAL / 6.0
        
        # Scatter to global
        for ii, gi in enumerate(dof_map):
            for jj, gj in enumerate(dof_map):
                M[gi, gj] += me[ii, jj]
    
    return M


def natural_frequencies(
    K: np.ndarray,
    M: np.ndarray,
    fixed_dofs: List[int],
    n_modes: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute natural frequencies and mode shapes.
    
    Solves the generalized eigenvalue problem: K·φ = ω²·M·φ
    
    Args:
        K: Global stiffness matrix
        M: Global mass matrix
        fixed_dofs: Constrained DOF indices
        n_modes: Number of modes to return
    
    Returns:
        frequencies_hz: Natural frequencies in Hz, sorted ascending
        mode_shapes: Mode shape matrix (n_free_dofs x n_modes)
    
    Raises:
        ValueError: If matrices are incompatible or no free DOFs
    """
    ndof = K.shape[0]
    
    # Partition to free DOFs
    fixed = set(fixed_dofs)
    free = np.array([i for i in range(ndof) if i not in fixed], dtype=int)
    
    if len(free) == 0:
        raise ValueError("No free DOFs - cannot compute modes")
    
    # Extract reduced matrices
    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]
    
    # Check mass matrix is positive
    M_diag = np.diag(Mff)
    if np.any(M_diag <= 0):
        raise ValueError("Mass matrix has non-positive diagonal entries")
    
    # Solve generalized eigenvalue problem
    # eigh returns eigenvalues in ascending order
    n_free = len(free)
    n_actual = min(n_modes, n_free)
    
    try:
        eigenvalues, eigenvectors = eigh(Kff, Mff, subset_by_index=[0, n_actual - 1])
    except Exception as e:
        raise ValueError(f"Eigenvalue solve failed: {e}")
    
    # Convert to frequencies (rad/s -> Hz)
    # ω² = eigenvalue, f = ω / (2π)
    omega = np.sqrt(np.maximum(eigenvalues, 0))  # Clamp negatives to 0
    frequencies_hz = omega / (2.0 * np.pi)
    
    return frequencies_hz, eigenvectors


def modal_participation_factors(
    mode_shapes: np.ndarray,
    M: np.ndarray,
    free_dofs: np.ndarray,
    direction: int = 2  # Z direction for gravity
) -> np.ndarray:
    """
    Compute modal participation factors for a given direction.
    
    Measures how much each mode participates in response to uniform excitation.
    Higher factors indicate modes that are more important for seismic/dynamic response.
    
    Args:
        mode_shapes: Mode shape matrix from natural_frequencies()
        M: Full mass matrix
        free_dofs: Array of free DOF indices
        direction: DOF direction (0=X, 1=Y, 2=Z)
    
    Returns:
        participation: Participation factor for each mode
    """
    n_modes = mode_shapes.shape[1]
    participation = np.zeros(n_modes)
    
    # Extract reduced mass matrix
    Mff = M[np.ix_(free_dofs, free_dofs)]
    
    # Influence vector (unit acceleration in specified direction)
    # Need to identify which DOFs correspond to the direction
    dof_per_node = 3  # Assume 3D
    r = np.zeros(len(free_dofs))
    
    for i, dof_idx in enumerate(free_dofs):
        if dof_idx % dof_per_node == direction:
            r[i] = 1.0
    
    for mode in range(n_modes):
        phi = mode_shapes[:, mode]
        
        # Generalized mass
        m_star = phi @ Mff @ phi
        
        # Participation factor
        if m_star > 0:
            L = phi @ Mff @ r
            participation[mode] = L / m_star
    
    return participation


def effective_modal_mass(
    mode_shapes: np.ndarray,
    M: np.ndarray,
    free_dofs: np.ndarray,
    direction: int = 2
) -> np.ndarray:
    """
    Compute effective modal mass for each mode.
    
    Effective mass indicates what fraction of total mass is mobilized by each mode.
    Sum of all effective masses equals total mass in that direction.
    
    Args:
        mode_shapes: Mode shape matrix
        M: Full mass matrix
        free_dofs: Array of free DOF indices
        direction: DOF direction
    
    Returns:
        eff_mass: Effective mass for each mode
    """
    n_modes = mode_shapes.shape[1]
    eff_mass = np.zeros(n_modes)
    
    Mff = M[np.ix_(free_dofs, free_dofs)]
    
    # Influence vector
    dof_per_node = 3
    r = np.zeros(len(free_dofs))
    for i, dof_idx in enumerate(free_dofs):
        if dof_idx % dof_per_node == direction:
            r[i] = 1.0
    
    for mode in range(n_modes):
        phi = mode_shapes[:, mode]
        
        m_star = phi @ Mff @ phi
        L = phi @ Mff @ r
        
        if m_star > 0:
            eff_mass[mode] = L**2 / m_star
    
    return eff_mass


