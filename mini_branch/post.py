# element end forces, diagrams, utilization, drift

import numpy as np
from typing import Dict, List, Tuple, Optional
from .model import Node, Frame2D
from .elements import element_geometry, frame2d_local_stiffness, frame2d_transform
from .assembly import dof_index, DOF_PER_NODE
from .loads import frame2d_equiv_nodal_load_udl

# Re-export diagram functions for convenience
from .diagrams import (
    DiagramPoint,
    DeflectedPoint,
    ElementDiagramData,
    hermite_shape_functions,
    compute_element_internal_forces,
    compute_curved_deflection,
    compute_frame_diagrams,
    get_frame_summary,
)


def element_end_forces_local(
    nodes: dict[int, Node],
    element: Frame2D,
    d_global: np.ndarray,
    udl_w: float = None
) -> np.ndarray:
    """
    Compute element end forces in LOCAL coordinates from global displacements.
    
    This function takes the solved global displacements and computes the
    internal forces (axial N, shear V, moment M) at both ends of an element
    in the element's local coordinate system.
    
    The process:
    1. Extract element's global displacements
    2. Transform to local coordinates
    3. Compute forces using f = k × d
    4. If UDL exists, subtract equivalent nodal loads
    
    Parameters:
    -----------
    nodes : dict[int, Node]
        Dictionary mapping node IDs to Node objects
    
    element : Frame2D
        The element for which to compute end forces
    
    d_global : np.ndarray
        Global displacement vector from solve_linear
        Shape: (ndof,) where ndof = 3 × number_of_nodes
    
    udl_w : float, optional
        Uniform distributed load on this element (N/m)
        If None, element has no UDL
        If provided, equivalent nodal loads are subtracted from computed forces
    
    Returns:
    --------
    np.ndarray
        Shape (6,) array: [Ni, Vi, Mi, Nj, Vj, Mj]
        Forces and moments in LOCAL element coordinates:
        - Ni, Nj: Axial forces (positive = tension)
        - Vi, Vj: Shear forces (positive = upward in local +y)
        - Mi, Mj: Moments (positive = counterclockwise)
    """
    # Get element geometry
    L, c, s = element_geometry(nodes, element)
    
    # Map element DOFs to global DOFs
    dof_map = [
        dof_index(element.ni, 0),  # Node i, ux
        dof_index(element.ni, 1),  # Node i, uy
        dof_index(element.ni, 2),  # Node i, rz
        dof_index(element.nj, 0),  # Node j, ux
        dof_index(element.nj, 1),  # Node j, uy
        dof_index(element.nj, 2),  # Node j, rz
    ]
    
    # Extract element's global displacements
    d_elem_global = np.array([d_global[dof] for dof in dof_map], dtype=float)
    
    # Transform to local coordinates
    T = frame2d_transform(c, s)
    d_local = T @ d_elem_global
    
    # Compute local forces from local displacements
    k_local = frame2d_local_stiffness(element.E, element.A, element.I, L)
    f_local = k_local @ d_local
    
    # If element has UDL, subtract equivalent nodal loads
    # (These were added during load assembly, but we want actual internal forces)
    if udl_w is not None:
        f_udl_local = frame2d_equiv_nodal_load_udl(L, udl_w)
        f_local = f_local - f_udl_local
    
    return f_local


def compute_nodal_displacements(
    nodes: Dict[int, Node],
    d_global: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Extract nodal displacements from global displacement vector.
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Node dictionary
    d_global : np.ndarray
        Global displacement vector from solver
    
    Returns:
    --------
    Dict[int, Dict[str, float]]
        Mapping of node_id to {'ux', 'uy', 'rz', 'magnitude'}
    """
    result = {}
    for node_id in nodes:
        ux = d_global[dof_index(node_id, 0)]
        uy = d_global[dof_index(node_id, 1)]
        rz = d_global[dof_index(node_id, 2)]
        mag = float(np.sqrt(ux**2 + uy**2))
        result[node_id] = {
            'ux': float(ux),
            'uy': float(uy),
            'rz': float(rz),
            'magnitude': mag,
        }
    return result


def compute_drift(
    nodes: Dict[int, Node],
    d_global: np.ndarray,
    height: float,
) -> Dict[str, float]:
    """
    Compute lateral drift (horizontal displacement at top of frame).
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Node dictionary
    d_global : np.ndarray  
        Global displacement vector from solver
    height : float
        Frame height for drift ratio calculation
    
    Returns:
    --------
    Dict with drift metrics:
        - max_drift: Maximum horizontal displacement (m)
        - max_drift_mm: Maximum horizontal displacement (mm)
        - drift_ratio: max_drift / height
        - drift_limit_H400: Allowable drift at H/400
        - drift_passes: Whether drift is within H/400 limit
    """
    # Find top nodes (maximum y coordinate)
    max_y = max(n.y for n in nodes.values())
    top_nodes = [nid for nid, n in nodes.items() if abs(n.y - max_y) < 1e-6]
    
    # Get maximum horizontal displacement at top
    max_drift = 0.0
    for nid in top_nodes:
        ux = abs(d_global[dof_index(nid, 0)])
        if ux > max_drift:
            max_drift = ux
    
    drift_limit = height / 400  # Common code limit
    
    return {
        'max_drift': max_drift,
        'max_drift_mm': max_drift * 1000,
        'drift_ratio': max_drift / height if height > 0 else 0,
        'drift_limit_H400': drift_limit,
        'drift_limit_H400_mm': drift_limit * 1000,
        'drift_passes': max_drift <= drift_limit,
    }


def compute_reactions(
    R: np.ndarray,
    fixed_dofs: List[int],
    nodes: Dict[int, Node],
) -> Dict[int, Dict[str, float]]:
    """
    Extract reaction forces at support nodes.
    
    Parameters:
    -----------
    R : np.ndarray
        Reaction vector from solver
    fixed_dofs : List[int]
        List of fixed DOF indices
    nodes : Dict[int, Node]
        Node dictionary
    
    Returns:
    --------
    Dict[int, Dict[str, float]]
        Mapping of node_id to {'Rx', 'Ry', 'Mz'} reaction components
    """
    # Find which nodes have fixed DOFs
    support_nodes = set()
    for dof in fixed_dofs:
        node_id = dof // DOF_PER_NODE
        support_nodes.add(node_id)
    
    result = {}
    for node_id in support_nodes:
        Rx = R[dof_index(node_id, 0)] if dof_index(node_id, 0) in fixed_dofs else 0.0
        Ry = R[dof_index(node_id, 1)] if dof_index(node_id, 1) in fixed_dofs else 0.0
        Mz = R[dof_index(node_id, 2)] if dof_index(node_id, 2) in fixed_dofs else 0.0
        result[node_id] = {
            'Rx': float(Rx),
            'Ry': float(Ry),
            'Mz': float(Mz),
        }
    
    return result