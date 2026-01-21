# element end forces, diagrams, utilization, drift

import numpy as np
from .model import Node, Frame2D
from .elements import element_geometry, frame2d_local_stiffness, frame2d_transform
from .assembly import dof_index, DOF_PER_NODE
from .loads import frame2d_equiv_nodal_load_udl


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