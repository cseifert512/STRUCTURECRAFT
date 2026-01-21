# loads.py - Equivalent nodal loads for distributed loads

import numpy as np
from .model import Node, Frame2D
from .elements import element_geometry, frame2d_transform
from .assembly import dof_index

def frame2d_equiv_nodal_load_udl(L: float, w: float) -> np.ndarray:
    """
    Compute the equivalent nodal load vector for a uniform distributed load (UDL)
    in LOCAL element coordinates.
    
    This function converts a distributed load (w N/m along the element) into
    equivalent point forces and moments at the element's two nodes. This is a
    standard technique in finite element analysis that allows us to handle
    distributed loads using the same point-load solver.
    
    Physical meaning:
    - A uniform load w over length L creates a total force of w×L
    - This total force is split equally between the two end nodes: wL/2 each
    - The distributed nature also creates end moments: ±wL²/12
    - These equivalent loads produce the same nodal displacements as the
      actual distributed load would (for linear beam elements)
    
    Parameters:
    -----------
    L : float
        Element length in meters
        Must be positive and non-zero
    
    w : float
        Uniform distributed load per unit length in local +y direction (N/m)
        Sign convention:
        - w > 0: Load in local +y direction (upward if element is horizontal)
        - w < 0: Load in local -y direction (downward, typical for gravity)
    
    Returns:
    --------
    np.ndarray
        Shape (6,) array representing equivalent nodal loads in local coordinates
        Format: [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
        Where:
        - Fx_i, Fx_j: Axial forces (always 0 for transverse UDL)
        - Fy_i, Fy_j: Transverse forces (wL/2 each)
        - Mz_i, Mz_j: End moments (+wL²/12 and -wL²/12, opposite signs)
    
    Examples:
    --------
    >>> L = 4.0  # 4 meter beam
    >>> w = -1000.0  # 1000 N/m downward
    >>> f = frame2d_equiv_nodal_load_udl(L, w)
    >>> print(f)
    [    0.    -2000.    -1333.33...     0.    -2000.     1333.33...]
    # Each node gets -2000 N (downward), with opposite moments
    """
    # Calculate force components
    total_load = w * L
    force_per_node = total_load / 2.0
    
    # Calculate moment components (fixed-end moment from beam theory)
    moment_magnitude = w * L * L / 12.0
    
    # Build the 6-DOF vector in local coordinates
    # DOF order: [ux_i, uy_i, rz_i, ux_j, uy_j, rz_j]
    f_local = np.array([
        0.0,                    # Fx_i: No axial force
        force_per_node,         # Fy_i: Vertical force at start (wL/2)
        moment_magnitude,       # Mz_i: Moment at start (+wL²/12)
        0.0,                    # Fx_j: No axial force at end
        force_per_node,         # Fy_j: Vertical force at end (wL/2)
        -moment_magnitude       # Mz_j: Moment at end (-wL²/12, opposite)
    ], dtype=float)
    
    return f_local


def assemble_element_loads_global(
    nodes: dict[int, Node], 
    elements: list[Frame2D], 
    element_udls
) -> np.ndarray:
    """
    Assemble equivalent nodal loads from element UDLs into the global force vector.
    
    This function processes each element that has a uniform distributed load (UDL),
    converts it to equivalent nodal loads in local coordinates, transforms those
    to global coordinates, and assembles them into the global force vector F.
    
    This follows the same assembly pattern as assemble_global_K: loop through
    elements, transform to global, map DOFs, and accumulate contributions.
    
    Parameters:
    -----------
    nodes : dict[int, Node]
        Dictionary mapping node IDs to Node objects
        Example: {0: Node(0, 0.0, 0.0), 1: Node(1, 4.0, 0.0)}
    
    elements : list[Frame2D]
        List of frame elements in the model
        Example: [Frame2D(0, 0, 1, E=210e9, A=0.01, I=8e-6), ...]
    
    element_udls : dict[int, float] or list[float or None]
        UDL values for each element. Two formats supported:
        
        Format 1 - Dictionary:
            {element_id: w_value}
            Example: {0: -1000.0, 1: -1000.0}
            Elements not in dict have no UDL (None)
        
        Format 2 - List:
            [w_value for element 0, w_value for element 1, ...]
            Example: [None, -1000.0] means element 0 has no UDL, element 1 has -1000 N/m
            List length should match elements list length (or can be shorter)
    
    Returns:
    --------
    np.ndarray
        Global force vector F of shape (ndof,) with UDL loads assembled
        where ndof = 3 × number_of_nodes
        This vector can be added directly to other force contributions
        (like point loads) before solving
    
    Examples:
    --------
    >>> nodes = {0: Node(0, 0.0, 0.0), 1: Node(1, 4.0, 0.0)}
    >>> elements = [Frame2D(0, 0, 1, E=210e9, A=0.01, I=8e-6)]
    >>> element_udls = {0: -1000.0}  # 1000 N/m downward on element 0
    >>> F = assemble_element_loads_global(nodes, elements, element_udls)
    >>> # F now contains forces at nodes 0 and 1 from the UDL
    """
    # Determine total number of degrees of freedom
    n_nodes = len(nodes)
    ndof = 3 * n_nodes  # 3 DOFs per node: ux, uy, rz
    
    # Initialize global force vector to zeros
    F_global = np.zeros(ndof, dtype=float)
    
    # Loop through each element
    for element in elements:
        # Determine the UDL value for this element
        if isinstance(element_udls, dict):
            # Dictionary format: look up by element.id
            w = element_udls.get(element.id, None)
        
        elif isinstance(element_udls, list):
            # List format: use position in elements list
            try:
                element_idx = elements.index(element)
                if element_idx < len(element_udls):
                    w = element_udls[element_idx]
                else:
                    w = None  # List too short, assume no UDL
            except ValueError:
                # Element not in list (shouldn't happen, but be safe)
                w = None
        
        else:
            # Invalid type
            raise TypeError(
                f"element_udls must be dict or list, got {type(element_udls)}"
            )
        
        # Skip if no UDL for this element
        if w is None:
            continue
        
        # Get element geometry (length, cosine, sine)
        # We need these for coordinate transformation
        L, c, s = element_geometry(nodes, element)
        
        # Calculate local equivalent nodal loads
        # This gives us the 6-DOF vector in element's local coordinates
        f_local = frame2d_equiv_nodal_load_udl(L, w)
        
        # Transform to global coordinates
        # Get transformation matrix from local to global
        T = frame2d_transform(c, s)
        # Transform: T.T transforms from local to global (transpose rotates back)
        f_global_element = T.T @ f_local
        
        # Map element DOFs to global DOFs
        # This is the SAME pattern as in assemble_global_K!
        # We need to know which global DOF corresponds to each element DOF
        dof_map = [
            dof_index(element.ni, 0),  # Node i, ux (horizontal)
            dof_index(element.ni, 1),  # Node i, uy (vertical)
            dof_index(element.ni, 2),  # Node i, rz (rotation)
            dof_index(element.nj, 0),  # Node j, ux (horizontal)
            dof_index(element.nj, 1),  # Node j, uy (vertical)
            dof_index(element.nj, 2),  # Node j, rz (rotation)
        ]
        
        # Add element contributions to global force vector
        # Scatter-add operation: each element DOF adds to its corresponding global DOF
        for local_dof in range(6):  # 6 DOFs per element
            global_dof = dof_map[local_dof]
            F_global[global_dof] += f_global_element[local_dof]
    
    return F_global