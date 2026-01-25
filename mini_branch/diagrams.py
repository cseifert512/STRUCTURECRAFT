# mini_branch/diagrams.py
"""
FORCE DIAGRAM COMPUTATIONS
==========================

This module computes internal force diagrams (N, V, M) and curved deflected 
shapes for 2D frame elements. These are essential for:

1. Engineering visualization - understand how forces flow through structure
2. Design verification - check if stresses are within limits
3. Code compliance - deflection and force checks

KEY CONCEPTS:
-------------
- N (Axial Force): Tension (+) or compression (-) along the member axis
- V (Shear Force): Force perpendicular to member, causes "cutting" action
- M (Bending Moment): Rotational force that causes bending

For a beam with UDL (uniformly distributed load):
- N is constant (no axial load from UDL on beam)
- V varies linearly: V(x) = V_i - w*x
- M varies parabolically: M(x) = M_i + V_i*x - w*x²/2

SIGN CONVENTIONS:
-----------------
- Positive N: Tension (member being pulled apart)
- Positive V: Causes clockwise rotation of element segment
- Positive M: Causes compression on top fiber (sagging)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .model import Node, Frame2D
from .elements import element_geometry, frame2d_local_stiffness, frame2d_transform
from .assembly import dof_index, DOF_PER_NODE
from .loads import frame2d_equiv_nodal_load_udl


@dataclass
class DiagramPoint:
    """A single point on a force diagram."""
    x_local: float      # Position along element (0 to L)
    x_global: float     # Global X coordinate
    y_global: float     # Global Y coordinate
    N: float            # Axial force (N)
    V: float            # Shear force (N)
    M: float            # Bending moment (N·m)


@dataclass
class DeflectedPoint:
    """A point on the deflected shape curve."""
    x: float            # Global X coordinate (deformed)
    y: float            # Global Y coordinate (deformed)


@dataclass
class ElementDiagramData:
    """Complete diagram data for one element."""
    element_id: int
    element_type: str   # "column", "beam", "brace"
    ni: int             # Start node ID
    nj: int             # End node ID
    length: float       # Element length (m)
    points: List[DiagramPoint]
    deflected_shape: List[DeflectedPoint]
    # Summary values
    N_i: float          # Axial at node i
    N_j: float          # Axial at node j
    V_i: float          # Shear at node i
    V_j: float          # Shear at node j
    M_i: float          # Moment at node i
    M_j: float          # Moment at node j
    max_N: float        # Maximum absolute axial
    max_V: float        # Maximum absolute shear
    max_M: float        # Maximum absolute moment


def hermite_shape_functions(xi: float) -> Tuple[float, float, float, float]:
    """
    Compute Hermite cubic shape functions for beam deflection interpolation.
    
    These shape functions interpolate the transverse deflection v(x) along
    a beam element given the nodal displacements and rotations.
    
    Parameters:
    -----------
    xi : float
        Normalized coordinate along element, 0 ≤ xi ≤ 1
        xi = x_local / L
    
    Returns:
    --------
    N1, N2, N3, N4 : float
        Shape function values where:
        v(xi) = N1*v_i + N2*theta_i*L + N3*v_j + N4*theta_j*L
        
        N1: Displacement at node i
        N2: Rotation at node i (scaled by L)
        N3: Displacement at node j
        N4: Rotation at node j (scaled by L)
    """
    # Hermite cubic polynomials
    N1 = 1 - 3*xi**2 + 2*xi**3           # v_i contribution
    N2 = xi - 2*xi**2 + xi**3             # theta_i contribution (needs *L)
    N3 = 3*xi**2 - 2*xi**3                # v_j contribution
    N4 = -xi**2 + xi**3                   # theta_j contribution (needs *L)
    
    return N1, N2, N3, N4


def compute_element_internal_forces(
    nodes: Dict[int, Node],
    element: Frame2D,
    d_global: np.ndarray,
    udl_w: Optional[float] = None,
    n_points: int = 21
) -> Tuple[List[DiagramPoint], np.ndarray]:
    """
    Compute N, V, M at points along a 2D frame element.
    
    This function takes the global displacement vector and computes the
    internal forces at sampled points along the element.
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Node dictionary
    element : Frame2D
        The frame element
    d_global : np.ndarray
        Global displacement vector from solver
    udl_w : float, optional
        Uniformly distributed load on element (N/m)
        Positive = in local +y direction
        Negative = downward (gravity)
    n_points : int
        Number of sample points along element
    
    Returns:
    --------
    points : List[DiagramPoint]
        Force values at each sample point
    f_local : np.ndarray
        Element end forces [Ni, Vi, Mi, Nj, Vj, Mj] in local coords
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
    
    # If element has UDL, subtract equivalent nodal loads to get actual forces
    if udl_w is not None and udl_w != 0:
        f_udl_local = frame2d_equiv_nodal_load_udl(L, udl_w)
        f_local = f_local - f_udl_local
    
    # Extract end forces (local coordinates)
    # f_local = [N_i, V_i, M_i, N_j, V_j, M_j]
    N_i = f_local[0]
    V_i = f_local[1]
    M_i = f_local[2]
    N_j = f_local[3]
    V_j = f_local[4]
    M_j = f_local[5]
    
    # Get node coordinates for global position calculation
    ni = nodes[element.ni]
    nj = nodes[element.nj]
    
    # Sample points along element
    points = []
    w = udl_w if udl_w is not None else 0.0
    
    for i in range(n_points):
        x_local = (i / (n_points - 1)) * L
        
        # Internal forces at this point
        # Axial is constant (no distributed axial load)
        N = -N_i  # Internal axial (sign convention: tension positive)
        
        # Shear varies linearly with UDL: V(x) = V_i - w*x
        V = V_i - w * x_local
        
        # Moment varies parabolically: M(x) = M_i + V_i*x - w*x²/2
        M = M_i + V_i * x_local - w * x_local**2 / 2
        
        # Global coordinates (undeformed)
        t = x_local / L
        x_global = ni.x + t * (nj.x - ni.x)
        y_global = ni.y + t * (nj.y - ni.y)
        
        points.append(DiagramPoint(
            x_local=round(x_local, 6),
            x_global=round(x_global, 6),
            y_global=round(y_global, 6),
            N=round(N, 2),
            V=round(V, 2),
            M=round(M, 2)
        ))
    
    return points, f_local


def compute_curved_deflection(
    nodes: Dict[int, Node],
    element: Frame2D,
    d_global: np.ndarray,
    scale: float = 1.0,
    n_points: int = 21
) -> List[DeflectedPoint]:
    """
    Compute the curved deflected shape of a 2D frame element using Hermite interpolation.
    
    For frame elements with bending, the deflected shape between nodes is curved,
    not straight. This function uses Hermite shape functions to interpolate the
    curved deflection.
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Node dictionary
    element : Frame2D
        The frame element
    d_global : np.ndarray
        Global displacement vector from solver
    scale : float
        Scale factor for visualization (deflections are typically small)
    n_points : int
        Number of sample points along element
    
    Returns:
    --------
    List[DeflectedPoint]
        Points defining the curved deflected shape in global coordinates
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
    
    # Local displacements: [u_i, v_i, theta_i, u_j, v_j, theta_j]
    u_i = d_local[0]      # Axial displacement at i
    v_i = d_local[1]      # Transverse displacement at i
    theta_i = d_local[2]  # Rotation at i
    u_j = d_local[3]      # Axial displacement at j
    v_j = d_local[4]      # Transverse displacement at j
    theta_j = d_local[5]  # Rotation at j
    
    # Get node coordinates
    ni = nodes[element.ni]
    nj = nodes[element.nj]
    
    # Sample points along element
    deflected_points = []
    
    for i in range(n_points):
        xi = i / (n_points - 1)  # Normalized coordinate (0 to 1)
        x_local = xi * L
        
        # Axial displacement (linear interpolation)
        u_local = u_i + xi * (u_j - u_i)
        
        # Transverse displacement (Hermite interpolation)
        N1, N2, N3, N4 = hermite_shape_functions(xi)
        v_local = N1 * v_i + N2 * theta_i * L + N3 * v_j + N4 * theta_j * L
        
        # Transform local deformations to global displacements
        # Local x-axis: along element (direction cosines c, s)
        # Local y-axis: perpendicular (-s, c)
        dx_global = c * u_local - s * v_local
        dy_global = s * u_local + c * v_local
        
        # Undeformed global position
        x_undef = ni.x + xi * (nj.x - ni.x)
        y_undef = ni.y + xi * (nj.y - ni.y)
        
        # Deformed global position (with scale)
        x_def = x_undef + scale * dx_global
        y_def = y_undef + scale * dy_global
        
        deflected_points.append(DeflectedPoint(
            x=round(x_def, 6),
            y=round(y_def, 6)
        ))
    
    return deflected_points


def compute_frame_diagrams(
    nodes: Dict[int, Node],
    elements: List[Frame2D],
    d_global: np.ndarray,
    element_udls: Optional[Dict[int, float]] = None,
    scale: float = 50.0,
    n_points: int = 21
) -> List[ElementDiagramData]:
    """
    Compute complete force diagrams for all elements in a 2D frame.
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Node dictionary
    elements : List[Frame2D]
        List of frame elements
    d_global : np.ndarray
        Global displacement vector from solver
    element_udls : Dict[int, float], optional
        Mapping of element ID to UDL value (N/m)
    scale : float
        Scale factor for deflected shape visualization
    n_points : int
        Number of sample points per element
    
    Returns:
    --------
    List[ElementDiagramData]
        Complete diagram data for each element
    """
    if element_udls is None:
        element_udls = {}
    
    results = []
    
    for element in elements:
        udl_w = element_udls.get(element.id, None)
        L, c, s = element_geometry(nodes, element)
        
        # Compute internal forces
        points, f_local = compute_element_internal_forces(
            nodes, element, d_global, udl_w, n_points
        )
        
        # Compute deflected shape
        deflected = compute_curved_deflection(
            nodes, element, d_global, scale, n_points
        )
        
        # Determine element type based on orientation
        if abs(c) < 0.1:  # Nearly vertical
            elem_type = "column"
        elif abs(s) < 0.1:  # Nearly horizontal
            elem_type = "beam"
        else:
            elem_type = "brace"
        
        # Extract end forces from f_local = [N_i, V_i, M_i, N_j, V_j, M_j]
        N_i = float(f_local[0])
        V_i = float(f_local[1])
        M_i = float(f_local[2])
        N_j = float(f_local[3])
        V_j = float(f_local[4])
        M_j = float(f_local[5])
        
        # Compute max values from all points
        all_N = [abs(p.N) for p in points]
        all_V = [abs(p.V) for p in points]
        all_M = [abs(p.M) for p in points]
        
        results.append(ElementDiagramData(
            element_id=element.id,
            element_type=elem_type,
            ni=element.ni,
            nj=element.nj,
            length=round(L, 4),
            points=points,
            deflected_shape=deflected,
            N_i=round(N_i, 2),
            N_j=round(N_j, 2),
            V_i=round(V_i, 2),
            V_j=round(V_j, 2),
            M_i=round(M_i, 2),
            M_j=round(M_j, 2),
            max_N=round(max(all_N) if all_N else 0, 2),
            max_V=round(max(all_V) if all_V else 0, 2),
            max_M=round(max(all_M) if all_M else 0, 2),
        ))
    
    return results


def get_frame_summary(diagrams: List[ElementDiagramData]) -> Dict:
    """
    Get summary statistics for the entire frame.
    
    Parameters:
    -----------
    diagrams : List[ElementDiagramData]
        Diagram data for all elements
    
    Returns:
    --------
    Dict with summary statistics
    """
    if not diagrams:
        return {
            "max_axial_force": 0,
            "max_shear_force": 0,
            "max_moment": 0,
            "critical_element_N": None,
            "critical_element_V": None,
            "critical_element_M": None,
        }
    
    # Find critical elements
    max_N_elem = max(diagrams, key=lambda d: d.max_N)
    max_V_elem = max(diagrams, key=lambda d: d.max_V)
    max_M_elem = max(diagrams, key=lambda d: d.max_M)
    
    return {
        "max_axial_force": max_N_elem.max_N,
        "max_shear_force": max_V_elem.max_V,
        "max_moment": max_M_elem.max_M,
        "critical_element_N": max_N_elem.element_id,
        "critical_element_V": max_V_elem.element_id,
        "critical_element_M": max_M_elem.element_id,
    }

