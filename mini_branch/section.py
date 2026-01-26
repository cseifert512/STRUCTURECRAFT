# mini_branch/section.py
"""
3D-to-2D SECTION EXTRACTION
===========================

Extract a 2D frame section from a 3D spaceframe by "slicing" at a given
X or Y coordinate. This allows detailed force diagram analysis of a 
cross-section through the structure.

CONCEPT:
--------
Given a 3D spaceframe (canopy) with a grid of nodes:

    Y
    ↑   ●───●───●───●───●
    │   │ ╲ │ ╱ │ ╲ │ ╱ │
    │   ●───●───●───●───●  ← Slice at Y = 4m
    │   │ ╱ │ ╲ │ ╱ │ ╲ │
    │   ●───●───●───●───●
    └──────────────────────→ X

We extract all nodes at Y ≈ 4m and the bars connecting them,
then project to 2D (X, Z) for frame analysis.

TRIBUTARY WIDTH:
----------------
The extracted 2D frame carries load from a "tributary width" - the
strip of area that drains to this section. For a regular grid, this
is typically the spacing between grid lines.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Optional

from .v3d.model import Node3D, Truss3D
from .model import Node, Frame2D
from .catalog import DEFAULT_MATERIAL, TIMBER_SECTIONS


@dataclass
class SectionExtractionResult:
    """Result of extracting a 2D section from 3D structure."""
    nodes_2d: Dict[int, Node]
    elements_2d: List[Frame2D]
    fixed_dofs: List[int]
    tributary_width: float
    slice_axis: str
    slice_value: float
    # Mapping from 2D node IDs to original 3D node IDs
    node_id_map: Dict[int, int]
    # Info about what was extracted
    n_nodes_found: int
    n_bars_found: int


def find_grid_spacing(
    nodes: Dict[int, Node3D],
    axis: Literal['x', 'y']
) -> float:
    """
    Find the grid spacing along an axis by analyzing node positions.
    
    Parameters:
    -----------
    nodes : Dict[int, Node3D]
        3D node dictionary
    axis : 'x' or 'y'
        Which axis to analyze
    
    Returns:
    --------
    float
        Estimated grid spacing
    """
    if len(nodes) < 2:
        return 1.0
    
    # Get unique coordinate values along the axis
    if axis == 'x':
        coords = sorted(set(n.x for n in nodes.values()))
    else:
        coords = sorted(set(n.y for n in nodes.values()))
    
    if len(coords) < 2:
        return 1.0
    
    # Find minimum spacing (grid spacing)
    spacings = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
    return min(spacings) if spacings else 1.0


def find_slice_coordinate(
    nodes: Dict[int, Node3D],
    axis: Literal['x', 'y'],
    normalized_position: float
) -> float:
    """
    Convert a normalized position (0-1) to an actual coordinate value
    that aligns with the grid.
    
    Parameters:
    -----------
    nodes : Dict[int, Node3D]
        3D node dictionary
    axis : 'x' or 'y'
        Slice axis
    normalized_position : float
        Position from 0 (min) to 1 (max)
    
    Returns:
    --------
    float
        Actual coordinate value (snapped to nearest grid line)
    """
    if axis == 'x':
        coords = sorted(set(n.x for n in nodes.values()))
    else:
        coords = sorted(set(n.y for n in nodes.values()))
    
    if len(coords) == 0:
        return 0.0
    
    if len(coords) == 1:
        return coords[0]
    
    # Map normalized position to index
    index = int(normalized_position * (len(coords) - 1))
    index = max(0, min(index, len(coords) - 1))
    
    return coords[index]


def extract_section(
    nodes_3d: Dict[int, Node3D],
    bars_3d: List[Truss3D],
    support_nodes_3d: List[int],
    slice_axis: Literal['x', 'y'],
    slice_value: float,
    tolerance: float = 0.01,
    E: float = None,
    A: float = None,
    I: float = None,
) -> SectionExtractionResult:
    """
    Extract a 2D frame section from a 3D spaceframe.
    
    Parameters:
    -----------
    nodes_3d : Dict[int, Node3D]
        3D node dictionary {id: Node3D(id, x, y, z)}
    bars_3d : List[Truss3D]
        List of 3D truss elements
    support_nodes_3d : List[int]
        IDs of supported nodes in 3D model
    slice_axis : 'x' or 'y'
        Which axis to slice along:
        - 'y': Slice at constant Y (extract X-Z section)
        - 'x': Slice at constant X (extract Y-Z section)
    slice_value : float
        Coordinate value at which to slice
    tolerance : float
        Distance tolerance for finding nodes on the slice plane
    E : float, optional
        Young's modulus for 2D elements (default: timber)
    A : float, optional
        Cross-section area (default: from timber catalog)
    I : float, optional
        Moment of inertia (default: from timber catalog)
    
    Returns:
    --------
    SectionExtractionResult
        Contains 2D nodes, elements, supports, and tributary width
    """
    # Default material properties
    if E is None:
        E = DEFAULT_MATERIAL.E
    
    # Default section properties (mid-range timber section)
    if A is None or I is None:
        section = TIMBER_SECTIONS[5]  # 4x8 or similar
        if A is None:
            A = section.A
        if I is None:
            I = section.I
    
    # Step 1: Find nodes on the slice plane
    nodes_on_slice = {}
    for nid, node in nodes_3d.items():
        if slice_axis == 'y':
            if abs(node.y - slice_value) < tolerance:
                nodes_on_slice[nid] = node
        else:  # slice_axis == 'x'
            if abs(node.x - slice_value) < tolerance:
                nodes_on_slice[nid] = node
    
    if len(nodes_on_slice) == 0:
        raise ValueError(f"No nodes found at {slice_axis}={slice_value} (tolerance={tolerance})")
    
    # Step 2: Find bars that connect nodes on the slice
    bars_on_slice = []
    for bar in bars_3d:
        if bar.ni in nodes_on_slice and bar.nj in nodes_on_slice:
            bars_on_slice.append(bar)
    
    # Step 3: Project to 2D coordinates
    # For Y-slice: (X, Y, Z) -> (X, Z) where X becomes the horizontal, Z becomes vertical
    # For X-slice: (X, Y, Z) -> (Y, Z) where Y becomes the horizontal, Z becomes vertical
    nodes_2d = {}
    node_id_map = {}  # 2D id -> 3D id
    new_id = 0
    
    # Sort nodes by their position along the slice for consistent ordering
    if slice_axis == 'y':
        sorted_nodes = sorted(nodes_on_slice.items(), key=lambda x: (x[1].x, x[1].z))
    else:
        sorted_nodes = sorted(nodes_on_slice.items(), key=lambda x: (x[1].y, x[1].z))
    
    old_to_new_id = {}
    for old_id, node3d in sorted_nodes:
        if slice_axis == 'y':
            # Project (X, Y, Z) -> (X, Z)
            nodes_2d[new_id] = Node(id=new_id, x=node3d.x, y=node3d.z)
        else:
            # Project (X, Y, Z) -> (Y, Z)
            nodes_2d[new_id] = Node(id=new_id, x=node3d.y, y=node3d.z)
        
        node_id_map[new_id] = old_id
        old_to_new_id[old_id] = new_id
        new_id += 1
    
    # Step 4: Create 2D frame elements
    elements_2d = []
    elem_id = 0
    for bar in bars_on_slice:
        ni_new = old_to_new_id[bar.ni]
        nj_new = old_to_new_id[bar.nj]
        
        elements_2d.append(Frame2D(
            id=elem_id,
            ni=ni_new,
            nj=nj_new,
            E=E,
            A=A,
            I=I,
        ))
        elem_id += 1
    
    # Step 5: Determine fixed DOFs (supports)
    # Find which 2D nodes correspond to 3D support nodes
    fixed_dofs = []
    DOF_PER_NODE = 3  # ux, uy, rz for 2D frame
    
    for new_id, old_id in node_id_map.items():
        if old_id in support_nodes_3d:
            # Fix all DOFs at support (pinned support with rotation free could be different)
            fixed_dofs.append(new_id * DOF_PER_NODE + 0)  # ux
            fixed_dofs.append(new_id * DOF_PER_NODE + 1)  # uy
            # Don't fix rotation - let it be a pin
    
    # Step 6: Calculate tributary width
    tributary_width = find_grid_spacing(nodes_3d, slice_axis)
    
    return SectionExtractionResult(
        nodes_2d=nodes_2d,
        elements_2d=elements_2d,
        fixed_dofs=fixed_dofs,
        tributary_width=tributary_width,
        slice_axis=slice_axis,
        slice_value=slice_value,
        node_id_map=node_id_map,
        n_nodes_found=len(nodes_on_slice),
        n_bars_found=len(bars_on_slice),
    )


def calculate_tributary_udl(
    total_gravity_load: float,
    footprint_area: float,
    tributary_width: float,
    section_span: float,
) -> float:
    """
    Calculate the UDL (uniformly distributed load) for a 2D section
    based on tributary area.
    
    Parameters:
    -----------
    total_gravity_load : float
        Total gravity load on the structure (N)
    footprint_area : float
        Total footprint area of 3D structure (m²)
    tributary_width : float
        Width of tributary strip for this section (m)
    section_span : float
        Span of the extracted 2D section (m)
    
    Returns:
    --------
    float
        UDL in N/m (load per unit length of section)
    """
    if footprint_area <= 0 or section_span <= 0:
        return 0.0
    
    # Pressure = total load / area (N/m²)
    pressure = total_gravity_load / footprint_area
    
    # UDL = pressure × tributary width (N/m)
    udl = pressure * tributary_width
    
    return udl


def get_section_span(nodes_2d: Dict[int, Node]) -> float:
    """Get the horizontal span of a 2D section."""
    if len(nodes_2d) < 2:
        return 0.0
    
    x_coords = [n.x for n in nodes_2d.values()]
    return max(x_coords) - min(x_coords)


def get_section_height(nodes_2d: Dict[int, Node]) -> float:
    """Get the maximum height of a 2D section."""
    if len(nodes_2d) == 0:
        return 0.0
    
    y_coords = [n.y for n in nodes_2d.values()]
    return max(y_coords) - min(y_coords)

