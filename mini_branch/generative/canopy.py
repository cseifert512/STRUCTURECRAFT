# mini_branch/generative/canopy.py
"""
CANOPY GENERATOR: Parametric Spaceframe Structures
===================================================

PURPOSE:
--------
Generate 3D spaceframe canopy structures from design parameters.
This is the "generative geometry" that turns a design brief into a structure.

ENGINEERING CONTEXT:
--------------------
A spaceframe canopy is a lightweight 3D truss structure that spans large areas.
Common applications:
- Stadium roofs
- Exhibition pavilions
- Airport terminals
- Covered walkways

The generator creates:
1. A grid of nodes over a rectangular footprint
2. A heightfield (flat, curved, peaked) defining the canopy shape
3. Bar connectivity (grid, triangulated, or diagrid patterns)
4. Support conditions at selected nodes
5. Gravity loads distributed to nodes

TOPOLOGIES:
-----------
- 'grid': Orthogonal grid with optional diagonals
- 'diagrid': Diamond/diagonal grid pattern
- 'triangulated': Triangular mesh (Delaunay-like)

HEIGHTFIELDS:
-------------
- 'flat': Constant height (horizontal canopy)
- 'paraboloid': Parabolic dome (bowl shape)
- 'ridge': Peaked along centerline (like a tent)
- 'saddle': Hyperbolic paraboloid (modern aesthetic)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal

from ..v3d.model import Node3D, Truss3D
from ..kernel.dof import DOFManager


@dataclass
class CanopyParams:
    """
    Parameters defining a canopy structure.
    
    Geometry:
    ---------
    width : float
        Footprint width in X direction (meters)
    depth : float
        Footprint depth in Y direction (meters)
    nx : int
        Number of grid divisions in X direction
    ny : int  
        Number of grid divisions in Y direction
    max_height : float
        Maximum height of canopy above ground (meters)
    min_height : float
        Minimum height (at edges for curved surfaces)
    
    Shape:
    ------
    heightfield : str
        Shape of the canopy surface:
        - 'flat': Constant height
        - 'paraboloid': Dome/bowl shape
        - 'ridge': Peaked along X centerline
        - 'saddle': Hyperbolic paraboloid
    
    Topology:
    ---------
    topology : str
        Bar connectivity pattern:
        - 'grid': Orthogonal grid + diagonals in each cell
        - 'diagrid': Diamond pattern (45-degree grid)
        - 'triangulated': Delaunay triangulation
    
    Supports:
    ---------
    support_layout : str
        Where supports are located:
        - 'corners': 4 corner nodes only
        - 'edges': All edge nodes
        - 'perimeter_n': Every nth node on perimeter
    
    Material:
    ---------
    E : float
        Young's modulus (Pa)
    A : float
        Cross-sectional area for all bars (m^2)
    
    Loading:
    --------
    gravity_load : float
        Total gravity load distributed to nodes (N)
        Negative = downward
    """
    # Geometry
    width: float = 10.0
    depth: float = 8.0
    nx: int = 5
    ny: int = 4
    max_height: float = 3.0
    min_height: float = 2.0
    
    # Shape
    heightfield: Literal['flat', 'paraboloid', 'ridge', 'saddle'] = 'paraboloid'
    
    # Topology
    topology: Literal['grid', 'diagrid', 'triangulated'] = 'grid'
    
    # Supports
    support_layout: Literal['corners', 'edges', 'perimeter_4'] = 'corners'
    
    # Material (default: steel)
    E: float = 210e9  # Pa
    A: float = 0.001  # m^2 (10 cm^2)
    
    # Loading
    gravity_load: float = -50000.0  # N total (distributed to nodes)


def _compute_heightfield(
    x: float, 
    y: float, 
    params: CanopyParams
) -> float:
    """
    Compute the Z-coordinate (height) at a given (x, y) position.
    
    The heightfield defines the 3D shape of the canopy surface.
    All heightfields are normalized to fit within [min_height, max_height].
    """
    # Normalize coordinates to [-1, 1] range centered on footprint
    xn = 2 * (x / params.width) - 1  # -1 to 1
    yn = 2 * (y / params.depth) - 1  # -1 to 1
    
    h_range = params.max_height - params.min_height
    
    if params.heightfield == 'flat':
        # Constant height
        return params.max_height
    
    elif params.heightfield == 'paraboloid':
        # Dome shape: highest at center, lower at edges
        # z = max_height - k*(x^2 + y^2)
        r2 = xn**2 + yn**2
        # Normalize so corners (r2=2) hit min_height
        z_normalized = 1 - r2 / 2  # 1 at center, 0 at corners
        return params.min_height + h_range * z_normalized
    
    elif params.heightfield == 'ridge':
        # Peaked along Y centerline (like a tent ridge along X)
        # Highest when yn=0, lowest at yn=+-1
        z_normalized = 1 - abs(yn)
        return params.min_height + h_range * z_normalized
    
    elif params.heightfield == 'saddle':
        # Hyperbolic paraboloid: z = x^2 - y^2
        # High at x-edges, low at y-edges, saddle at center
        z_raw = xn**2 - yn**2  # Range: -1 to 1
        z_normalized = (z_raw + 1) / 2  # Shift to 0-1
        return params.min_height + h_range * z_normalized
    
    else:
        raise ValueError(f"Unknown heightfield: {params.heightfield}")


def _generate_grid_nodes(params: CanopyParams) -> Dict[int, Node3D]:
    """Generate nodes on a regular grid with heightfield applied."""
    nodes = {}
    node_id = 0
    
    for iy in range(params.ny + 1):
        for ix in range(params.nx + 1):
            x = ix * params.width / params.nx
            y = iy * params.depth / params.ny
            z = _compute_heightfield(x, y, params)
            
            nodes[node_id] = Node3D(id=node_id, x=x, y=y, z=z)
            node_id += 1
    
    return nodes


def _node_index(ix: int, iy: int, nx: int) -> int:
    """Convert grid indices to node ID."""
    return iy * (nx + 1) + ix


def _generate_grid_bars(
    params: CanopyParams, 
    nodes: Dict[int, Node3D]
) -> List[Truss3D]:
    """
    Generate bars for 'grid' topology.
    
    Creates:
    - Horizontal bars along X
    - Horizontal bars along Y
    - Diagonal bars in each cell (both directions)
    """
    bars = []
    bar_id = 0
    nx, ny = params.nx, params.ny
    
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            curr = _node_index(ix, iy, nx)
            
            # Bar to right (X direction)
            if ix < nx:
                right = _node_index(ix + 1, iy, nx)
                bars.append(Truss3D(id=bar_id, ni=curr, nj=right, E=params.E, A=params.A))
                bar_id += 1
            
            # Bar upward (Y direction)
            if iy < ny:
                up = _node_index(ix, iy + 1, nx)
                bars.append(Truss3D(id=bar_id, ni=curr, nj=up, E=params.E, A=params.A))
                bar_id += 1
            
            # Diagonals in cell (ix, iy) to (ix+1, iy+1)
            if ix < nx and iy < ny:
                # Diagonal: bottom-left to top-right
                top_right = _node_index(ix + 1, iy + 1, nx)
                bars.append(Truss3D(id=bar_id, ni=curr, nj=top_right, E=params.E, A=params.A))
                bar_id += 1
                
                # Diagonal: bottom-right to top-left
                bottom_right = _node_index(ix + 1, iy, nx)
                top_left = _node_index(ix, iy + 1, nx)
                bars.append(Truss3D(id=bar_id, ni=bottom_right, nj=top_left, E=params.E, A=params.A))
                bar_id += 1
    
    return bars


def _generate_diagrid_bars(
    params: CanopyParams,
    nodes: Dict[int, Node3D]
) -> List[Truss3D]:
    """
    Generate bars for 'diagrid' topology.
    
    Creates a diamond/diagonal grid pattern - only diagonal bars,
    no orthogonal bars. This creates a distinctive aesthetic.
    """
    bars = []
    bar_id = 0
    nx, ny = params.nx, params.ny
    
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            curr = _node_index(ix, iy, nx)
            
            # Only diagonal connections
            # Diagonal to top-right
            if ix < nx and iy < ny:
                top_right = _node_index(ix + 1, iy + 1, nx)
                bars.append(Truss3D(id=bar_id, ni=curr, nj=top_right, E=params.E, A=params.A))
                bar_id += 1
            
            # Diagonal to top-left
            if ix > 0 and iy < ny:
                top_left = _node_index(ix - 1, iy + 1, nx)
                bars.append(Truss3D(id=bar_id, ni=curr, nj=top_left, E=params.E, A=params.A))
                bar_id += 1
    
    # Add edge bars to close the structure
    # Bottom edge
    for ix in range(nx):
        curr = _node_index(ix, 0, nx)
        right = _node_index(ix + 1, 0, nx)
        bars.append(Truss3D(id=bar_id, ni=curr, nj=right, E=params.E, A=params.A))
        bar_id += 1
    
    # Top edge
    for ix in range(nx):
        curr = _node_index(ix, ny, nx)
        right = _node_index(ix + 1, ny, nx)
        bars.append(Truss3D(id=bar_id, ni=curr, nj=right, E=params.E, A=params.A))
        bar_id += 1
    
    # Left edge
    for iy in range(ny):
        curr = _node_index(0, iy, nx)
        up = _node_index(0, iy + 1, nx)
        bars.append(Truss3D(id=bar_id, ni=curr, nj=up, E=params.E, A=params.A))
        bar_id += 1
    
    # Right edge
    for iy in range(ny):
        curr = _node_index(nx, iy, nx)
        up = _node_index(nx, iy + 1, nx)
        bars.append(Truss3D(id=bar_id, ni=curr, nj=up, E=params.E, A=params.A))
        bar_id += 1
    
    return bars


def _generate_triangulated_bars(
    params: CanopyParams,
    nodes: Dict[int, Node3D]
) -> List[Truss3D]:
    """
    Generate bars for 'triangulated' topology.
    
    Creates a triangular mesh by adding one diagonal per cell
    (alternating direction to create a regular pattern).
    """
    bars = []
    bar_id = 0
    nx, ny = params.nx, params.ny
    
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            curr = _node_index(ix, iy, nx)
            
            # Bar to right (X direction)
            if ix < nx:
                right = _node_index(ix + 1, iy, nx)
                bars.append(Truss3D(id=bar_id, ni=curr, nj=right, E=params.E, A=params.A))
                bar_id += 1
            
            # Bar upward (Y direction)
            if iy < ny:
                up = _node_index(ix, iy + 1, nx)
                bars.append(Truss3D(id=bar_id, ni=curr, nj=up, E=params.E, A=params.A))
                bar_id += 1
            
            # One diagonal per cell (alternating pattern)
            if ix < nx and iy < ny:
                if (ix + iy) % 2 == 0:
                    # Diagonal: bottom-left to top-right
                    top_right = _node_index(ix + 1, iy + 1, nx)
                    bars.append(Truss3D(id=bar_id, ni=curr, nj=top_right, E=params.E, A=params.A))
                else:
                    # Diagonal: bottom-right to top-left
                    bottom_right = _node_index(ix + 1, iy, nx)
                    top_left = _node_index(ix, iy + 1, nx)
                    bars.append(Truss3D(id=bar_id, ni=bottom_right, nj=top_left, E=params.E, A=params.A))
                bar_id += 1
    
    return bars


def _get_support_nodes(params: CanopyParams) -> List[int]:
    """Get node IDs that should be supported based on support_layout."""
    nx, ny = params.nx, params.ny
    
    if params.support_layout == 'corners':
        # Four corner nodes
        return [
            _node_index(0, 0, nx),      # Bottom-left
            _node_index(nx, 0, nx),     # Bottom-right
            _node_index(0, ny, nx),     # Top-left
            _node_index(nx, ny, nx),    # Top-right
        ]
    
    elif params.support_layout == 'edges':
        # All edge nodes
        edge_nodes = set()
        
        # Bottom and top edges
        for ix in range(nx + 1):
            edge_nodes.add(_node_index(ix, 0, nx))
            edge_nodes.add(_node_index(ix, ny, nx))
        
        # Left and right edges
        for iy in range(ny + 1):
            edge_nodes.add(_node_index(0, iy, nx))
            edge_nodes.add(_node_index(nx, iy, nx))
        
        return sorted(edge_nodes)
    
    elif params.support_layout == 'perimeter_4':
        # Every 4th node on perimeter (for larger canopies)
        edge_nodes = []
        
        # Walk around perimeter
        # Bottom edge (left to right)
        for ix in range(0, nx + 1, 4):
            edge_nodes.append(_node_index(ix, 0, nx))
        
        # Right edge (bottom to top, skip corner)
        for iy in range(4, ny + 1, 4):
            edge_nodes.append(_node_index(nx, iy, nx))
        
        # Top edge (right to left, skip corner)
        for ix in range(nx - 4, -1, -4):
            edge_nodes.append(_node_index(ix, ny, nx))
        
        # Left edge (top to bottom, skip corners)
        for iy in range(ny - 4, 0, -4):
            edge_nodes.append(_node_index(0, iy, nx))
        
        # Ensure corners are included
        corners = [
            _node_index(0, 0, nx),
            _node_index(nx, 0, nx),
            _node_index(0, ny, nx),
            _node_index(nx, ny, nx),
        ]
        for c in corners:
            if c not in edge_nodes:
                edge_nodes.append(c)
        
        return sorted(set(edge_nodes))
    
    else:
        raise ValueError(f"Unknown support_layout: {params.support_layout}")


def generate_canopy(
    params: CanopyParams
) -> Tuple[Dict[int, Node3D], List[Truss3D], List[int], np.ndarray]:
    """
    Generate a complete canopy structure from parameters.
    
    This is the main entry point for canopy generation. It creates:
    1. Nodes on a grid with heightfield applied
    2. Bars connecting nodes based on topology
    3. Fixed DOF list for support nodes
    4. Load vector with gravity distributed to nodes
    
    Parameters:
    -----------
    params : CanopyParams
        Design parameters for the canopy
    
    Returns:
    --------
    nodes : Dict[int, Node3D]
        Dictionary mapping node IDs to Node3D objects
        
    bars : List[Truss3D]
        List of truss bar elements
        
    fixed_dofs : List[int]
        List of fixed DOF indices (all 3 DOFs at each support node)
        
    F : np.ndarray
        Load vector with gravity distributed to all nodes
    
    Example:
    --------
    >>> params = CanopyParams(width=10, depth=8, nx=5, ny=4)
    >>> nodes, bars, fixed_dofs, F = generate_canopy(params)
    >>> print(f"Generated {len(nodes)} nodes, {len(bars)} bars")
    """
    # Generate nodes
    nodes = _generate_grid_nodes(params)
    n_nodes = len(nodes)
    
    # Generate bars based on topology
    if params.topology == 'grid':
        bars = _generate_grid_bars(params, nodes)
    elif params.topology == 'diagrid':
        bars = _generate_diagrid_bars(params, nodes)
    elif params.topology == 'triangulated':
        bars = _generate_triangulated_bars(params, nodes)
    else:
        raise ValueError(f"Unknown topology: {params.topology}")
    
    # Generate fixed DOFs (pin supports: fix ux, uy, uz)
    dof = DOFManager(dof_per_node=3)
    support_nodes = _get_support_nodes(params)
    fixed_dofs = []
    for node_id in support_nodes:
        fixed_dofs.extend(dof.node_dofs(node_id))
    
    # Generate load vector (gravity distributed to all nodes)
    ndof = dof.ndof(n_nodes)
    F = np.zeros(ndof)
    
    # Distribute gravity load equally to all nodes
    load_per_node = params.gravity_load / n_nodes
    for node_id in nodes:
        F[dof.idx(node_id, 2)] = load_per_node  # Fz (vertical)
    
    return nodes, bars, fixed_dofs, F


def compute_member_lengths(
    nodes: Dict[int, Node3D],
    bars: List[Truss3D]
) -> List[Tuple[int, float]]:
    """
    Compute lengths of all members for cut list.
    
    Returns:
    --------
    List of (bar_id, length) tuples sorted by length
    """
    from ..v3d.elements import element_geometry_3d
    
    lengths = []
    for bar in bars:
        L, _, _, _ = element_geometry_3d(nodes, bar)
        lengths.append((bar.id, L))
    
    return sorted(lengths, key=lambda x: x[1])


def compute_length_bins(
    lengths: List[Tuple[int, float]],
    tolerance: float = 0.005  # 5mm tolerance
) -> Dict[str, List[int]]:
    """
    Group members into length bins for fabrication.
    
    Members within `tolerance` of each other are in the same bin.
    Fewer bins = easier fabrication.
    
    Returns:
    --------
    Dict mapping bin name to list of bar IDs in that bin
    """
    if not lengths:
        return {}
    
    bins = {}
    bin_count = 0
    
    for bar_id, length in lengths:
        # Find existing bin within tolerance
        found_bin = None
        for bin_name, (ref_length, bar_ids) in bins.items():
            if abs(length - ref_length) <= tolerance:
                found_bin = bin_name
                break
        
        if found_bin:
            bins[found_bin][1].append(bar_id)
        else:
            # Create new bin
            bin_name = f"L{bin_count + 1}"
            bins[bin_name] = (length, [bar_id])
            bin_count += 1
    
    # Convert to simpler format
    result = {}
    for bin_name, (ref_length, bar_ids) in bins.items():
        result[f"{bin_name} ({ref_length*1000:.0f}mm)"] = bar_ids
    
    return result

