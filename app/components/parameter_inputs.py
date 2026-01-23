# app/components/parameter_inputs.py
"""
Parameter input components for the design brief.
"""

import streamlit as st
from typing import Tuple, List, Optional
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG


def check_stability_warning(
    topology: str,
    support_layout: str,
    nx: int,
    ny: int,
) -> Optional[str]:
    """
    Check if the parameter combination is likely to produce an unstable structure.
    
    Returns a warning message if problematic, None if OK.
    """
    # corners support is very restrictive
    if support_layout == 'corners':
        if topology == 'diagrid':
            return "⚠️ **Unstable combination**: 'corners' support with 'diagrid' topology almost always fails. Use 'edges' support or 'grid' topology."
        if topology == 'triangulated':
            return "⚠️ **Risky combination**: 'corners' support with 'triangulated' may be unstable. Consider 'edges' support or 'grid' topology."
        if nx < 3 or ny < 3:
            return "⚠️ **Unstable**: 'corners' support needs at least 3x3 grid divisions. Increase grid density or use 'edges' support."
    
    # perimeter_4 needs enough nodes to work
    if support_layout == 'perimeter_4':
        if nx < 4 or ny < 4:
            return "⚠️ **Too few supports**: 'perimeter_4' needs at least 4x4 grid. Use 'edges' support for smaller grids."
    
    return None


def get_stable_support_options(topology: str, nx: int, ny: int) -> List[str]:
    """
    Get support layout options that are likely to be stable for the given topology/grid.
    
    Returns list of support layouts, with unstable ones filtered out.
    """
    all_options = CONFIG.support_layouts.copy()
    
    # Filter out known-bad combinations
    if topology == 'diagrid':
        # diagrid + corners is almost always unstable
        if 'corners' in all_options:
            all_options = [s for s in all_options if s != 'corners']
    
    if nx < 3 or ny < 3:
        # Small grids need edge support
        if 'corners' in all_options:
            all_options = [s for s in all_options if s != 'corners']
    
    if nx < 4 or ny < 4:
        # perimeter_4 needs larger grids
        if 'perimeter_4' in all_options:
            all_options = [s for s in all_options if s != 'perimeter_4']
    
    return all_options


def render_geometry_inputs(
    current_width: float,
    current_depth: float,
    current_min_height: float,
    current_max_height: float,
    current_nx: int,
    current_ny: int,
) -> Tuple[float, float, float, float, int, int]:
    """
    Render geometry input controls.
    
    Returns tuple of (width, depth, min_height, max_height, nx, ny)
    """
    st.subheader("Footprint")
    
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Width (m)",
            min_value=CONFIG.width_range[0],
            max_value=CONFIG.width_range[1],
            value=current_width,
            step=0.5,
            help="Footprint width in X direction"
        )
    with col2:
        depth = st.slider(
            "Depth (m)",
            min_value=CONFIG.depth_range[0],
            max_value=CONFIG.depth_range[1],
            value=current_depth,
            step=0.5,
            help="Footprint depth in Y direction"
        )
    
    st.subheader("Height")
    
    col1, col2 = st.columns(2)
    with col1:
        min_height = st.slider(
            "Min Height (m)",
            min_value=CONFIG.height_range[0],
            max_value=CONFIG.height_range[1],
            value=current_min_height,
            step=0.25,
            help="Minimum height at edges"
        )
    with col2:
        max_height = st.slider(
            "Max Height (m)",
            min_value=CONFIG.height_range[0],
            max_value=CONFIG.height_range[1],
            value=max(current_max_height, min_height + 0.5),
            step=0.25,
            help="Maximum height at peak"
        )
    
    # Ensure max > min
    if max_height <= min_height:
        max_height = min_height + 0.5
    
    st.subheader("Grid Density")
    
    col1, col2 = st.columns(2)
    with col1:
        nx = st.slider(
            "Divisions X",
            min_value=CONFIG.nx_range[0],
            max_value=CONFIG.nx_range[1],
            value=current_nx,
            help="Number of grid divisions along width"
        )
    with col2:
        ny = st.slider(
            "Divisions Y",
            min_value=CONFIG.ny_range[0],
            max_value=CONFIG.ny_range[1],
            value=current_ny,
            help="Number of grid divisions along depth"
        )
    
    return width, depth, min_height, max_height, nx, ny


def render_structure_inputs(
    current_heightfield: str,
    current_topology: str,
    current_support_layout: str,
    nx: int = 5,
    ny: int = 4,
) -> Tuple[str, str, str]:
    """
    Render structure configuration inputs.
    
    Returns tuple of (heightfield, topology, support_layout)
    """
    st.subheader("Shape")
    
    heightfield = st.selectbox(
        "Heightfield",
        options=CONFIG.heightfields,
        index=CONFIG.heightfields.index(current_heightfield) if current_heightfield in CONFIG.heightfields else 0,
        help="Surface shape of the canopy"
    )
    
    # Show shape description
    descriptions = {
        'flat': "Flat horizontal surface at max height",
        'paraboloid': "Dome shape - highest at center, lower at edges",
        'ridge': "Peaked along centerline like a tent",
        'saddle': "Hyperbolic paraboloid - modern sculptural form",
    }
    st.caption(descriptions.get(heightfield, ""))
    
    st.subheader("Topology")
    
    topology = st.selectbox(
        "Bar Pattern",
        options=CONFIG.topologies,
        index=CONFIG.topologies.index(current_topology) if current_topology in CONFIG.topologies else 0,
        help="Pattern of structural bars"
    )
    
    topo_descriptions = {
        'grid': "Orthogonal grid with cross-bracing in each cell (most stable)",
        'diagrid': "Diamond pattern - diagonal bars only (requires edge supports)",
        'triangulated': "Triangular mesh - alternating diagonals",
    }
    st.caption(topo_descriptions.get(topology, ""))
    
    st.subheader("Supports")
    
    # Get stable support options based on topology and grid size
    available_supports = get_stable_support_options(topology, nx, ny)
    
    # If current selection is no longer valid, default to 'edges'
    if current_support_layout not in available_supports:
        current_support_layout = 'edges'
    
    support_layout = st.selectbox(
        "Support Layout",
        options=available_supports,
        index=available_supports.index(current_support_layout) if current_support_layout in available_supports else 0,
        help="Where the structure is supported"
    )
    
    support_descriptions = {
        'edges': "All edge nodes supported (most stable)",
        'corners': "Four corners only (requires grid topology + 3x3 min)",
        'perimeter_4': "Every 4th node on perimeter (requires 4x4 min grid)",
    }
    st.caption(support_descriptions.get(support_layout, ""))
    
    # Show warning if combination is risky
    warning = check_stability_warning(topology, support_layout, nx, ny)
    if warning:
        st.warning(warning)
    
    return heightfield, topology, support_layout


def render_material_inputs(
    current_A: float,
    current_gravity: float,
) -> Tuple[float, float]:
    """
    Render material and loading inputs.
    
    Returns tuple of (A, gravity_load)
    """
    st.subheader("Section")
    
    # Convert to cm^2 for display
    A_cm2 = current_A * 10000
    
    A_cm2_input = st.slider(
        "Cross-section Area (cm^2)",
        min_value=3.0,
        max_value=20.0,
        value=A_cm2,
        step=0.5,
        help="Cross-sectional area of all bars"
    )
    
    A = A_cm2_input / 10000  # Convert back to m^2
    
    st.subheader("Loading")
    
    gravity_kn = abs(current_gravity) / 1000
    
    gravity_kn_input = st.slider(
        "Total Gravity Load (kN)",
        min_value=20.0,
        max_value=150.0,
        value=gravity_kn,
        step=5.0,
        help="Total vertical load distributed across all nodes"
    )
    
    gravity_load = -gravity_kn_input * 1000  # Convert to N, negative for downward
    
    return A, gravity_load

