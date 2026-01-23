# app/pages/1_Design_Brief.py
"""
Design Brief Page - Define constraints and preferences
"""

import streamlit as st
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from state import get_brief, update_brief, clear_exploration
from state.session import DesignBrief
from components import (
    render_geometry_inputs,
    render_structure_inputs,
    render_material_inputs,
)

st.set_page_config(page_title="Design Brief", layout="wide")

st.title("Design Brief")
st.markdown("Define your canopy design constraints and preferences.")

# Get current brief
brief = get_brief()

# Layout: two columns
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("Geometry")
    
    width, depth, min_height, max_height, nx, ny = render_geometry_inputs(
        current_width=brief.width,
        current_depth=brief.depth,
        current_min_height=brief.min_height,
        current_max_height=brief.max_height,
        current_nx=brief.nx,
        current_ny=brief.ny,
    )

with col_right:
    st.header("Structure")
    
    heightfield, topology, support_layout = render_structure_inputs(
        current_heightfield=brief.heightfield,
        current_topology=brief.topology,
        current_support_layout=brief.support_layout,
        nx=nx,
        ny=ny,
    )
    
    st.header("Material & Loading")
    
    A, gravity_load = render_material_inputs(
        current_A=brief.A,
        current_gravity=brief.gravity_load,
    )

# Update brief if values changed
if (width != brief.width or depth != brief.depth or 
    min_height != brief.min_height or max_height != brief.max_height or
    nx != brief.nx or ny != brief.ny or
    heightfield != brief.heightfield or topology != brief.topology or
    support_layout != brief.support_layout or
    A != brief.A or gravity_load != brief.gravity_load):
    
    update_brief(
        width=width,
        depth=depth,
        min_height=min_height,
        max_height=max_height,
        nx=nx,
        ny=ny,
        heightfield=heightfield,
        topology=topology,
        support_layout=support_layout,
        A=A,
        gravity_load=gravity_load,
    )

st.divider()

# Summary
st.subheader("Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Geometry**")
    st.write(f"Footprint: {width:.1f} x {depth:.1f} m")
    st.write(f"Height: {min_height:.1f} - {max_height:.1f} m")
    st.write(f"Grid: {nx} x {ny} = {(nx+1)*(ny+1)} nodes")

with col2:
    st.markdown("**Structure**")
    st.write(f"Shape: {heightfield}")
    st.write(f"Topology: {topology}")
    st.write(f"Supports: {support_layout}")

with col3:
    st.markdown("**Loading**")
    st.write(f"Section: {A*10000:.1f} cm^2")
    st.write(f"Gravity: {abs(gravity_load)/1000:.0f} kN")

st.divider()

# Navigation
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/2_Preview.py", label="Preview Design", icon="➡️")

with col2:
    if st.button("Reset to Defaults"):
        st.session_state.brief = DesignBrief()
        clear_exploration()
        st.rerun()

