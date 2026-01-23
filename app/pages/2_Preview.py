# app/pages/2_Preview.py
"""
Preview Page - Generate and visualize a single design
"""

import streamlit as st
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from state import get_brief
from services import DesignService
from components import render_3d_model, render_metrics_panel

st.set_page_config(page_title="Preview Design", layout="wide")

st.title("Design Preview")
st.markdown("Generate and visualize a single canopy design based on your brief.")

# Get current brief
brief = get_brief()

# Show brief summary
with st.expander("Current Design Brief", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Footprint:** {brief.width:.1f} x {brief.depth:.1f} m")
        st.write(f"**Height:** {brief.min_height:.1f} - {brief.max_height:.1f} m")
    with col2:
        st.write(f"**Grid:** {brief.nx} x {brief.ny}")
        st.write(f"**Shape:** {brief.heightfield}")
    with col3:
        st.write(f"**Topology:** {brief.topology}")
        st.write(f"**Supports:** {brief.support_layout}")

st.divider()

# Generate button
if st.button("Generate Design", type="primary"):
    with st.spinner("Generating and solving structure..."):
        success, result, error = DesignService.generate_preview(brief)
    
    if success:
        st.session_state.preview_result = result
        st.success("Design generated successfully!")
    else:
        st.error(f"Failed to generate design: {error}")
        st.info("Try adjusting parameters - some configurations may be unstable.")

# Show results if available
if 'preview_result' in st.session_state:
    result = st.session_state.preview_result
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("3D Model")
        
        # Color options
        color_by = st.radio(
            "Color bars by:",
            options=['none', 'force'],
            horizontal=True,
            index=0,
        )
        
        fig = render_3d_model(
            nodes=result['nodes'],
            bars=result['bars'],
            forces=result['forces'],
            support_nodes=result['support_nodes'],
            height=500,
            color_by=color_by,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Drag to rotate, scroll to zoom. Red nodes are supports.")
    
    with col_right:
        st.subheader("Metrics")
        render_metrics_panel(result['metrics'])
        
        st.divider()
        
        # Quick export
        st.subheader("Quick Export")
        st.page_link("pages/3_Export.py", label="Go to Export Page", icon="➡️")

else:
    st.info("Click 'Generate Design' to preview your canopy structure.")

st.divider()

# Navigation
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Design_Brief.py", label="Back to Design Brief", icon="⬅️")

with col2:
    st.page_link("pages/3_Export.py", label="Export Results", icon="➡️")

