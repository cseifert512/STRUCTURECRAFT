# app/pages/3_Export.py
"""
Export Page - Download cut lists and model data
"""

import streamlit as st
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from state import get_brief
from services import DesignService, ExportService

st.set_page_config(page_title="Export", layout="wide")

st.title("Export Design")
st.markdown("Download cut lists and model data for fabrication.")

# Check if we have a preview result
if 'preview_result' not in st.session_state:
    st.warning("No design available for export.")
    st.info("Go to the Preview page to generate a design first.")
    st.page_link("pages/2_Preview.py", label="Go to Preview", icon="➡️")
    st.stop()

result = st.session_state.preview_result
brief = get_brief()

# Summary
st.subheader("Design Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Geometry**")
    st.write(f"Footprint: {brief.width:.1f} x {brief.depth:.1f} m")
    st.write(f"Nodes: {result['metrics']['n_nodes']}")
    st.write(f"Bars: {result['metrics']['n_bars']}")

with col2:
    st.markdown("**Performance**")
    disp_mm = result['metrics']['max_displacement'] * 1000
    st.write(f"Max displacement: {disp_mm:.2f} mm")
    st.write(f"Max tension: {result['metrics']['max_tension']/1000:.1f} kN")
    st.write(f"Max compression: {result['metrics']['max_compression']/1000:.1f} kN")

with col3:
    st.markdown("**Fabrication**")
    st.write(f"Volume: {result['metrics']['volume']:.4f} m^3")
    st.write(f"Length bins: {result['metrics']['n_length_bins']}")
    st.write(f"Longest bar: {result['metrics']['max_member_length']*1000:.0f} mm")

st.divider()

# Export options
st.subheader("Available Exports")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Cut List (CSV)")
    st.write("Member lengths and forces for fabrication.")
    
    csv_content = ExportService.generate_cutlist_csv(
        result['nodes'],
        result['bars'],
        result['forces']
    )
    
    st.download_button(
        label="Download Cut List",
        data=csv_content,
        file_name="canopy_cutlist.csv",
        mime="text/csv",
    )
    
    # Preview
    with st.expander("Preview CSV"):
        st.code(csv_content[:1000] + "..." if len(csv_content) > 1000 else csv_content)

with col2:
    st.markdown("### Model Data (JSON)")
    st.write("Complete model for interchange or re-analysis.")
    
    params_dict = brief.to_dict()
    
    json_content = ExportService.generate_model_json(
        result['nodes'],
        result['bars'],
        params_dict,
        result['metrics'],
        result['forces']
    )
    
    st.download_button(
        label="Download Model JSON",
        data=json_content,
        file_name="canopy_model.json",
        mime="application/json",
    )
    
    # Preview
    with st.expander("Preview JSON"):
        st.code(json_content[:1500] + "..." if len(json_content) > 1500 else json_content, language="json")

st.divider()

# Summary text
st.markdown("### Design Summary (Text)")

summary_text = ExportService.generate_summary_text(
    params_dict,
    result['metrics']
)

st.download_button(
    label="Download Summary",
    data=summary_text,
    file_name="canopy_summary.txt",
    mime="text/plain",
)

with st.expander("Preview Summary"):
    st.code(summary_text)

st.divider()

# Navigation
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/2_Preview.py", label="Back to Preview", icon="⬅️")

with col2:
    st.page_link("pages/1_Design_Brief.py", label="Start New Design", icon="🔄")

