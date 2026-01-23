# app/main.py
"""
StructureCraft Design Explorer - Real-Time Live Interface

A single cohesive interface where parameter changes instantly 
update the 3D model and metrics.

Run with:
    streamlit run app/main.py
"""

import streamlit as st
import sys
from pathlib import Path
import hashlib
import json

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from services import DesignService, ExportService
from components.model_viewer import render_3d_model
from components.parameter_inputs import get_stable_support_options, check_stability_warning

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title=CONFIG.app_name,
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Tighter spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Success/error indicators */
    .status-success {
        color: #28a745;
        font-weight: 600;
    }
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def get_params_hash(params: dict) -> str:
    """Generate a hash of parameters to detect changes."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


# =============================================================================
# SIDEBAR - All Parameter Controls
# =============================================================================

with st.sidebar:
    st.title("🏗️ StructureCraft")
    st.caption("Real-time 3D Canopy Designer")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # GEOMETRY
    # -------------------------------------------------------------------------
    st.markdown('<p class="sidebar-header">📐 Geometry</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider("Width (m)", 4.0, 20.0, 10.0, 0.5, key="width")
    with col2:
        depth = st.slider("Depth (m)", 4.0, 16.0, 8.0, 0.5, key="depth")
    
    col1, col2 = st.columns(2)
    with col1:
        min_height = st.slider("Min H (m)", 1.5, 6.0, 2.5, 0.25, key="min_h")
    with col2:
        max_height = st.slider("Max H (m)", 1.5, 6.0, 4.0, 0.25, key="max_h")
    
    # Ensure max > min
    if max_height <= min_height:
        max_height = min_height + 0.5
    
    col1, col2 = st.columns(2)
    with col1:
        nx = st.slider("Grid X", 2, 10, 5, key="nx")
    with col2:
        ny = st.slider("Grid Y", 2, 10, 4, key="ny")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # STRUCTURE
    # -------------------------------------------------------------------------
    st.markdown('<p class="sidebar-header">🔧 Structure</p>', unsafe_allow_html=True)
    
    heightfield = st.selectbox(
        "Shape",
        options=['flat', 'paraboloid', 'ridge', 'saddle'],
        index=1,
        key="heightfield",
        help="Surface shape of the canopy"
    )
    
    topology = st.selectbox(
        "Topology",
        options=['grid', 'diagrid', 'triangulated'],
        index=0,
        key="topology",
        help="Bar connectivity pattern"
    )
    
    # Filter support options based on stability
    available_supports = get_stable_support_options(topology, nx, ny)
    
    # Get current selection or default
    current_support = st.session_state.get('support_layout', 'edges')
    if current_support not in available_supports:
        current_support = 'edges'
    
    support_layout = st.selectbox(
        "Supports",
        options=available_supports,
        index=available_supports.index(current_support) if current_support in available_supports else 0,
        key="support_layout",
        help="Support locations"
    )
    
    # Show stability warning if needed
    warning = check_stability_warning(topology, support_layout, nx, ny)
    if warning:
        st.warning(warning, icon="⚠️")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # MATERIAL & LOADING
    # -------------------------------------------------------------------------
    st.markdown('<p class="sidebar-header">⚙️ Material & Load</p>', unsafe_allow_html=True)
    
    A_cm2 = st.slider("Section (cm²)", 3.0, 20.0, 8.0, 0.5, key="A_cm2")
    A = A_cm2 / 10000  # Convert to m²
    
    gravity_kn = st.slider("Load (kN)", 20.0, 150.0, 50.0, 5.0, key="gravity_kn")
    gravity_load = -gravity_kn * 1000  # Convert to N, negative for downward
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # INFO
    # -------------------------------------------------------------------------
    with st.expander("ℹ️ Help", expanded=False):
        st.markdown("""
        **Shapes:**
        - *Flat*: Horizontal surface
        - *Paraboloid*: Dome/bowl
        - *Ridge*: Tent-like peak
        - *Saddle*: Hyperbolic paraboloid
        
        **Topology:**
        - *Grid*: Orthogonal + diagonals (most stable)
        - *Diagrid*: Diamond pattern
        - *Triangulated*: Alternating diagonals
        
        **Tips:**
        - Use 'edges' support for stability
        - Larger grids = more bars but stiffer
        - 'grid' topology is most versatile
        """)


# =============================================================================
# BUILD PARAMETER DICT FOR DESIGN SERVICE
# =============================================================================

from dataclasses import dataclass

@dataclass
class LiveBrief:
    width: float
    depth: float
    nx: int
    ny: int
    min_height: float
    max_height: float
    heightfield: str
    topology: str
    support_layout: str
    E: float = 210e9
    A: float = 0.0008
    gravity_load: float = -50000.0

# Create brief from sidebar values
brief = LiveBrief(
    width=width,
    depth=depth,
    nx=nx,
    ny=ny,
    min_height=min_height,
    max_height=max_height,
    heightfield=heightfield,
    topology=topology,
    support_layout=support_layout,
    E=210e9,
    A=A,
    gravity_load=gravity_load,
)


# =============================================================================
# MAIN AREA - 3D Model + Metrics
# =============================================================================

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("Live Design Preview")

# Generate design
with st.spinner("Computing structure..."):
    success, result, error = DesignService.generate_preview(brief)

# Status indicator
with col_status:
    if success:
        st.success("✓ Stable", icon="✅")
    else:
        st.error("✗ Unstable", icon="❌")

if success:
    # Two-column layout: 3D view on left (larger), metrics on right
    col_3d, col_metrics = st.columns([2, 1])
    
    # -------------------------------------------------------------------------
    # 3D MODEL VIEW
    # -------------------------------------------------------------------------
    with col_3d:
        # Color options
        color_by = st.radio(
            "Color by:",
            options=['none', 'force'],
            horizontal=True,
            index=0,
            label_visibility="collapsed"
        )
        
        fig = render_3d_model(
            nodes=result['nodes'],
            bars=result['bars'],
            forces=result['forces'],
            support_nodes=result['support_nodes'],
            height=550,
            color_by=color_by,
        )
        
        st.plotly_chart(fig, use_container_width=True, key="main_3d")
        
        st.caption("🖱️ Drag to rotate • Scroll to zoom • Red = supports • Blue/Red = compression/tension")
    
    # -------------------------------------------------------------------------
    # METRICS PANEL
    # -------------------------------------------------------------------------
    with col_metrics:
        metrics = result['metrics']
        
        st.subheader("📊 Structure")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", metrics['n_nodes'])
            st.metric("Bars", metrics['n_bars'])
        with col2:
            st.metric("Supports", metrics['n_supports'])
            st.metric("Grid", f"{nx}×{ny}")
        
        st.divider()
        
        st.subheader("📈 Performance")
        disp_mm = metrics['max_displacement'] * 1000
        st.metric("Max Displacement", f"{disp_mm:.2f} mm")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Tension", f"{metrics['max_tension']/1000:.1f} kN")
        with col2:
            st.metric("Max Compression", f"{metrics['max_compression']/1000:.1f} kN")
        
        st.divider()
        
        st.subheader("🔨 Fabrication")
        st.metric("Material Volume", f"{metrics['volume']:.4f} m³")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Length Bins", metrics['n_length_bins'])
        with col2:
            max_len_mm = metrics['max_member_length'] * 1000
            st.metric("Longest Bar", f"{max_len_mm:.0f} mm")
        
        # Shipping warning
        if max_len_mm > 6000:
            st.warning("⚠️ Bars exceed 6m shipping limit")
        
        st.divider()
        
        # -------------------------------------------------------------------------
        # QUICK EXPORT
        # -------------------------------------------------------------------------
        st.subheader("📥 Export")
        
        # CSV download
        csv_content = ExportService.generate_cutlist_csv(
            result['nodes'],
            result['bars'],
            result['forces']
        )
        
        st.download_button(
            label="📄 Cut List (CSV)",
            data=csv_content,
            file_name="canopy_cutlist.csv",
            mime="text/csv",
            use_container_width=True,
        )
        
        # JSON download
        params_dict = {
            'width': brief.width,
            'depth': brief.depth,
            'nx': brief.nx,
            'ny': brief.ny,
            'min_height': brief.min_height,
            'max_height': brief.max_height,
            'heightfield': brief.heightfield,
            'topology': brief.topology,
            'support_layout': brief.support_layout,
            'E': brief.E,
            'A': brief.A,
            'gravity_load': brief.gravity_load,
        }
        
        json_content = ExportService.generate_model_json(
            result['nodes'],
            result['bars'],
            params_dict,
            metrics,
            result['forces']
        )
        
        st.download_button(
            label="📦 Model (JSON)",
            data=json_content,
            file_name="canopy_model.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    # Error state
    st.error(f"**Design Failed:** {error}")
    
    st.info("""
    **Try adjusting:**
    - Use 'edges' support layout (most stable)
    - Use 'grid' topology (best bracing)  
    - Increase grid density (nx, ny ≥ 3)
    - Avoid 'corners' support with 'diagrid' topology
    """)
    
    # Show a placeholder
    st.markdown("---")
    st.markdown("### Adjust parameters in the sidebar to generate a stable design")

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(f"StructureCraft v{CONFIG.version} • Real-time 3D Spaceframe Designer")
