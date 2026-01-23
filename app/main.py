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
import pandas as pd

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from services import DesignService, ExportService
from services.exploration_service import ExplorationService
from services.design_service import compute_deflection_check, compute_utilization
from services.pdf_service import generate_design_pdf, is_pdf_available
from components.model_viewer import render_3d_model
from components.parameter_inputs import get_stable_support_options, check_stability_warning
from components.pareto_plot import render_pareto_plot, create_design_summary_table, get_design_details
from components.design_card import render_design_card, render_compact_checks

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
    
    /* Pareto selection highlight */
    .selected-design {
        background-color: rgba(0, 200, 100, 0.1);
        border-left: 4px solid #00c864;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_params_hash(params: dict) -> str:
    """Generate a hash of parameters to detect changes."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def apply_selected_design(row: pd.Series):
    """Apply selected design parameters to session state (updates sliders)."""
    # Update all slider values in session state
    st.session_state['width'] = float(row['width'])
    st.session_state['depth'] = float(row['depth'])
    st.session_state['nx'] = int(row['nx'])
    st.session_state['ny'] = int(row['ny'])
    st.session_state['min_h'] = float(row['min_height'])
    st.session_state['max_h'] = float(row['max_height'])
    st.session_state['heightfield'] = row['heightfield']
    st.session_state['topology'] = row['topology']
    st.session_state['support_layout'] = row['support_layout']
    st.session_state['A_cm2'] = float(row['A']) * 10000  # Convert back to cm²
    st.session_state['gravity_kn'] = abs(float(row['gravity_load'])) / 1000  # Convert to kN


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
    
    # Load factor toggle for service vs ultimate loads
    st.markdown('<p class="sidebar-header">📊 Load Factor</p>', unsafe_allow_html=True)
    load_factor = st.radio(
        "Design Level",
        options=["Service (1.0)", "Ultimate (1.4)", "Ultimate (1.6)"],
        index=0,
        horizontal=True,
        key="load_factor",
        help="Service = deflection check, Ultimate = strength check"
    )
    
    # Parse factor and apply to gravity load
    factor_map = {"Service (1.0)": 1.0, "Ultimate (1.4)": 1.4, "Ultimate (1.6)": 1.6}
    applied_factor = factor_map[load_factor]
    gravity_load = -gravity_kn * 1000 * applied_factor  # Convert to N with factor
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # EXPLORE VARIANTS
    # -------------------------------------------------------------------------
    st.markdown('<p class="sidebar-header">🔍 Explore Variants</p>', unsafe_allow_html=True)
    
    n_explore = st.slider("Number of variants", 50, 200, 100, 10, key="n_explore")
    
    col1, col2 = st.columns(2)
    with col1:
        explore_seed = st.number_input("Seed", value=42, min_value=1, key="explore_seed")
    with col2:
        variation_pct = st.slider("Variation %", 10, 50, 30, 5, key="variation_pct")
    
    run_explore = st.button("🚀 Generate Variants", type="primary", use_container_width=True)
    
    if 'exploration_df' in st.session_state:
        n_total = len(st.session_state.exploration_df)
        n_success = st.session_state.exploration_df['ok'].sum()
        n_pareto = st.session_state.pareto_mask.sum() if 'pareto_mask' in st.session_state else 0
        st.caption(f"✅ {n_success}/{n_total} designs • ★ {n_pareto} Pareto-optimal")
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            del st.session_state['exploration_df']
            if 'pareto_mask' in st.session_state:
                del st.session_state['pareto_mask']
            if 'selected_explore_idx' in st.session_state:
                del st.session_state['selected_explore_idx']
            st.rerun()
    
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
        
        **Explore:**
        - Generates variants around current design
        - Pareto = best trade-offs (can't improve one metric without worsening another)
        - Click a design to apply its parameters
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

# Build params dict for exploration
current_params = {
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


# =============================================================================
# RUN EXPLORATION IF REQUESTED
# =============================================================================

if run_explore:
    with st.spinner(f"Generating and evaluating {n_explore} design variants..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Evaluating design {current}/{total}...")
        
        df = ExplorationService.run_exploration(
            base_params=current_params,
            n_designs=n_explore,
            seed=int(explore_seed),
            variation_pct=variation_pct / 100,
            progress_callback=update_progress,
        )
        
        # Compute Pareto mask
        pareto_mask = ExplorationService.get_pareto_mask(df)
        
        # Store in session state
        st.session_state.exploration_df = df
        st.session_state.pareto_mask = pareto_mask
        
        progress_bar.empty()
        status_text.empty()
        
        n_success = df['ok'].sum()
        n_pareto = pareto_mask.sum()
        st.success(f"✅ Generated {n_success} successful designs, {n_pareto} are Pareto-optimal!")
        st.rerun()


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
        
        # Compute design checks
        span = min(brief.width, brief.depth)
        defl_check = compute_deflection_check(metrics['max_displacement'], span)
        util_check = compute_utilization(
            result['nodes'], 
            result['bars'], 
            result['forces'], 
            brief.A, 
            brief.E
        )
        
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
        
        # Show compact design checks
        render_compact_checks(defl_check, util_check, metrics)
        
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
        # DESIGN REPORT CARD
        # -------------------------------------------------------------------------
        all_checks_pass = render_design_card(defl_check, util_check, metrics)
        
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
        
        # PDF Summary Export
        if is_pdf_available():
            try:
                pdf_bytes = generate_design_pdf(
                    params=params_dict,
                    metrics=metrics,
                    deflection_check=defl_check,
                    utilization_check=util_check,
                )
                
                st.download_button(
                    label="📑 Design Summary (PDF)",
                    data=pdf_bytes,
                    file_name="canopy_design_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")
        else:
            st.info("💡 Install `reportlab` for PDF export: `pip install reportlab`")

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
# EXPLORE DESIGN VARIANTS - Expandable Section
# =============================================================================

st.divider()

with st.expander("🔍 **Explore Design Variants** — Click to reveal Pareto frontier", expanded='exploration_df' in st.session_state):
    
    if 'exploration_df' not in st.session_state:
        st.info("""
        **Explore the design space around your current design!**
        
        1. Adjust the number of variants in the sidebar (50-200)
        2. Click **"Generate Variants"** to explore
        3. View the trade-off space in the Pareto plot
        4. Select a design to apply its parameters
        
        The exploration varies parameters around your current design to find optimal trade-offs
        between material volume, stiffness, and fabrication complexity.
        """)
    else:
        df = st.session_state.exploration_df
        pareto_mask = st.session_state.pareto_mask
        
        # Summary stats
        n_total = len(df)
        n_success = df['ok'].sum()
        n_pareto = pareto_mask.sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Designs", n_total)
        with col2:
            st.metric("Successful", n_success)
        with col3:
            st.metric("Pareto-Optimal", n_pareto)
        with col4:
            success_rate = 100 * n_success / n_total if n_total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.0f}%")
        
        st.markdown("---")
        
        # ---------------------------------------------------------------------
        # FILTERS
        # ---------------------------------------------------------------------
        st.markdown("**Filters**")
        
        successful_df = df[df['ok'] == True]
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            max_disp_filter = st.slider(
                "Max Displacement (mm)",
                min_value=0.0,
                max_value=float(successful_df['max_displacement'].max() * 1000) if len(successful_df) > 0 else 100.0,
                value=float(successful_df['max_displacement'].max() * 1000) if len(successful_df) > 0 else 100.0,
                key="filter_disp"
            )
        
        with filter_col2:
            max_length_filter = st.slider(
                "Max Member Length (mm)",
                min_value=0.0,
                max_value=float(successful_df['max_member_length'].max() * 1000) if len(successful_df) > 0 else 10000.0,
                value=float(successful_df['max_member_length'].max() * 1000) if len(successful_df) > 0 else 10000.0,
                key="filter_length"
            )
        
        with filter_col3:
            topology_options = ['All'] + list(successful_df['topology'].unique())
            topology_filter = st.selectbox(
                "Topology",
                options=topology_options,
                index=0,
                key="filter_topology"
            )
        
        # Apply filters
        filtered_df = successful_df.copy()
        filtered_mask = pareto_mask.copy()
        
        filtered_df = filtered_df[filtered_df['max_displacement'] * 1000 <= max_disp_filter]
        filtered_df = filtered_df[filtered_df['max_member_length'] * 1000 <= max_length_filter]
        
        if topology_filter != 'All':
            filtered_df = filtered_df[filtered_df['topology'] == topology_filter]
        
        # Update mask for filtered indices
        filtered_mask = pareto_mask.loc[filtered_df.index] if len(filtered_df) > 0 else pd.Series(dtype=bool)
        
        st.caption(f"Showing {len(filtered_df)} of {n_success} successful designs")
        
        st.markdown("---")
        
        # ---------------------------------------------------------------------
        # PARETO PLOT
        # ---------------------------------------------------------------------
        
        if len(filtered_df) > 0:
            # Get selected index if any
            selected_idx = st.session_state.get('selected_explore_idx', None)
            
            # Create plot with full pareto mask but only filtered data
            plot_df = df.copy()
            plot_df['_show'] = False
            plot_df.loc[filtered_df.index, '_show'] = True
            
            # Only show filtered designs in plot
            display_df = filtered_df.copy()
            display_df['ok'] = True  # All filtered are successful
            
            fig_pareto = render_pareto_plot(
                df=display_df,
                pareto_mask=filtered_mask,
                selected_idx=selected_idx,
                height=400,
            )
            
            st.plotly_chart(fig_pareto, use_container_width=True, key="pareto_plot")
            
            st.caption("📊 **Volume** vs **Displacement** trade-off • Marker size = fabrication complexity (length bins)")
            
            st.markdown("---")
            
            # ---------------------------------------------------------------------
            # DESIGN SELECTION TABLE
            # ---------------------------------------------------------------------
            
            st.markdown("**Select a Design**")
            
            # Create summary table
            summary_table = create_design_summary_table(display_df, filtered_mask, top_n=30)
            
            if len(summary_table) > 0:
                # Add selection column
                summary_table.insert(0, 'Select', False)
                
                # Use data editor for selection
                edited_df = st.data_editor(
                    summary_table,
                    column_config={
                        "Select": st.column_config.CheckboxColumn(
                            "Select",
                            help="Select this design",
                            default=False,
                        ),
                        "★": st.column_config.TextColumn("★", help="Pareto-optimal"),
                        "ID": st.column_config.NumberColumn("ID", help="Design index"),
                        "Vol (m³)": st.column_config.NumberColumn("Volume", format="%.4f"),
                        "Disp (mm)": st.column_config.NumberColumn("Displacement", format="%.2f"),
                        "Bins": st.column_config.NumberColumn("Length Bins"),
                    },
                    disabled=["★", "ID", "Vol (m³)", "Disp (mm)", "Bins", "Topology", "Grid", "Size"],
                    hide_index=True,
                    use_container_width=True,
                    key="design_table",
                )
                
                # Find selected row
                selected_rows = edited_df[edited_df['Select'] == True]
                
                if len(selected_rows) > 0:
                    # Get the first selected design
                    selected_id = int(selected_rows.iloc[0]['ID'])
                    st.session_state.selected_explore_idx = selected_id
                    
                    # Get full design details
                    selected_row = df.loc[selected_id]
                    
                    st.markdown("---")
                    st.markdown("**Selected Design Details**")
                    
                    col_detail1, col_detail2, col_detail3 = st.columns(3)
                    
                    with col_detail1:
                        st.markdown("**Geometry**")
                        st.write(f"Size: {selected_row['width']:.1f} × {selected_row['depth']:.1f} m")
                        st.write(f"Grid: {int(selected_row['nx'])} × {int(selected_row['ny'])}")
                        st.write(f"Height: {selected_row['min_height']:.2f} - {selected_row['max_height']:.2f} m")
                    
                    with col_detail2:
                        st.markdown("**Structure**")
                        st.write(f"Shape: {selected_row['heightfield']}")
                        st.write(f"Topology: {selected_row['topology']}")
                        st.write(f"Supports: {selected_row['support_layout']}")
                    
                    with col_detail3:
                        st.markdown("**Performance**")
                        st.write(f"Volume: {selected_row['volume']:.4f} m³")
                        st.write(f"Displacement: {selected_row['max_displacement']*1000:.2f} mm")
                        st.write(f"Length bins: {int(selected_row['n_length_bins'])}")
                    
                    # Apply button
                    st.markdown("---")
                    
                    if st.button("✅ **Apply Selected Design**", type="primary", use_container_width=True):
                        apply_selected_design(selected_row)
                        st.success("Design parameters applied! The 3D view above will update.")
                        st.rerun()
                    
                    st.caption("This will update the sidebar parameters to match the selected design.")
        else:
            st.warning("No designs match the current filters. Try relaxing the filter constraints.")


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(f"StructureCraft v{CONFIG.version} • Real-time 3D Spaceframe Designer")
