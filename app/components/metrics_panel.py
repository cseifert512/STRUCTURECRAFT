# app/components/metrics_panel.py
"""
Metrics display panel component.
"""

import streamlit as st
from typing import Dict, Any


def render_metrics_panel(metrics: Dict[str, Any], columns: int = 3) -> None:
    """
    Render a metrics panel with key performance indicators.
    
    Parameters:
    -----------
    metrics : Dict
        Dictionary of metric values
    columns : int
        Number of columns for layout
    """
    # Structure metrics
    st.subheader("Structure")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Nodes", metrics.get('n_nodes', 0))
    with cols[1]:
        st.metric("Bars", metrics.get('n_bars', 0))
    with cols[2]:
        st.metric("Supports", metrics.get('n_supports', 0))
    
    # Performance metrics
    st.subheader("Performance")
    cols = st.columns(3)
    with cols[0]:
        disp_mm = metrics.get('max_displacement', 0) * 1000
        st.metric("Max Displacement", f"{disp_mm:.2f} mm")
    with cols[1]:
        tension_kn = metrics.get('max_tension', 0) / 1000
        st.metric("Max Tension", f"{tension_kn:.1f} kN")
    with cols[2]:
        comp_kn = metrics.get('max_compression', 0) / 1000
        st.metric("Max Compression", f"{comp_kn:.1f} kN")
    
    # Fabrication metrics
    st.subheader("Fabrication")
    cols = st.columns(3)
    with cols[0]:
        vol = metrics.get('volume', 0)
        st.metric("Volume", f"{vol:.4f} m^3")
    with cols[1]:
        st.metric("Length Bins", metrics.get('n_length_bins', 0))
    with cols[2]:
        max_len = metrics.get('max_member_length', 0) * 1000
        st.metric("Longest Bar", f"{max_len:.0f} mm")


def render_metrics_compact(metrics: Dict[str, Any]) -> None:
    """Render metrics in a compact single-line format."""
    disp_mm = metrics.get('max_displacement', 0) * 1000
    vol = metrics.get('volume', 0)
    bins = metrics.get('n_length_bins', 0)
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Disp", f"{disp_mm:.1f} mm")
    with cols[1]:
        st.metric("Vol", f"{vol:.3f} m^3")
    with cols[2]:
        st.metric("Bins", bins)
    with cols[3]:
        st.metric("Bars", metrics.get('n_bars', 0))





