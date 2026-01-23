# mini_branch/viz/viz3d.py
"""
3D VISUALIZATION: Interactive Spaceframe Viewer
================================================

PURPOSE:
--------
Create interactive 3D visualizations of spaceframe structures using Plotly.
This enables:
- Rotation/zoom/pan of the 3D model
- Color coding by member force or utilization
- Support and load visualization
- Export to HTML for sharing

WHY PLOTLY?
-----------
- Interactive (rotate, zoom, pan)
- Works in Jupyter notebooks and as standalone HTML
- No external viewer needed
- Professional-looking output
"""

import numpy as np
from typing import Dict, List, Optional, Literal
import plotly.graph_objects as go

from ..v3d.model import Node3D, Truss3D
from ..v3d.elements import element_geometry_3d


def create_spaceframe_figure(
    nodes: Dict[int, Node3D],
    bars: List[Truss3D],
    forces: Optional[Dict[int, float]] = None,
    fixed_nodes: Optional[List[int]] = None,
    title: str = "Spaceframe Structure",
    color_by: Literal['none', 'force', 'length'] = 'none',
    show_nodes: bool = True,
    show_supports: bool = True,
) -> go.Figure:
    """
    Create a Plotly figure for a 3D spaceframe structure.
    
    Parameters:
    -----------
    nodes : Dict[int, Node3D]
        Dictionary mapping node IDs to Node3D objects
        
    bars : List[Truss3D]
        List of truss bar elements
        
    forces : Optional[Dict[int, float]]
        Dictionary mapping bar ID to axial force (for coloring)
        If None, all bars are same color
        
    fixed_nodes : Optional[List[int]]
        List of node IDs that are supported (for marking)
        
    title : str
        Plot title
        
    color_by : str
        How to color bars:
        - 'none': All bars same color (blue)
        - 'force': Color by axial force (red=tension, blue=compression)
        - 'length': Color by member length
        
    show_nodes : bool
        Whether to show node markers
        
    show_supports : bool
        Whether to highlight support nodes
    
    Returns:
    --------
    go.Figure
        Plotly figure object (can be shown or saved)
    """
    fig = go.Figure()
    
    # =========================================================================
    # DRAW BARS
    # =========================================================================
    
    # Collect bar data for plotting
    bar_x, bar_y, bar_z = [], [], []
    bar_colors = []
    bar_texts = []
    
    # Compute forces/lengths for coloring
    if color_by == 'force' and forces:
        max_force = max(abs(f) for f in forces.values()) if forces else 1.0
    elif color_by == 'length':
        lengths = [element_geometry_3d(nodes, bar)[0] for bar in bars]
        max_length = max(lengths) if lengths else 1.0
        min_length = min(lengths) if lengths else 0.0
    
    for bar in bars:
        ni = nodes[bar.ni]
        nj = nodes[bar.nj]
        
        # Each bar is a line segment: (xi, yi, zi) -> (xj, yj, zj)
        # Add None to break line segments
        bar_x.extend([ni.x, nj.x, None])
        bar_y.extend([ni.y, nj.y, None])
        bar_z.extend([ni.z, nj.z, None])
        
        # Compute color
        if color_by == 'force' and forces and bar.id in forces:
            force = forces[bar.id]
            # Normalize to -1 (compression) to +1 (tension)
            norm_force = force / max_force if max_force > 0 else 0
            bar_colors.append(norm_force)
            force_type = "Tension" if force > 0 else "Compression"
            bar_texts.append(f"Bar {bar.id}: {force/1000:.1f} kN ({force_type})")
        elif color_by == 'length':
            L = element_geometry_3d(nodes, bar)[0]
            norm_length = (L - min_length) / (max_length - min_length) if max_length > min_length else 0.5
            bar_colors.append(norm_length)
            bar_texts.append(f"Bar {bar.id}: L={L*1000:.0f} mm")
        else:
            bar_colors.append(0)
            L = element_geometry_3d(nodes, bar)[0]
            bar_texts.append(f"Bar {bar.id}: L={L*1000:.0f} mm")
    
    # Add bars trace
    if color_by == 'none':
        fig.add_trace(go.Scatter3d(
            x=bar_x, y=bar_y, z=bar_z,
            mode='lines',
            line=dict(color='steelblue', width=4),
            name='Members',
            hoverinfo='skip',
        ))
    else:
        # For colored bars, we need to draw them individually
        for i, bar in enumerate(bars):
            ni = nodes[bar.ni]
            nj = nodes[bar.nj]
            
            if color_by == 'force' and forces and bar.id in forces:
                force = forces[bar.id]
                # Red for tension, blue for compression
                if force > 0:
                    color = f'rgb({int(255 * abs(force) / max_force)}, 50, 50)'
                else:
                    color = f'rgb(50, 50, {int(255 * abs(force) / max_force)})'
            else:
                color = 'steelblue'
            
            fig.add_trace(go.Scatter3d(
                x=[ni.x, nj.x], y=[ni.y, nj.y], z=[ni.z, nj.z],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'Bar {bar.id}',
                showlegend=False,
                hovertext=bar_texts[i] if i < len(bar_texts) else '',
                hoverinfo='text',
            ))
    
    # =========================================================================
    # DRAW NODES
    # =========================================================================
    
    if show_nodes:
        node_x = [n.x for n in nodes.values()]
        node_y = [n.y for n in nodes.values()]
        node_z = [n.z for n in nodes.values()]
        node_texts = [f"Node {n.id}: ({n.x:.2f}, {n.y:.2f}, {n.z:.2f})" for n in nodes.values()]
        
        # Determine node colors (highlight supports)
        if show_supports and fixed_nodes:
            node_colors = ['red' if n.id in fixed_nodes else 'darkgray' for n in nodes.values()]
            node_sizes = [10 if n.id in fixed_nodes else 5 for n in nodes.values()]
        else:
            node_colors = 'darkgray'
            node_sizes = 5
        
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='black'),
            ),
            name='Nodes',
            text=node_texts,
            hoverinfo='text',
        ))
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    
    # Compute bounds for axis scaling
    all_x = [n.x for n in nodes.values()]
    all_y = [n.y for n in nodes.values()]
    all_z = [n.z for n in nodes.values()]
    
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)
    max_range = max(x_range, y_range, z_range, 1.0)
    
    # Center the view
    x_mid = (max(all_x) + min(all_x)) / 2
    y_mid = (max(all_y) + min(all_y)) / 2
    z_mid = (max(all_z) + min(all_z)) / 2
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(
                title='X (m)',
                range=[x_mid - max_range/2 - 0.5, x_mid + max_range/2 + 0.5],
            ),
            yaxis=dict(
                title='Y (m)',
                range=[y_mid - max_range/2 - 0.5, y_mid + max_range/2 + 0.5],
            ),
            zaxis=dict(
                title='Z (m)',
                range=[min(all_z) - 0.5, max(all_z) + 0.5],
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),  # Isometric-ish view
            ),
        ),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    return fig


def plot_spaceframe_3d(
    nodes: Dict[int, Node3D],
    bars: List[Truss3D],
    forces: Optional[Dict[int, float]] = None,
    fixed_nodes: Optional[List[int]] = None,
    title: str = "Spaceframe Structure",
    outpath: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> go.Figure:
    """
    Create and optionally display/save a 3D spaceframe visualization.
    
    Parameters:
    -----------
    nodes, bars, forces, fixed_nodes, title:
        See create_spaceframe_figure()
        
    outpath : Optional[str]
        If provided, save as HTML file
        
    show : bool
        Whether to display the figure (default: True)
        
    **kwargs:
        Additional arguments passed to create_spaceframe_figure()
    
    Returns:
    --------
    go.Figure
        The created figure
    
    Example:
    --------
    >>> fig = plot_spaceframe_3d(nodes, bars, forces=force_dict, 
    ...                          title="Canopy Structure",
    ...                          outpath="artifacts/canopy.html")
    """
    fig = create_spaceframe_figure(
        nodes=nodes,
        bars=bars,
        forces=forces,
        fixed_nodes=fixed_nodes,
        title=title,
        **kwargs
    )
    
    if outpath:
        import os
        os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
        fig.write_html(outpath)
        print(f"3D visualization saved to: {outpath}")
    
    if show:
        fig.show()
    
    return fig


