# app/components/model_viewer.py
"""
3D model viewer component using Plotly.
"""

import plotly.graph_objects as go
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mini_branch.v3d.elements import element_geometry_3d


def render_3d_model(
    nodes: Dict,
    bars: List,
    forces: Optional[Dict] = None,
    support_nodes: Optional[List[int]] = None,
    height: int = 500,
    color_by: str = 'none',
) -> go.Figure:
    """
    Create a 3D visualization of the structure.
    
    Parameters:
    -----------
    nodes : Dict
        Node dictionary {id: Node3D}
    bars : List
        List of Truss3D bars
    forces : Optional[Dict]
        Dictionary {bar_id: force} for coloring
    support_nodes : Optional[List[int]]
        List of supported node IDs
    height : int
        Figure height in pixels
    color_by : str
        'none', 'force', or 'length'
    
    Returns:
    --------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Compute force range for coloring
    if color_by == 'force' and forces:
        max_force = max(abs(f) for f in forces.values()) if forces else 1.0
    
    # Draw bars
    for bar in bars:
        ni = nodes[bar.ni]
        nj = nodes[bar.nj]
        
        # Determine color
        if color_by == 'force' and forces and bar.id in forces:
            force = forces[bar.id]
            if force > 0:
                # Tension: red
                intensity = min(abs(force) / max_force, 1.0)
                color = f'rgb({int(200 * intensity + 55)}, 50, 50)'
            else:
                # Compression: blue
                intensity = min(abs(force) / max_force, 1.0)
                color = f'rgb(50, 50, {int(200 * intensity + 55)})'
        else:
            color = 'steelblue'
        
        L, _, _, _ = element_geometry_3d(nodes, bar)
        force_val = forces.get(bar.id, 0) if forces else 0
        hover_text = f"Bar {bar.id}<br>L: {L*1000:.0f} mm<br>F: {force_val/1000:.1f} kN"
        
        fig.add_trace(go.Scatter3d(
            x=[ni.x, nj.x],
            y=[ni.y, nj.y],
            z=[ni.z, nj.z],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False,
            hovertext=hover_text,
            hoverinfo='text',
        ))
    
    # Draw nodes
    node_x = [n.x for n in nodes.values()]
    node_y = [n.y for n in nodes.values()]
    node_z = [n.z for n in nodes.values()]
    
    # Node colors (highlight supports)
    if support_nodes:
        node_colors = ['red' if n.id in support_nodes else 'gray' for n in nodes.values()]
        node_sizes = [8 if n.id in support_nodes else 4 for n in nodes.values()]
    else:
        node_colors = 'gray'
        node_sizes = 4
    
    node_texts = [f"Node {n.id}<br>({n.x:.2f}, {n.y:.2f}, {n.z:.2f})" for n in nodes.values()]
    
    fig.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
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
    
    # Layout
    all_x = [n.x for n in nodes.values()]
    all_y = [n.y for n in nodes.values()]
    all_z = [n.z for n in nodes.values()]
    
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)
    max_range = max(x_range, y_range, z_range, 1.0)
    
    x_mid = (max(all_x) + min(all_x)) / 2
    y_mid = (max(all_y) + min(all_y)) / 2
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X (m)',
                range=[x_mid - max_range/2 - 1, x_mid + max_range/2 + 1],
                backgroundcolor='rgba(240,240,240,0.5)',
            ),
            yaxis=dict(
                title='Y (m)',
                range=[y_mid - max_range/2 - 1, y_mid + max_range/2 + 1],
                backgroundcolor='rgba(240,240,240,0.5)',
            ),
            zaxis=dict(
                title='Z (m)',
                range=[0, max(all_z) + 1],
                backgroundcolor='rgba(240,240,240,0.5)',
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=False,
    )
    
    return fig





