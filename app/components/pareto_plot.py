# app/components/pareto_plot.py
"""
Interactive Pareto plot component for design exploration.

Displays the trade-off space between objectives and allows
selection of designs for detailed inspection.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def render_pareto_plot(
    df: pd.DataFrame,
    pareto_mask: pd.Series,
    x_col: str = 'volume',
    y_col: str = 'max_displacement',
    size_col: str = 'n_length_bins',
    selected_idx: Optional[int] = None,
    height: int = 450,
) -> go.Figure:
    """
    Create an interactive Pareto plot visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe with design metrics
    pareto_mask : pd.Series
        Boolean mask identifying Pareto-optimal designs
    x_col : str
        Column for x-axis (default: volume)
    y_col : str
        Column for y-axis (default: max_displacement)
    size_col : str
        Column for marker size (default: n_length_bins)
    selected_idx : Optional[int]
        Index of currently selected design (highlighted)
    height : int
        Figure height in pixels
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Filter to successful designs
    successful = df[df['ok'] == True].copy()
    
    if len(successful) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No successful designs to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=height)
        return fig
    
    # Get Pareto and dominated masks
    pareto_designs = successful[pareto_mask[successful.index]]
    dominated_designs = successful[~pareto_mask[successful.index]]
    
    # Convert displacement to mm for display
    y_scale = 1000 if y_col == 'max_displacement' else 1
    y_label = 'Max Displacement (mm)' if y_col == 'max_displacement' else y_col.replace('_', ' ').title()
    x_label = 'Material Volume (m³)' if x_col == 'volume' else x_col.replace('_', ' ').title()
    
    # Normalize size for better visualization
    if size_col in successful.columns:
        size_values = successful[size_col].fillna(successful[size_col].median())
        size_min, size_max = size_values.min(), size_values.max()
        if size_max > size_min:
            # Map to size range 8-25
            size_normalized = 8 + 17 * (size_values - size_min) / (size_max - size_min)
        else:
            size_normalized = pd.Series(12, index=size_values.index)
    else:
        size_normalized = pd.Series(12, index=successful.index)
    
    fig = go.Figure()
    
    # Plot dominated designs (gray)
    if len(dominated_designs) > 0:
        hover_texts = []
        for idx, row in dominated_designs.iterrows():
            text = (
                f"<b>Design #{idx}</b><br>"
                f"Volume: {row['volume']:.4f} m³<br>"
                f"Displacement: {row['max_displacement']*1000:.2f} mm<br>"
                f"Length Bins: {row['n_length_bins']:.0f}<br>"
                f"Topology: {row['topology']}<br>"
                f"Grid: {row['nx']:.0f}×{row['ny']:.0f}<br>"
                f"Size: {row['width']:.1f}×{row['depth']:.1f} m"
            )
            hover_texts.append(text)
        
        fig.add_trace(go.Scatter(
            x=dominated_designs[x_col],
            y=dominated_designs[y_col] * y_scale,
            mode='markers',
            marker=dict(
                size=size_normalized.loc[dominated_designs.index].values,
                color='rgba(180, 180, 180, 0.6)',
                line=dict(width=1, color='rgba(150, 150, 150, 0.8)'),
            ),
            name=f'Dominated ({len(dominated_designs)})',
            text=hover_texts,
            hoverinfo='text',
            customdata=dominated_designs.index.tolist(),
        ))
    
    # Plot Pareto-optimal designs (red/orange gradient)
    if len(pareto_designs) > 0:
        hover_texts = []
        for idx, row in pareto_designs.iterrows():
            text = (
                f"<b>★ Pareto Design #{idx}</b><br>"
                f"Volume: {row['volume']:.4f} m³<br>"
                f"Displacement: {row['max_displacement']*1000:.2f} mm<br>"
                f"Length Bins: {row['n_length_bins']:.0f}<br>"
                f"Topology: {row['topology']}<br>"
                f"Grid: {row['nx']:.0f}×{row['ny']:.0f}<br>"
                f"Size: {row['width']:.1f}×{row['depth']:.1f} m"
            )
            hover_texts.append(text)
        
        fig.add_trace(go.Scatter(
            x=pareto_designs[x_col],
            y=pareto_designs[y_col] * y_scale,
            mode='markers',
            marker=dict(
                size=size_normalized.loc[pareto_designs.index].values + 4,
                color='rgba(220, 60, 60, 0.85)',
                line=dict(width=2, color='darkred'),
            ),
            name=f'Pareto-optimal ({len(pareto_designs)})',
            text=hover_texts,
            hoverinfo='text',
            customdata=pareto_designs.index.tolist(),
        ))
        
        # Connect Pareto points with line (frontier)
        pareto_sorted = pareto_designs.sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=pareto_sorted[x_col],
            y=pareto_sorted[y_col] * y_scale,
            mode='lines',
            line=dict(color='rgba(220, 60, 60, 0.4)', width=2, dash='dot'),
            name='Pareto Frontier',
            hoverinfo='skip',
            showlegend=False,
        ))
    
    # Highlight selected design if any
    if selected_idx is not None and selected_idx in successful.index:
        row = successful.loc[selected_idx]
        fig.add_trace(go.Scatter(
            x=[row[x_col]],
            y=[row[y_col] * y_scale],
            mode='markers',
            marker=dict(
                size=25,
                color='rgba(0, 200, 100, 0.9)',
                line=dict(width=3, color='darkgreen'),
                symbol='star',
            ),
            name='Selected',
            hoverinfo='skip',
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'Design Trade-offs: {len(successful)} variants, {pareto_mask.sum()} Pareto-optimal',
            font=dict(size=14),
        ),
        xaxis=dict(
            title=x_label,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
        ),
        yaxis=dict(
            title=y_label,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(200, 200, 200, 0.5)',
            borderwidth=1,
        ),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode='closest',
        plot_bgcolor='rgba(250, 250, 250, 0.8)',
    )
    
    # Add annotation explaining size
    fig.add_annotation(
        text=f"Marker size = {size_col.replace('_', ' ')}",
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor='right',
        yanchor='bottom',
    )
    
    return fig


def create_design_summary_table(
    df: pd.DataFrame,
    pareto_mask: pd.Series,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Create a summary table of top designs for selection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    pareto_mask : pd.Series
        Boolean mask for Pareto designs
    top_n : int
        Number of top designs to show
    
    Returns:
    --------
    pd.DataFrame
        Formatted summary table
    """
    successful = df[df['ok'] == True].copy()
    
    if len(successful) == 0:
        return pd.DataFrame()
    
    # Add Pareto flag
    successful['is_pareto'] = pareto_mask[successful.index]
    
    # Sort: Pareto first, then by displacement
    successful = successful.sort_values(
        ['is_pareto', 'max_displacement'],
        ascending=[False, True]
    ).head(top_n)
    
    # Format for display
    summary = pd.DataFrame({
        'ID': successful.index,
        '★': ['★' if p else '' for p in successful['is_pareto']],
        'Vol (m³)': successful['volume'].round(4),
        'Disp (mm)': (successful['max_displacement'] * 1000).round(2),
        'Bins': successful['n_length_bins'].astype(int),
        'Topology': successful['topology'],
        'Grid': [f"{int(r['nx'])}×{int(r['ny'])}" for _, r in successful.iterrows()],
        'Size': [f"{r['width']:.1f}×{r['depth']:.1f}" for _, r in successful.iterrows()],
    })
    
    return summary.reset_index(drop=True)


def get_design_details(df: pd.DataFrame, idx: int) -> dict:
    """
    Get detailed information for a single design.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    idx : int
        Index of the design to get details for
    
    Returns:
    --------
    dict
        Design parameters and metrics
    """
    if idx not in df.index:
        return {}
    
    row = df.loc[idx]
    
    return {
        'index': idx,
        'parameters': {
            'width': row['width'],
            'depth': row['depth'],
            'nx': int(row['nx']),
            'ny': int(row['ny']),
            'min_height': row['min_height'],
            'max_height': row['max_height'],
            'heightfield': row['heightfield'],
            'topology': row['topology'],
            'support_layout': row['support_layout'],
            'A': row['A'],
            'gravity_load': row['gravity_load'],
        },
        'metrics': {
            'max_displacement': row['max_displacement'],
            'max_displacement_mm': row['max_displacement'] * 1000,
            'volume': row['volume'],
            'n_length_bins': int(row['n_length_bins']) if pd.notna(row['n_length_bins']) else 0,
            'max_tension': row.get('max_tension', 0),
            'max_compression': row.get('max_compression', 0),
            'n_nodes': int(row.get('n_nodes', 0)),
            'n_bars': int(row.get('n_bars', 0)),
        },
    }




