#!/usr/bin/env python3
"""
RUN_CANOPY_SEARCH: Batch Design Exploration with Pareto Frontier
=================================================================

This demo shows the complete design exploration workflow:
1. Generate hundreds of random canopy variants
2. Evaluate each for structural performance + fabrication metrics
3. Identify Pareto-optimal designs (best trade-offs)
4. Visualize the trade-off space
5. Export results for further analysis

This is the "I have a design tool, not a homework solver" moment.

Run with:
    python demos/run_canopy_search.py
    
Outputs:
    artifacts/canopy_search_results.csv  - All design evaluations
    artifacts/canopy_pareto.html         - Interactive Pareto plot
    artifacts/canopy_pareto.png          - Static Pareto plot
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mini_branch.explore3d import (
    run_batch_exploration,
    pareto_mask,
    rank_designs,
)


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def plot_pareto_2d(
    df: pd.DataFrame,
    mask: pd.Series,
    x: str,
    y: str,
    outpath: str,
    title: str = "Pareto Frontier",
):
    """Create a 2D Pareto plot using matplotlib."""
    import matplotlib.pyplot as plt
    
    # Filter successful designs
    successful = df[df['ok'] == True].copy()
    pareto_designs = successful[mask[successful.index]]
    dominated_designs = successful[~mask[successful.index]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot dominated designs
    if len(dominated_designs) > 0:
        ax.scatter(
            dominated_designs[x],
            dominated_designs[y],
            c='lightgray',
            s=40,
            alpha=0.5,
            label=f'Dominated ({len(dominated_designs)})',
            edgecolors='none',
        )
    
    # Plot Pareto designs
    if len(pareto_designs) > 0:
        ax.scatter(
            pareto_designs[x],
            pareto_designs[y],
            c='red',
            s=100,
            alpha=0.8,
            label=f'Pareto-optimal ({len(pareto_designs)})',
            edgecolors='darkred',
            linewidths=2,
        )
        
        # Connect Pareto points with line
        pareto_sorted = pareto_designs.sort_values(x)
        ax.plot(
            pareto_sorted[x],
            pareto_sorted[y],
            'r--',
            alpha=0.5,
            linewidth=1,
        )
    
    # Labels
    xlabel = x.replace('_', ' ').title()
    ylabel = y.replace('_', ' ').title()
    
    if x == 'volume':
        xlabel = 'Material Volume (m^3)'
    if y == 'max_displacement':
        ylabel = 'Max Displacement (m)'
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Info box
    info_text = (
        f"Total designs: {len(successful)}\n"
        f"Pareto-optimal: {len(pareto_designs)}\n"
        f"\nLower = better (both axes)"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Pareto plot saved to: {outpath}")


def plot_pareto_interactive(
    df: pd.DataFrame,
    mask: pd.Series,
    outpath: str,
):
    """Create an interactive Pareto plot using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed, skipping interactive plot")
        return
    
    successful = df[df['ok'] == True].copy()
    pareto_designs = successful[mask[successful.index]]
    dominated_designs = successful[~mask[successful.index]]
    
    # Create subplot with 2 views
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Volume vs Displacement',
            'Volume vs Fabrication Complexity'
        ),
        horizontal_spacing=0.12,
    )
    
    # View 1: Volume vs Displacement
    if len(dominated_designs) > 0:
        fig.add_trace(
            go.Scatter(
                x=dominated_designs['volume'],
                y=dominated_designs['max_displacement'] * 1000,  # to mm
                mode='markers',
                marker=dict(size=8, color='lightgray', opacity=0.5),
                name='Dominated',
                text=[f"ID: {i}<br>Vol: {r['volume']:.4f} m^3<br>Disp: {r['max_displacement']*1000:.2f} mm<br>Topo: {r['topology']}"
                      for i, r in dominated_designs.iterrows()],
                hoverinfo='text',
            ),
            row=1, col=1
        )
    
    if len(pareto_designs) > 0:
        fig.add_trace(
            go.Scatter(
                x=pareto_designs['volume'],
                y=pareto_designs['max_displacement'] * 1000,
                mode='markers',
                marker=dict(size=12, color='red', line=dict(width=2, color='darkred')),
                name='Pareto-optimal',
                text=[f"ID: {i}<br>Vol: {r['volume']:.4f} m^3<br>Disp: {r['max_displacement']*1000:.2f} mm<br>Topo: {r['topology']}<br>Bins: {r['n_length_bins']:.0f}"
                      for i, r in pareto_designs.iterrows()],
                hoverinfo='text',
            ),
            row=1, col=1
        )
    
    # View 2: Volume vs Length Bins
    if len(dominated_designs) > 0:
        fig.add_trace(
            go.Scatter(
                x=dominated_designs['volume'],
                y=dominated_designs['n_length_bins'],
                mode='markers',
                marker=dict(size=8, color='lightgray', opacity=0.5),
                name='Dominated',
                showlegend=False,
                text=[f"ID: {i}<br>Vol: {r['volume']:.4f} m^3<br>Bins: {r['n_length_bins']:.0f}<br>Topo: {r['topology']}"
                      for i, r in dominated_designs.iterrows()],
                hoverinfo='text',
            ),
            row=1, col=2
        )
    
    if len(pareto_designs) > 0:
        fig.add_trace(
            go.Scatter(
                x=pareto_designs['volume'],
                y=pareto_designs['n_length_bins'],
                mode='markers',
                marker=dict(size=12, color='red', line=dict(width=2, color='darkred')),
                name='Pareto-optimal',
                showlegend=False,
                text=[f"ID: {i}<br>Vol: {r['volume']:.4f} m^3<br>Bins: {r['n_length_bins']:.0f}<br>Topo: {r['topology']}"
                      for i, r in pareto_designs.iterrows()],
                hoverinfo='text',
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Material Volume (m^3)", row=1, col=1)
    fig.update_yaxes(title_text="Max Displacement (mm)", row=1, col=1)
    fig.update_xaxes(title_text="Material Volume (m^3)", row=1, col=2)
    fig.update_yaxes(title_text="Unique Length Bins", row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text=f"Canopy Design Exploration: {len(successful)} Designs, {len(pareto_designs)} Pareto-Optimal",
            font=dict(size=16),
        ),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=500,
        width=1100,
    )
    
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    fig.write_html(outpath)
    print(f"Interactive Pareto plot saved to: {outpath}")


def main():
    print_header("CANOPY DESIGN SPACE EXPLORATION")
    print("\nExploring the trade-off space between:")
    print("  - Material volume (cost)")
    print("  - Structural stiffness (displacement)")
    print("  - Fabrication simplicity (length bins)")
    
    # =========================================================================
    # STEP 1: RUN BATCH EXPLORATION
    # =========================================================================
    print_header("STEP 1: Generate and Evaluate Designs")
    
    n_designs = 200  # Number of variants to explore
    seed = 42
    
    print(f"\nGenerating {n_designs} random canopy variants...")
    print(f"(This may take a minute...)\n")
    
    df = run_batch_exploration(
        n=n_designs,
        seed=seed,
        show_progress=True,
        # Sampling ranges
        width_range=(6.0, 14.0),
        depth_range=(5.0, 11.0),
        nx_range=(3, 7),
        ny_range=(3, 6),
        height_range=(2.0, 4.5),
        heightfields=['flat', 'paraboloid', 'ridge', 'saddle'],
        topologies=['grid', 'diagrid', 'triangulated'],
        A_range=(0.0004, 0.0015),
        gravity_range=(-80000, -40000),
    )
    
    n_success = df['ok'].sum()
    n_failed = (~df['ok']).sum()
    
    print(f"\n  Completed: {n_success} successful, {n_failed} failed")
    
    # =========================================================================
    # STEP 2: COMPUTE PARETO FRONTIER
    # =========================================================================
    print_header("STEP 2: Identify Pareto-Optimal Designs")
    
    # Pareto for volume vs displacement vs fabrication
    mask = pareto_mask(df, objectives=[
        ('volume', 'min'),
        ('max_displacement', 'min'),
        ('n_length_bins', 'min'),
    ])
    
    n_pareto = mask.sum()
    pareto_pct = 100 * n_pareto / n_success if n_success > 0 else 0
    
    print(f"\n  Pareto-optimal designs: {n_pareto} ({pareto_pct:.1f}%)")
    print("  (These represent the best trade-offs between objectives)")
    
    # =========================================================================
    # STEP 3: ANALYZE RESULTS
    # =========================================================================
    print_header("STEP 3: Analysis Summary")
    
    successful = df[df['ok'] == True]
    pareto_designs = successful[mask[successful.index]]
    
    print("\n  SUCCESSFUL DESIGNS:")
    print(f"    Volume range:        {successful['volume'].min():.4f} - {successful['volume'].max():.4f} m^3")
    print(f"    Displacement range:  {successful['max_displacement'].min()*1000:.2f} - {successful['max_displacement'].max()*1000:.2f} mm")
    print(f"    Length bins range:   {successful['n_length_bins'].min():.0f} - {successful['n_length_bins'].max():.0f}")
    
    print("\n  PARETO-OPTIMAL DESIGNS:")
    print(f"    Volume range:        {pareto_designs['volume'].min():.4f} - {pareto_designs['volume'].max():.4f} m^3")
    print(f"    Displacement range:  {pareto_designs['max_displacement'].min()*1000:.2f} - {pareto_designs['max_displacement'].max()*1000:.2f} mm")
    print(f"    Length bins range:   {pareto_designs['n_length_bins'].min():.0f} - {pareto_designs['n_length_bins'].max():.0f}")
    
    # Topology breakdown
    print("\n  TOPOLOGY DISTRIBUTION (Pareto):")
    for topo in pareto_designs['topology'].unique():
        count = (pareto_designs['topology'] == topo).sum()
        print(f"    {topo}: {count} designs")
    
    # =========================================================================
    # STEP 4: RANK TOP DESIGNS
    # =========================================================================
    print_header("STEP 4: Top 10 Designs (Weighted Score)")
    
    scores = rank_designs(df, weights={
        'volume': 1.0,
        'max_displacement': 1.5,  # Prioritize stiffness slightly
        'n_length_bins': 0.5,
    })
    
    df['score'] = scores
    df['is_pareto'] = mask
    
    top_10 = df[df['ok'] == True].nsmallest(10, 'score')
    
    print("\n  Rank | Volume  | Disp (mm) | Bins | Topology     | Pareto?")
    print("  " + "-" * 65)
    for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
        pareto_str = "Yes" if row['is_pareto'] else "No"
        print(f"  {rank:4d} | {row['volume']:.4f} | {row['max_displacement']*1000:9.2f} | {row['n_length_bins']:4.0f} | {row['topology']:12s} | {pareto_str}")
    
    # =========================================================================
    # STEP 5: EXPORT RESULTS
    # =========================================================================
    print_header("STEP 5: Export Results")
    
    # Save CSV
    csv_path = "artifacts/canopy_search_results.csv"
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n  Results CSV: {csv_path}")
    
    # Save Pareto plots
    png_path = "artifacts/canopy_pareto.png"
    plot_pareto_2d(df, mask, 'volume', 'max_displacement', png_path,
                   title=f"Canopy Design Trade-offs ({n_pareto} Pareto-optimal)")
    
    html_path = "artifacts/canopy_pareto.html"
    plot_pareto_interactive(df, mask, html_path)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY")
    
    best = top_10.iloc[0]
    
    print(f"""
    DESIGN EXPLORATION COMPLETE
    
    Explored:         {n_designs} variants
    Successful:       {n_success}
    Pareto-optimal:   {n_pareto}
    
    Best Design (weighted score):
      Volume:         {best['volume']:.4f} m^3
      Displacement:   {best['max_displacement']*1000:.2f} mm
      Length bins:    {best['n_length_bins']:.0f}
      Topology:       {best['topology']}
      Dimensions:     {best['width']:.1f} x {best['depth']:.1f} m
      Grid:           {best['nx']:.0f} x {best['ny']:.0f}
    
    Outputs:
      Results CSV:    {csv_path}
      Pareto plot:    {png_path}
      Interactive:    {html_path}
    
    Open {html_path} in browser for interactive exploration!
    """)
    
    return df, mask


if __name__ == "__main__":
    main()





