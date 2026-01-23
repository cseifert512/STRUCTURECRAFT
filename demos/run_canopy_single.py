#!/usr/bin/env python3
"""
RUN_CANOPY_SINGLE: Generate and Analyze a Spaceframe Canopy
============================================================

This demo shows the complete design-to-analysis workflow:
1. Define design parameters (footprint, shape, topology)
2. Generate 3D geometry (nodes, bars)
3. Solve for structural response
4. Compute member forces
5. Export cut list
6. Visualize in 3D

This is the "I have a design tool, not a homework solver" milestone.

Run with:
    python demos/run_canopy_single.py
    
Outputs:
    artifacts/canopy_cutlist.csv  - Member cut list for fabrication
    artifacts/canopy_3d.html      - Interactive 3D visualization
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mini_branch.generative import generate_canopy, CanopyParams
from mini_branch.v3d.elements import truss3d_global_stiffness, truss3d_axial_force, element_geometry_3d
from mini_branch.kernel.dof import DOFManager
from mini_branch.kernel.assemble import assemble_global_K
from mini_branch.kernel.solve import solve_linear


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def export_cutlist(nodes, bars, forces, outpath: str):
    """Export member cut list to CSV."""
    import csv
    
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    
    # Compute data for each bar
    data = []
    for bar in bars:
        L, l, m, n = element_geometry_3d(nodes, bar)
        force = forces.get(bar.id, 0)
        force_type = "T" if force > 0 else "C"  # Tension or Compression
        
        data.append({
            'bar_id': bar.id,
            'node_i': bar.ni,
            'node_j': bar.nj,
            'length_m': round(L, 4),
            'length_mm': round(L * 1000, 1),
            'force_kN': round(force / 1000, 2),
            'force_type': force_type,
        })
    
    # Sort by length for fabrication efficiency
    data.sort(key=lambda x: x['length_m'])
    
    # Write CSV
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Cut list exported to: {outpath}")
    return data


def main():
    print_header("SPACEFRAME CANOPY DESIGN TOOL")
    print("\nThis demo generates a 3D canopy structure, analyzes it,")
    print("and exports fabrication-ready outputs.")
    
    # =========================================================================
    # STEP 1: DEFINE DESIGN PARAMETERS
    # =========================================================================
    print_header("STEP 1: Design Parameters")
    
    params = CanopyParams(
        # Geometry
        width=12.0,       # 12m x 10m footprint
        depth=10.0,
        nx=6,             # 6x5 grid = 7x6 = 42 nodes
        ny=5,
        max_height=4.0,   # Peak at 4m
        min_height=2.5,   # Edges at 2.5m
        
        # Shape
        heightfield='paraboloid',  # Dome shape
        
        # Topology
        topology='grid',  # Grid with diagonals
        
        # Supports (edges for stability - spaceframes need distributed support)
        support_layout='edges',  # All edge nodes supported
        
        # Material (steel tubes)
        E=210e9,          # Steel
        A=0.0008,         # ~80mm diameter tube
        
        # Loading
        gravity_load=-80000.0,  # 80 kN total (distributed)
    )
    
    print(f"""
    Footprint:     {params.width} x {params.depth} m
    Grid density:  {params.nx} x {params.ny}
    Height range:  {params.min_height} - {params.max_height} m
    Heightfield:   {params.heightfield}
    Topology:      {params.topology}
    Supports:      {params.support_layout}
    
    Material:      Steel (E = {params.E/1e9:.0f} GPa)
    Section:       A = {params.A*1e4:.1f} cm^2
    
    Total load:    {abs(params.gravity_load)/1000:.0f} kN (gravity)
    """)
    
    # =========================================================================
    # STEP 2: GENERATE GEOMETRY
    # =========================================================================
    print_header("STEP 2: Generate Geometry")
    
    nodes, bars, fixed_dofs, F = generate_canopy(params)
    
    n_nodes = len(nodes)
    n_bars = len(bars)
    n_supports = len(fixed_dofs) // 3  # Each support has 3 fixed DOFs
    
    print(f"""
    Generated structure:
      Nodes:        {n_nodes}
      Bars:         {n_bars}
      Supports:     {n_supports} (at edges)
      Total DOFs:   {n_nodes * 3}
      Fixed DOFs:   {len(fixed_dofs)}
      Free DOFs:    {n_nodes * 3 - len(fixed_dofs)}
    """)
    
    # Member length statistics
    lengths = [element_geometry_3d(nodes, bar)[0] for bar in bars]
    print(f"""    Member lengths:
      Min:          {min(lengths)*1000:.0f} mm
      Max:          {max(lengths)*1000:.0f} mm
      Mean:         {np.mean(lengths)*1000:.0f} mm
      Std dev:      {np.std(lengths)*1000:.0f} mm
    """)
    
    # =========================================================================
    # STEP 3: ASSEMBLE AND SOLVE
    # =========================================================================
    print_header("STEP 3: Structural Analysis")
    
    dof = DOFManager(dof_per_node=3)
    ndof = dof.ndof(n_nodes)
    
    # Assemble stiffness matrix
    print("  Assembling stiffness matrix...")
    contributions = []
    for bar in bars:
        dof_map = dof.element_dof_map([bar.ni, bar.nj])
        ke = truss3d_global_stiffness(nodes, bar)
        contributions.append((dof_map, ke))
    
    K = assemble_global_K(ndof, contributions)
    print(f"    K matrix: {K.shape[0]} x {K.shape[1]}")
    
    # Solve
    print("  Solving...")
    d, R, free = solve_linear(K, F, fixed_dofs)
    print("    Solution converged!")
    
    # =========================================================================
    # STEP 4: POST-PROCESSING
    # =========================================================================
    print_header("STEP 4: Results")
    
    # Displacements
    displacements = []
    for node_id in nodes:
        ux = d[dof.idx(node_id, 0)]
        uy = d[dof.idx(node_id, 1)]
        uz = d[dof.idx(node_id, 2)]
        total = np.sqrt(ux**2 + uy**2 + uz**2)
        displacements.append((node_id, ux, uy, uz, total))
    
    # Find max displacement
    max_disp = max(displacements, key=lambda x: x[4])
    max_node_id, max_ux, max_uy, max_uz, max_total = max_disp
    
    print(f"""
    Displacements:
      Max total:    {max_total*1000:.2f} mm (at node {max_node_id})
      Max vertical: {abs(max_uz)*1000:.2f} mm
    """)
    
    # Member forces
    forces = {}
    max_tension = 0
    max_compression = 0
    
    for bar in bars:
        N = truss3d_axial_force(nodes, bar, d)
        forces[bar.id] = N
        
        if N > max_tension:
            max_tension = N
        if N < max_compression:
            max_compression = N
    
    print(f"""    Member forces:
      Max tension:     {max_tension/1000:.1f} kN
      Max compression: {abs(max_compression)/1000:.1f} kN
    """)
    
    # Reactions
    support_nodes = [i for i in range(n_nodes) if dof.idx(i, 0) in fixed_dofs]
    Rz_total = sum(R[dof.idx(i, 2)] for i in support_nodes)
    
    print(f"""    Reactions:
      Total vertical: {Rz_total/1000:.1f} kN
      Applied load:   {abs(params.gravity_load)/1000:.1f} kN
      Equilibrium:    {"OK" if abs(Rz_total + params.gravity_load) < 1 else "CHECK"}
    """)
    
    # =========================================================================
    # STEP 5: EXPORT CUT LIST
    # =========================================================================
    print_header("STEP 5: Export Cut List")
    
    cutlist_path = "artifacts/canopy_cutlist.csv"
    cutlist_data = export_cutlist(nodes, bars, forces, cutlist_path)
    
    # Length bin analysis
    from mini_branch.generative.canopy import compute_member_lengths, compute_length_bins
    lengths_list = compute_member_lengths(nodes, bars)
    bins = compute_length_bins(lengths_list, tolerance=0.010)  # 10mm tolerance
    
    print(f"\n    Length bins (10mm tolerance):")
    for bin_name, bar_ids in bins.items():
        print(f"      {bin_name}: {len(bar_ids)} bars")
    
    print(f"\n    Unique length bins: {len(bins)}")
    print(f"    (Fewer bins = easier fabrication)")
    
    # =========================================================================
    # STEP 6: 3D VISUALIZATION
    # =========================================================================
    print_header("STEP 6: 3D Visualization")
    
    try:
        from mini_branch.viz import plot_spaceframe_3d
        
        viz_path = "artifacts/canopy_3d.html"
        
        fig = plot_spaceframe_3d(
            nodes=nodes,
            bars=bars,
            forces=forces,
            fixed_nodes=support_nodes,
            title=f"Canopy: {params.width}x{params.depth}m, {params.heightfield} heightfield",
            outpath=viz_path,
            show=False,  # Don't open browser automatically
            color_by='force',
        )
        
        print(f"\n    Interactive 3D view saved to: {viz_path}")
        print("    Open in browser to rotate/zoom/pan")
        
    except ImportError as e:
        print(f"\n    Plotly not installed, skipping 3D visualization")
        print(f"    Install with: pip install plotly")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY")
    
    print(f"""
    CANOPY DESIGN COMPLETE
    
    Structure:
      Footprint:        {params.width} x {params.depth} m
      Height:           {params.min_height} - {params.max_height} m  
      Nodes:            {n_nodes}
      Bars:             {n_bars}
      Supports:         {n_supports} edge nodes
    
    Performance:
      Max displacement: {max_total*1000:.2f} mm
      Max tension:      {max_tension/1000:.1f} kN
      Max compression:  {abs(max_compression)/1000:.1f} kN
    
    Fabrication:
      Length bins:      {len(bins)} unique lengths
      Longest member:   {max(lengths)*1000:.0f} mm
      Shortest member:  {min(lengths)*1000:.0f} mm
    
    Outputs:
      Cut list:         {cutlist_path}
      3D visualization: artifacts/canopy_3d.html
    
    Design-to-fabrication workflow complete!
    """)
    
    return nodes, bars, d, forces


if __name__ == "__main__":
    main()

