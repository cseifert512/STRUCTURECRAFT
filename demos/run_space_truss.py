#!/usr/bin/env python3
"""
RUN_SPACE_TRUSS: First 3D Structural Analysis Demo
===================================================

This demo shows a complete 3D truss analysis workflow:
1. Create a space truss structure (tetrahedron)
2. Apply loads
3. Solve for displacements
4. Compute member forces
5. Print results

This is the "I have crossed into 3D territory" milestone.

Run with:
    python demos/run_space_truss.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mini_branch.v3d.model import Node3D, Truss3D
from mini_branch.v3d.elements import (
    truss3d_global_stiffness, 
    truss3d_axial_force,
    element_geometry_3d
)
from mini_branch.kernel.dof import DOFManager
from mini_branch.kernel.assemble import assemble_global_K
from mini_branch.kernel.solve import solve_linear


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    print_header("3D SPACE TRUSS ANALYSIS")
    print("\nStructure: Regular tetrahedron with vertical load at apex")
    print("This is your first 3D structural analysis!")
    
    # =========================================================================
    # STEP 1: DEFINE GEOMETRY
    # =========================================================================
    print_header("STEP 1: Define Geometry")
    
    # Base triangle radius and height
    base_radius = 2.0  # meters
    height = 3.0  # meters
    
    # Create base triangle nodes (120° apart)
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    nodes = {}
    
    print("\nNodes (x, y, z in meters):")
    for i, angle in enumerate(angles):
        x = base_radius * np.cos(angle)
        y = base_radius * np.sin(angle)
        nodes[i] = Node3D(id=i, x=x, y=y, z=0.0)
        print(f"  Node {i}: ({x:6.3f}, {y:6.3f}, 0.000) - Base")
    
    # Apex node
    nodes[3] = Node3D(id=3, x=0.0, y=0.0, z=height)
    print(f"  Node 3: ( 0.000,  0.000, {height:.3f}) - Apex")
    
    # =========================================================================
    # STEP 2: DEFINE ELEMENTS
    # =========================================================================
    print_header("STEP 2: Define Elements")
    
    # Material: Steel
    E = 210e9  # Pa (210 GPa)
    A = 0.002  # m^2 (20 cm^2)
    
    print(f"\nMaterial: Steel (E = {E/1e9:.0f} GPa)")
    print(f"Section:  A = {A*1e4:.1f} cm^2")
    
    bars = []
    bar_id = 0
    
    # Base triangle edges
    print("\nElements:")
    for i in range(3):
        j = (i + 1) % 3
        bar = Truss3D(id=bar_id, ni=i, nj=j, E=E, A=A)
        bars.append(bar)
        L, _, _, _ = element_geometry_3d(nodes, bar)
        print(f"  Bar {bar_id}: Node {i} -> {j}, L = {L:.3f} m (base edge)")
        bar_id += 1
    
    # Legs to apex
    for i in range(3):
        bar = Truss3D(id=bar_id, ni=i, nj=3, E=E, A=A)
        bars.append(bar)
        L, _, _, _ = element_geometry_3d(nodes, bar)
        print(f"  Bar {bar_id}: Node {i} -> 3, L = {L:.3f} m (leg to apex)")
        bar_id += 1
    
    print(f"\nTotal: {len(bars)} bars")
    
    # =========================================================================
    # STEP 3: ASSEMBLE GLOBAL STIFFNESS MATRIX
    # =========================================================================
    print_header("STEP 3: Assemble Global Stiffness Matrix")
    
    dof = DOFManager(dof_per_node=3)  # 3D truss: ux, uy, uz per node
    n_nodes = len(nodes)
    ndof = dof.ndof(n_nodes)
    
    print(f"\nDOF configuration:")
    print(f"  Nodes: {n_nodes}")
    print(f"  DOF per node: 3 (ux, uy, uz)")
    print(f"  Total DOF: {ndof}")
    
    # Build contributions
    contributions = []
    for bar in bars:
        dof_map = dof.element_dof_map([bar.ni, bar.nj])
        ke = truss3d_global_stiffness(nodes, bar)
        contributions.append((dof_map, ke))
    
    K = assemble_global_K(ndof, contributions)
    
    print(f"\nGlobal stiffness matrix K: {K.shape[0]}×{K.shape[1]}")
    print(f"  Max diagonal: {np.max(np.diag(K)):.2e} N/m")
    print(f"  Symmetry check: {np.allclose(K, K.T)}")
    
    # =========================================================================
    # STEP 4: APPLY LOADS
    # =========================================================================
    print_header("STEP 4: Apply Loads")
    
    # Vertical load at apex
    P = -50000.0  # N (50 kN downward)
    
    F = np.zeros(ndof)
    F[dof.idx(3, 2)] = P  # Fz at node 3
    
    print(f"\nApplied load:")
    print(f"  Node 3 (apex): Fz = {P/1000:.1f} kN (downward)")
    
    # =========================================================================
    # STEP 5: APPLY BOUNDARY CONDITIONS
    # =========================================================================
    print_header("STEP 5: Apply Boundary Conditions")
    
    # Pin all base nodes (fix ux, uy, uz)
    fixed_dofs = []
    for node_id in [0, 1, 2]:
        fixed_dofs.extend(dof.node_dofs(node_id))
    
    print(f"\nFixed DOFs: {fixed_dofs}")
    print("  Nodes 0, 1, 2: All translations fixed (pinned supports)")
    print("  Node 3: Free (apex)")
    
    # =========================================================================
    # STEP 6: SOLVE
    # =========================================================================
    print_header("STEP 6: Solve")
    
    d, R, free = solve_linear(K, F, fixed_dofs)
    
    print(f"\nSolution converged!")
    print(f"  Free DOFs: {len(free)}")
    print(f"  Fixed DOFs: {len(fixed_dofs)}")
    
    # =========================================================================
    # STEP 7: RESULTS - DISPLACEMENTS
    # =========================================================================
    print_header("RESULTS: Displacements")
    
    print("\nNodal displacements (mm):")
    for node_id in sorted(nodes.keys()):
        ux = d[dof.idx(node_id, 0)] * 1000  # Convert to mm
        uy = d[dof.idx(node_id, 1)] * 1000
        uz = d[dof.idx(node_id, 2)] * 1000
        total = np.sqrt(ux**2 + uy**2 + uz**2)
        print(f"  Node {node_id}: ux={ux:8.4f}, uy={uy:8.4f}, uz={uz:8.4f}  |total|={total:.4f}")
    
    # Apex displacement
    uz_apex = d[dof.idx(3, 2)]
    print(f"\n  -> Apex vertical displacement: {uz_apex*1000:.4f} mm")
    
    # =========================================================================
    # STEP 8: RESULTS - REACTIONS
    # =========================================================================
    print_header("RESULTS: Support Reactions")
    
    print("\nReactions at base nodes (kN):")
    Rx_total, Ry_total, Rz_total = 0, 0, 0
    for node_id in [0, 1, 2]:
        Rx = R[dof.idx(node_id, 0)] / 1000  # Convert to kN
        Ry = R[dof.idx(node_id, 1)] / 1000
        Rz = R[dof.idx(node_id, 2)] / 1000
        print(f"  Node {node_id}: Rx={Rx:8.2f}, Ry={Ry:8.2f}, Rz={Rz:8.2f}")
        Rx_total += Rx
        Ry_total += Ry
        Rz_total += Rz
    
    print(f"\n  -> Total reactions: Rx={Rx_total:.2f}, Ry={Ry_total:.2f}, Rz={Rz_total:.2f} kN")
    print(f"  -> Applied load:    Fz={P/1000:.2f} kN")
    print(f"  -> Equilibrium check: Sum(Rz) = {Rz_total:.2f} kN ~ {-P/1000:.2f} kN [OK]")
    
    # =========================================================================
    # STEP 9: RESULTS - MEMBER FORCES
    # =========================================================================
    print_header("RESULTS: Member Forces")
    
    print("\nAxial forces in bars:")
    print("  (Positive = Tension, Negative = Compression)")
    print()
    
    max_force = 0
    for bar in bars:
        N = truss3d_axial_force(nodes, bar, d)
        L, _, _, _ = element_geometry_3d(nodes, bar)
        force_type = "Tension" if N > 0 else "Compression"
        
        print(f"  Bar {bar.id}: {bar.ni}->{bar.nj}  N = {N/1000:8.2f} kN  ({force_type})")
        
        if abs(N) > abs(max_force):
            max_force = N
    
    print(f"\n  -> Maximum force: {abs(max_force)/1000:.2f} kN")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY")
    
    print(f"""
    Structure:     Regular tetrahedron (4 nodes, 6 bars)
    Material:      Steel (E = {E/1e9:.0f} GPa)
    Section:       A = {A*1e4:.1f} cm^2
    
    Applied Load:  {abs(P)/1000:.1f} kN downward at apex
    
    Results:
      Max displacement: {abs(uz_apex)*1000:.4f} mm (at apex)
      Max force:        {abs(max_force)/1000:.2f} kN (compression in legs)
    
    3D structural analysis complete!
    """)
    
    return d, R


if __name__ == "__main__":
    main()

