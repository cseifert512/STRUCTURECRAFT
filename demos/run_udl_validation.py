# File: demos/run_udl_validation.py
"""
Simple demo showing UDL validation against theoretical formula.
This is a visual version of the test_simply_supported_udl.py test.

MESH VISUALIZATION:
This demo clearly shows the finite element mesh structure:
- Nodes (N0, N1, N2, ...) are labeled on the beam
- Elements (E0, E1, E2, ...) are labeled below the beam
This makes it clear that the beam is divided into multiple elements
connected at nodes, not a single continuous piece.
"""

import numpy as np
import matplotlib.pyplot as plt

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear

def main():
    print("=" * 60)
    print("UDL Validation Demo")
    print("=" * 60)
    
    # Setup (same as test)
    L = 4.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    w0 = 1000.0
    w = -w0
    
    # Create meshed beam
    num_elements = 10
    num_nodes = num_elements + 1
    
    nodes = {}
    for i in range(num_nodes):
        x = L * i / num_elements
        nodes[i] = Node(i, x, 0.0)
    
    elements = []
    for i in range(num_elements):
        elements.append(Frame2D(i, i, i+1, E=E, A=A, I=I))
    
    # Solve
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    element_udls = {i: w for i in range(num_elements)}
    F += assemble_element_loads_global(nodes, elements, element_udls)
    
    last_node = num_nodes - 1
    fixed = [
        DOF_PER_NODE * 0 + 0,
        DOF_PER_NODE * 0 + 1,
        DOF_PER_NODE * last_node + 1,
    ]
    
    d, R, _ = solve_linear(K, F, fixed)
    
    # Extract results
    mid_node = num_nodes // 2
    uy_midspan = d[DOF_PER_NODE * mid_node + 1]
    delta_max_theory = 5 * w * L**4 / (384 * E * I)
    Ry_left = R[DOF_PER_NODE * 0 + 1]
    Ry_right = R[DOF_PER_NODE * last_node + 1]
    
    # Print results
    print(f"\nResults:")
    print(f"  Midspan deflection: {uy_midspan:.6f} m")
    print(f"  Theoretical:        {delta_max_theory:.6f} m")
    print(f"  Error:              {abs(uy_midspan - delta_max_theory):.2e} m")
    print(f"  Relative error:     {abs(uy_midspan - delta_max_theory) / abs(delta_max_theory) * 100:.2f}%")
    print(f"  Left reaction:      {Ry_left:.2f} N")
    print(f"  Right reaction:     {Ry_right:.2f} N")
    
    # Verify (like the test does)
    error_pct = abs(uy_midspan - delta_max_theory) / abs(delta_max_theory) * 100
    if error_pct < 2.0:
        print(f"\n✓ Validation PASSED (error < 2%)")
    else:
        print(f"\n✗ Validation FAILED (error = {error_pct:.2f}%)")
    
    # Visualize
    xs = np.array([nodes[i].x for i in range(num_nodes)])
    ys = np.array([nodes[i].y for i in range(num_nodes)])
    uy_all = np.array([d[i * DOF_PER_NODE + 1] for i in range(num_nodes)])
    
    scale = 1000
    ys_deformed = ys + scale * uy_all
    
    # Create figure with more space for labels
    plt.figure(figsize=(14, 8))
    
    # Plot original (undeformed) beam
    plt.plot(xs, ys, 'b-o', label='Undeformed', linewidth=2, markersize=6)
    
    # Plot deformed beam (scaled up)
    plt.plot(xs, ys_deformed, 'r-o', label=f'Deformed (×{scale} scale, exaggerated)', 
             linewidth=2, markersize=6)
    
    # Add support symbols
    plt.plot(nodes[0].x, 0, 'g^', markersize=15, label='Support (pinned)')
    plt.plot(nodes[last_node].x, 0, 'g^', markersize=15)
    
    # Add reference line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # ========================================================================
    # MESH VISUALIZATION: Node Labels
    # ========================================================================
    # Label all nodes on the undeformed beam (blue labels)
    for i in range(num_nodes):
        plt.text(xs[i], ys[i] + 0.08, f'N{i}', 
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                weight='bold')
    
    # Label key nodes on deformed beam (red labels) - only important ones to avoid clutter
    for i in [0, mid_node, last_node]:
        plt.text(xs[i], ys_deformed[i] - 0.15, f'N{i}', 
                ha='center', va='top', fontsize=8, color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8),
                weight='bold')
    
    # ========================================================================
    # MESH VISUALIZATION: Element Labels
    # ========================================================================
    # Label elements below the undeformed beam
    for i in range(num_elements):
        # Calculate midpoint of element
        x_mid = (nodes[i].x + nodes[i+1].x) / 2
        y_mid = -0.25  # Position below the beam
        
        # Create element label
        plt.text(x_mid, y_mid, f'E{i}', 
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                weight='bold')
        
        # Draw a light line connecting element label to element midpoint (optional)
        plt.plot([x_mid, x_mid], [y_mid + 0.05, ys[0] - 0.03], 
                'k--', linewidth=0.5, alpha=0.2)
    
    # ========================================================================
    # Mesh Information Text
    # ========================================================================
    mesh_info = f'Mesh: {num_elements} elements, {num_nodes} nodes'
    plt.text(0.5, 0.96, mesh_info, transform=plt.gca().transAxes,
             ha='center', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    # ========================================================================
    # Formatting and Labels
    # ========================================================================
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('UDL Validation: Finite Element Mesh Visualization', 
              fontsize=13, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis limits to accommodate labels
    y_min = min(np.min(ys_deformed) - 0.3, -0.4)
    y_max = max(np.max(ys) + 0.2, 0.15)
    plt.ylim(y_min, y_max)
    
    # Add result text box (bottom left)
    textstr = f'Results:\n'
    textstr += f'Deflection: {uy_midspan:.6f} m\n'
    textstr += f'Theory: {delta_max_theory:.6f} m\n'
    textstr += f'Error: {error_pct:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    plt.text(0.02, 0.05, textstr, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom', bbox=props)
    
    # Add legend explaining labels
    legend_text = 'Mesh Labels:\nN0-N10 = Nodes\nE0-E9 = Elements'
    plt.text(0.98, 0.05, legend_text, transform=plt.gca().transAxes,
             fontsize=8, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Print mesh information
    print(f"\nMesh Information:")
    print(f"  Elements: {num_elements}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Element length: {L/num_elements:.3f} m")
    print(f"  Midspan node: N{mid_node} (at x = {nodes[mid_node].x:.2f} m)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()