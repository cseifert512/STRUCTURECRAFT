# File: demos/run_postprocessing_demo.py
"""
DEMO: Postprocessing End Forces (N/V/M)
========================================

This demo shows how to use element_end_forces_local to compute
internal forces after solving a structure. It demonstrates:

1. How to extract end forces for each element
2. How to find maximum moments/shear forces
3. How to interpret the results
4. How to visualize the forces
"""

import numpy as np
import matplotlib.pyplot as plt

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear
from mini_branch.post import element_end_forces_local


def main():
    """
    PSEUDOCODE:
    ==========
    
    STEP 1: Create and solve a beam with UDL
    STEP 2: Postprocess to get end forces for each element
    STEP 3: Extract and display key results (max moment, max shear)
    STEP 4: Visualize the forces
    """
    
    print("=" * 70)
    print("DEMO: Postprocessing End Forces (N/V/M)")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: CREATE AND SOLVE A BEAM WITH UDL
    # ========================================================================
    print("STEP 1: Creating and Solving Beam with UDL")
    print("-" * 70)
    
    L = 4.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    w = -1000.0  # UDL downward
    
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
    
    print(f"Beam solved: {num_elements} elements, {num_nodes} nodes")
    print()
    
    # ========================================================================
    # STEP 2: POSTPROCESS TO GET END FORCES
    # ========================================================================
    print("STEP 2: Postprocessing End Forces")
    print("-" * 70)
    
    # Store results for each element
    element_forces = []
    max_moments = []
    max_shears = []
    
    for element in elements:
        # Get UDL for this element
        udl_w = element_udls.get(element.id, None)
        
        # Compute end forces
        f_local = element_end_forces_local(nodes, element, d, udl_w=udl_w)
        
        # Extract forces
        Ni = f_local[0]  # Axial at node i
        Vi = f_local[1]  # Shear at node i
        Mi = f_local[2]  # Moment at node i
        Nj = f_local[3]  # Axial at node j
        Vj = f_local[4]  # Shear at node j
        Mj = f_local[5]  # Moment at node j
        
        # Store results
        element_forces.append({
            'element_id': element.id,
            'Ni': Ni, 'Vi': Vi, 'Mi': Mi,
            'Nj': Nj, 'Vj': Vj, 'Mj': Mj,
        })
        
        # Find max moment and shear for this element
        max_M = max(abs(Mi), abs(Mj))
        max_V = max(abs(Vi), abs(Vj))
        max_moments.append(max_M)
        max_shears.append(max_V)
    
    
    print(f"Computed end forces for {len(elements)} elements")
    print()
    
    # ========================================================================
    # EXTRACT NODE VALUES FOR DIAGRAMS (do this BEFORE printing)
    # ========================================================================
    # Extract node positions
    xs = np.array([nodes[i].x for i in range(num_nodes)])
    
    # Extract moments and shears at each node for diagrams
    # Strategy: For shear diagrams, show the shear on the LEFT side of each node
    # - At left support: use Vi from element 0
    # - At internal nodes: use Vi from the element that STARTS at that node
    #   (This ensures continuity and correct sign convention)
    # - At right support: use Vj from last element, then negate for diagram convention
    node_moments = np.zeros(num_nodes)
    node_shears = np.zeros(num_nodes)
    
    # Left end node (node 0): use Vi from element 0
    node_moments[0] = element_forces[0]['Mi']
    node_shears[0] = element_forces[0]['Vi']
    
    # Internal nodes: use Vi from the element that starts at that node
    # Since elements are sequential: element i goes from node i to node i+1
    # So for node j, we use Vi from element j (which starts at node j)
    for j in range(1, last_node):  # Internal nodes only (skip 0 and last_node)
        if j < len(elements):  # Make sure element j exists
            node_moments[j] = element_forces[j]['Mi']
            node_shears[j] = element_forces[j]['Vi']
    
    # Right end node: use Vj from last element, then negate for diagram convention
    node_moments[last_node] = element_forces[-1]['Mj']
    node_shears[last_node] = -element_forces[-1]['Vj']
    
    # ========================================================================
    # STEP 3: EXTRACT AND DISPLAY KEY RESULTS
    # ========================================================================

    print("STEP 3: Key Results")
    print("-" * 70)
    
    # Find maximum moment across all elements
    max_moment_overall = max(max_moments)
    max_moment_element = np.argmax(max_moments)
    
    # Find maximum shear across all elements
    max_shear_overall = max(max_shears)
    max_shear_element = np.argmax(max_shears)
    
    print(f"Maximum Moment:")
    print(f"  Value: {max_moment_overall:.2f} N·m")
    print(f"  Location: Element {max_moment_element}")
    print(f"  Expected (theory): {abs(w) * L**2 / 8:.2f} N·m (at midspan)")
    
    print(f"\nMaximum Shear:")
    print(f"  Value: {max_shear_overall:.2f} N")
    print(f"  Location: Element {max_shear_element}")
    print(f"  Expected (theory): {abs(w) * L / 2:.2f} N (at supports)")
    
    # ========================================================================
    # DISPLAY NODE VALUES (what's actually in the diagrams)
    # ========================================================================
    mid_node = num_nodes // 2
    print(f"\nNode Values (Used in Diagrams):")
    print(f"  Node 0 (left support, x=0.0 m):")
    print(f"    V = {node_shears[0]:.2f} N, M = {node_moments[0]:.2f} N·m")
    print(f"  Node {mid_node} (midspan, x={nodes[mid_node].x:.2f} m):")
    print(f"    V = {node_shears[mid_node]:.2f} N, M = {node_moments[mid_node]:.2f} N·m")
    print(f"  Node {last_node} (right support, x={nodes[last_node].x:.2f} m):")
    print(f"    V = {node_shears[last_node]:.2f} N, M = {node_moments[last_node]:.2f} N·m")
    
    # Verify symmetry
    print(f"\nSymmetry Check:")
    print(f"  Left shear: {node_shears[0]:.2f} N (expected: +2000 N)")
    print(f"  Right shear: {node_shears[last_node]:.2f} N (expected: -2000 N)")
    print(f"  Anti-symmetric: {abs(node_shears[0] + node_shears[last_node]) < 1e-6} ✓")
    print(f"  Midspan shear: {node_shears[mid_node]:.2f} N (expected: ~0 N)")
    print(f"  Midspan moment: {node_moments[mid_node]:.2f} N·m (expected: -2000 N·m)")
    
    # Optional: Show element end forces for reference
    mid_element = num_elements // 2
    print(f"\nElement End Forces (for reference):")
    print(f"  Element 0:")
    print(f"    Node i (0): V = {element_forces[0]['Vi']:.2f} N, M = {element_forces[0]['Mi']:.2f} N·m")
    print(f"    Node j (1): V = {element_forces[0]['Vj']:.2f} N, M = {element_forces[0]['Mj']:.2f} N·m")
    print(f"  Element {mid_element}:")
    print(f"    Node i ({mid_element}): V = {element_forces[mid_element]['Vi']:.2f} N, M = {element_forces[mid_element]['Mi']:.2f} N·m")
    print(f"    Node j ({mid_element+1}): V = {element_forces[mid_element]['Vj']:.2f} N, M = {element_forces[mid_element]['Mj']:.2f} N·m")
    print(f"  Element {num_elements-1}:")
    print(f"    Node i ({num_elements-1}): V = {element_forces[-1]['Vi']:.2f} N, M = {element_forces[-1]['Mi']:.2f} N·m")
    print(f"    Node j ({num_elements}): V = {element_forces[-1]['Vj']:.2f} N, M = {element_forces[-1]['Mj']:.2f} N·m")
    print()
    
    # ========================================================================
    # STEP 4: VISUALIZE THE FORCES
    # ========================================================================
    print("STEP 4: Creating Visualization")
    print("-" * 70)
    

    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Moment diagram
    ax1.plot(xs, node_moments, 'b-o', linewidth=2, markersize=6, label='Moment')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('Moment (N·m)', fontsize=11)
    ax1.set_title('Bending Moment Diagram', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add max moment annotation
    max_moment_node = np.argmax(np.abs(node_moments))
    ax1.annotate(f'Max |M| = {max_moment_overall:.2f} N·m',
                xy=(xs[max_moment_node], node_moments[max_moment_node]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Shear diagram
    ax2.plot(xs, node_shears, 'r-o', linewidth=2, markersize=6, label='Shear')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('x (m)', fontsize=11)
    ax2.set_ylabel('Shear (N)', fontsize=11)
    ax2.set_title('Shear Force Diagram', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add max shear annotation
    max_shear_node = np.argmax(np.abs(node_shears))
    ax2.annotate(f'Max |V| = {max_shear_overall:.2f} N',
                xy=(xs[max_shear_node], node_shears[max_shear_node]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization created!")
    print("  Top plot: Bending moment diagram")
    print("  Bottom plot: Shear force diagram")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("KEY TAKEAWAYS:")
    print("1. element_end_forces_local computes N/V/M from displacements")
    print("2. Must subtract UDL equivalent loads to get actual forces")
    print("3. Results are in LOCAL element coordinates")
    print("4. Can extract max moments/shear for design checks")
    print()


if __name__ == "__main__":
    main()