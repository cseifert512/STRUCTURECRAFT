import numpy as np
import matplotlib.pyplot as plt

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.solve import solve_linear

def main():
    """
    VISUAL DEMONSTRATION OF SIMPLY SUPPORTED BEAM
    =============================================
    This script does the same analysis but also draws a picture
    so you can SEE the beam bending!
    """
    
    # ========================================================================
    # SETUP: Same as the test above
    # ========================================================================
    L = 4.0  # Beam length (meters)
    E = 210e9  # Material stiffness (Pascals) - Steel
    I = 8.0e-6  # Bending resistance (m⁴)
    A = 0.01  # Cross-section area (m²)
    P = 1000.0  # Load (Newtons) - about 100 kg weight
    
    # Create the model with more elements for smoother visualization
    num_elements = 8  # More elements = smoother curve, more accurate
    num_nodes = num_elements + 1

    nodes = {}
    for i in range(num_nodes):
        x = L * i / num_elements  # Evenly spaced along beam length
        nodes[i] = Node(i, x, 0.0)

    # Create elements connecting adjacent nodes
    elements = []
    for i in range(num_elements):
        elements.append(Frame2D(i, i, i+1, E=E, A=A, I=I))

    # Find the midspan node (closest to L/2)
    mid_node = num_elements // 2
    
    # Build stiffness matrix and apply load
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    F[DOF_PER_NODE*mid_node + 1] = -P  # Load at middle, downward
    

    # Supports: prevent rigid body motion
    # Left support: fix horizontal AND vertical (prevents sliding)
    # Right support: fix only vertical (allows horizontal expansion/contraction)
    # Both supports can still rotate (pinned supports)
    last_node = num_nodes - 1  # Last node (right support)
    fixed = [
        DOF_PER_NODE*0 + 0,  # Node 0, ux = 0 (prevents horizontal translation)
        DOF_PER_NODE*0 + 1,  # Node 0, uy = 0 (prevents vertical movement)
        DOF_PER_NODE*last_node + 1,  # Last node, uy = 0 (prevents vertical movement)
    ]

    # Solve!
    d, R, _ = solve_linear(K, F, fixed)

    # ========================================================================
    # EXTRACT RESULTS
    # ========================================================================
    Ry_left = R[DOF_PER_NODE*0 + 1]    # Left support pushes up with this force
    Ry_right = R[DOF_PER_NODE*last_node + 1]   # Right support pushes up with this force
    uy_midspan = d[DOF_PER_NODE*mid_node + 1] # How much middle moved down
        
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    print("Simply Supported Beam - Midspan Point Load")
    print("=" * 50)
    print(f"Left reaction (N): {Ry_left:.2f}")
    print(f"Right reaction (N): {Ry_right:.2f}")
    print(f"Midspan deflection (m): {uy_midspan:.6f}")
    print()
    print("Expected (from textbook):")
    print(f"Reactions: {P/2:.2f} N each (each support takes half)")
    print(f"Max deflection: {-P * L**3 / (48 * E * I):.6f} m")
    
    # ========================================================================
    # DRAW THE PICTURE
    # ========================================================================
    # Get x and y coordinates of ALL nodes (not just first 3!)
    xs = np.array([nodes[i].x for i in range(num_nodes)])
    ys = np.array([nodes[i].y for i in range(num_nodes)])

    # Get vertical displacements for ALL nodes
    uy = np.array([d[i*DOF_PER_NODE + 1] for i in range(num_nodes)])
    # Scale up the deflection so we can see it (deflections are tiny!)
    scale = 1000  # Multiply by 1000 to make visible
    
    # Create the plot
    plt.figure(figsize=(10, 4))
    
    # Draw original (straight) beam
    plt.plot(xs, ys, 'b-o', label='Undeformed (original)', linewidth=2, markersize=8)
    
    # Draw deformed (bent) beam
    # ys + scale*uy means: original y position + (scaled displacement)
    plt.plot(xs, ys + scale*uy, 'r-o', label=f'Deformed (×{scale} scale)', 
             linewidth=2, markersize=8)
    
    # Draw a reference line at y=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Reference')
    
    # Make it pretty
    plt.legend()
    plt.title("Simply Supported Beam - Midspan Point Load")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.show()
    
    # WHAT YOU'LL SEE:
    # - Blue line: straight beam (before loading)
    # - Red line: bent beam (after loading) - middle sags down
    # The red line is exaggerated 1000× so you can see the bending

if __name__ == "__main__":
    main()