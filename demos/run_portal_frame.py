# File: demos/run_portal_frame.py
"""
DEMO: PORTAL FRAME (GRAVITY + LATERAL LOADS)
============================================

PURPOSE:
--------
This demo demonstrates analyzing a real-world structural system: a portal frame
subjected to both gravity loads (beam weight) and lateral loads (wind/seismic).

This is the culmination of Day 2 - combining everything we've built:
- UDL loads (from loads.py)
- Postprocessing (from post.py)
- Multi-element structures
- Engineering metrics (drift, reactions, moments)

PHYSICAL PROBLEM:
---------------
Imagine a simple building frame:
- Two columns support a horizontal beam (roof)
- Gravity: The beam has weight (snow, equipment) = UDL downward
- Lateral: Wind pushes horizontally on the frame = point load at top

We want to know:
- How much does the frame sway sideways? (drift - critical for building codes)
- What forces do the supports push back with? (reactions - for foundation design)
- What are the maximum moments? (for sizing the beams/columns)

ENGINEERING SIGNIFICANCE:
-------------------------
Portal frames are THE fundamental building block of structural engineering.
Every building, bridge, and frame structure uses this concept. By solving this,
we've built a solver that can handle real-world problems.

THEORETICAL BACKGROUND:
----------------------
For a portal frame with:
- Height H, Width L
- Gravity UDL w on beam
- Lateral load P at top

Key behaviors:
- Lateral load causes "sway" - frame leans to one side
- Gravity load causes beam to sag (bending)
- Combined: frame deforms in both directions simultaneously
- Drift is typically limited to H/400 or H/500 by building codes
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
    PSEUDOCODE OVERVIEW:
    ====================
    
    STEP 1: Define the physical problem (geometry, loads, material properties)
    STEP 2: Create the finite element model (4 nodes, 3 elements)
    STEP 3: Build the stiffness matrix K
    STEP 4: Assemble load vector F (UDL on beam + lateral point load)
    STEP 5: Define boundary conditions (pinned bases)
    STEP 6: Solve: K × d = F
    STEP 7: Postprocess: extract drift, reactions, moments
    STEP 8: Visualize: plot undeformed vs deformed frame
    """
    
    print("=" * 70)
    print("DEMO: PORTAL FRAME (GRAVITY + LATERAL LOADS)")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: DEFINE THE PHYSICAL PROBLEM
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 1:
    ---------------------
    
    DEFINE geometry:
        L = frame width (beam span, meters)
        H = frame height (column height, meters)
    
    DEFINE material properties:
        E = Young's modulus (material stiffness, Pascals)
        I = Moment of inertia (bending resistance, m⁴)
        A = Cross-sectional area (m²)
    
    DEFINE loads:
        w = UDL on beam (N/m, negative = downward gravity)
        P = Lateral point load (N, positive = push to the right)
    
    ENGINEERING CONTEXT:
    -------------------
    - L = 6.0 m: Typical building bay width
    - H = 3.0 m: Typical story height
    - w = -2000 N/m: Beam weight + snow load (about 200 kg/m)
    - P = 5000 N: Wind load (about 500 kg horizontal push)
    
    These are realistic values for a small building frame.
    """
    
    # Geometry
    L = 6.0          # Frame width / beam span (meters)
    H = 3.0          # Frame height / column height (meters)
    
    # Material properties (steel)
    E = 210e9        # Young's modulus (Pascals) - Steel stiffness
    I = 8.0e-6       # Moment of inertia (m⁴) - Bending resistance
    A = 0.01         # Cross-sectional area (m²)
    
    # Loads
    w = -2000.0      # UDL on beam (N/m, negative = downward)
    P = 5000.0       # Lateral point load (N, positive = push right)
    
    print("STEP 1: Physical Problem Definition")
    print("-" * 70)
    print(f"Geometry:")
    print(f"  Frame width (L): {L:.1f} m")
    print(f"  Frame height (H): {H:.1f} m")
    print(f"  Aspect ratio: {L/H:.2f}")
    print()
    print(f"Material (Steel):")
    print(f"  Young's modulus (E): {E/1e9:.0f} GPa")
    print(f"  Moment of inertia (I): {I*1e6:.2f} × 10⁻⁶ m⁴")
    print(f"  Cross-sectional area (A): {A:.3f} m²")
    print()
    print(f"Loads:")
    print(f"  Beam UDL (gravity): {w:.0f} N/m (downward)")
    print(f"  Total beam load: {abs(w)*L:.0f} N")
    print(f"  Lateral load: {P:.0f} N (horizontal push)")
    print()
    
    # ========================================================================
    # STEP 2: CREATE THE FINITE ELEMENT MODEL
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 2:
    ---------------------
    
    CREATE nodes (4 points forming a rectangle):
        Node 0: Left base (0, 0)      - Left column bottom
        Node 1: Left top (0, H)       - Left column top / beam left end
        Node 2: Right top (L, H)      - Right column top / beam right end
        Node 3: Right base (L, 0)     - Right column bottom
    
    CREATE elements (3 members):
        Element 0: Left column (0 → 1)   - Vertical, connects base to top
        Element 1: Beam (1 → 2)          - Horizontal, connects left to right top
        Element 2: Right column (2 → 3)  - Vertical, connects top to base
    
    ENGINEERING CONTEXT:
    -------------------
    This is the standard portal frame topology. The nodes form a rectangle,
    and the elements connect them to create a rigid frame structure.
    
    IMPORTANT: Element orientation matters!
    - Left column: goes UP (from y=0 to y=H)
    - Beam: goes RIGHT (from x=0 to x=L)
    - Right column: goes DOWN (from y=H to y=0)
    
    The local coordinate system for each element:
    - For vertical elements: local +x = global +y (upward)
    - For horizontal elements: local +x = global +x (rightward)
    """
    
    print("STEP 2: Creating Finite Element Model")
    print("-" * 70)
    
    # Create nodes
    nodes = {
        0: Node(0, 0.0, 0.0),      # Left base
        1: Node(1, 0.0, H),         # Left top
        2: Node(2, L, H),           # Right top
        3: Node(3, L, 0.0),         # Right base
    }
    
    # Create elements
    elements = [
        Frame2D(0, 0, 1, E=E, A=A, I=I),  # Left column: 0 → 1
        Frame2D(1, 1, 2, E=E, A=A, I=I),  # Beam: 1 → 2
        Frame2D(2, 2, 3, E=E, A=A, I=I),  # Right column: 2 → 3
    ]
    
    print(f"Nodes: {len(nodes)}")
    for node_id, node in nodes.items():
        print(f"  Node {node_id}: ({node.x:.1f}, {node.y:.1f}) m")
    print()
    print(f"Elements: {len(elements)}")
    print(f"  Element 0: Left column (Node {elements[0].ni} → {elements[0].nj})")
    print(f"  Element 1: Beam (Node {elements[1].ni} → {elements[1].nj})")
    print(f"  Element 2: Right column (Node {elements[2].ni} → {elements[2].nj})")
    print()
    
    # ========================================================================
    # STEP 3: BUILD STIFFNESS MATRIX
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 3:
    ---------------------
    
    ASSEMBLE global stiffness matrix:
        K = assemble_global_K(nodes, elements)
    
    This creates a matrix that describes how stiff the entire frame is.
    Each element contributes its stiffness, transformed to global coordinates
    and assembled at the shared nodes.
    
    PHYSICAL MEANING:
    ----------------
    K[i,j] = force at DOF i required to cause unit displacement at DOF j
    
    The stiffness matrix is symmetric (Maxwell's reciprocal theorem):
    - Force at A to move B = Force at B to move A
    
    SIZE:
    - 4 nodes × 3 DOF/node = 12 total DOFs
    - K is 12×12 matrix
    """
    
    print("STEP 3: Building Stiffness Matrix")
    print("-" * 70)
    
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    
    print(f"Total DOFs: {ndof} (4 nodes × 3 DOF/node)")
    print(f"Stiffness matrix size: {K.shape}")
    print(f"Matrix is symmetric: {np.allclose(K, K.T)}")
    print()
    
    # ========================================================================
    # STEP 4: ASSEMBLE LOAD VECTOR
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 4:
    ---------------------
    
    CREATE empty force vector:
        F = zeros(ndof)
    
    ADD UDL loads on beam (Element 1):
        element_udls = {1: w}  # UDL only on beam element
        F += assemble_element_loads_global(nodes, elements, element_udls)
    
    ADD lateral point load at top node:
        F[dof_index(node_1, 0)] += P  # Horizontal force at left top
    
    ENGINEERING CONTEXT:
    -------------------
    We're applying two types of loads:
    1. Gravity (UDL): Distributed along the beam (snow, weight)
    2. Lateral (point): Concentrated at the top (wind, seismic)
    
    The UDL gets converted to equivalent nodal loads automatically.
    The point load is applied directly to the node.
    
    LOAD COMBINATION:
    - This represents a typical design case: dead load (gravity) + live load (lateral)
    - Building codes require checking multiple load combinations
    - This is one of the most critical combinations
    """
    
    print("STEP 4: Assembling Load Vector")
    print("-" * 70)
    
    # Initialize force vector
    F = np.zeros(ndof)
    
    # Add UDL on beam (Element 1)
    element_udls = {1: w}  # Only beam has UDL
    F += assemble_element_loads_global(nodes, elements, element_udls)
    
    # Add lateral point load at left top (Node 1)
    # DOF 0 = horizontal (ux), DOF 1 = vertical (uy)
    F[DOF_PER_NODE * 1 + 0] += P  # Horizontal force at node 1
    
    print(f"Loads applied:")
    print(f"  UDL on beam (Element 1): {w:.0f} N/m")
    print(f"  Lateral load at Node 1: {P:.0f} N (horizontal)")
    print(f"  Total vertical load: {abs(w)*L:.0f} N")
    print()
    
    # ========================================================================
    # STEP 5: DEFINE BOUNDARY CONDITIONS
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 5:
    ---------------------
    
    PINNED BASES (standard portal frame support):
        Node 0 (left base): fix ux=0, uy=0 (can rotate)
        Node 3 (right base): fix ux=0, uy=0 (can rotate)
    
    WHY PINNED?
    ----------
    Pinned supports are common in real structures:
    - Easier to construct (no moment connection needed)
    - Allows rotation (reduces moment at base)
    - Still prevents translation (structure can't move)
    
    ALTERNATIVES:
    - Fixed bases: Would fix rotation too (ux=0, uy=0, rz=0)
      This creates larger moments at base but smaller drift
    - One pinned, one roller: Would allow horizontal expansion
      But less stable, not used for portal frames
    
    STABILITY:
    ---------
    With 2 pinned supports, we fix 4 DOFs (2 per support).
    This prevents rigid body motion (translation and rotation).
    The structure is stable and can be solved.
    """
    
    print("STEP 5: Defining Boundary Conditions")
    print("-" * 70)
    
    # Pinned bases: fix translation, allow rotation
    fixed = [
        DOF_PER_NODE * 0 + 0,  # Node 0, ux = 0
        DOF_PER_NODE * 0 + 1,  # Node 0, uy = 0
        DOF_PER_NODE * 3 + 0,  # Node 3, ux = 0
        DOF_PER_NODE * 3 + 1,  # Node 3, uy = 0
    ]
    
    print(f"Boundary conditions: Pinned bases")
    print(f"  Node 0 (left base): ux=0, uy=0 (rotation free)")
    print(f"  Node 3 (right base): ux=0, uy=0 (rotation free)")
    print(f"  Fixed DOFs: {len(fixed)}")
    print()
    
    # ========================================================================
    # STEP 6: SOLVE THE SYSTEM
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 6:
    ---------------------
    
    SOLVE: K × d = F
        d, R, _ = solve_linear(K, F, fixed)
    
    WHERE:
    - d = displacement vector (all node movements)
    - R = reaction vector (forces at supports)
    
    PHYSICAL MEANING:
    ----------------
    The solver finds the equilibrium position:
    - How much each node moves (d)
    - What forces the supports push back with (R)
    
    The structure deforms until internal forces balance external loads.
    """
    
    print("STEP 6: Solving System")
    print("-" * 70)
    
    d, R, _ = solve_linear(K, F, fixed)
    
    print("System solved successfully!")
    print()
    
    # ========================================================================
    # STEP 7: POSTPROCESS RESULTS
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 7:
    ---------------------
    
    EXTRACT displacements:
        ux_top_left = d[dof_index(1, 0)]   # Horizontal at left top
        ux_top_right = d[dof_index(2, 0)]  # Horizontal at right top
        uy_midspan = d[dof_index(1 or 2, 1)]  # Vertical at beam
    
    CALCULATE drift:
        drift = max(|ux_top_left|, |ux_top_right|)
    
    EXTRACT reactions:
        Rx_left = R[dof_index(0, 0)]   # Horizontal reaction at left base
        Ry_left = R[dof_index(0, 1)]   # Vertical reaction at left base
        Rx_right = R[dof_index(3, 0)]  # Horizontal reaction at right base
        Ry_right = R[dof_index(3, 1)]  # Vertical reaction at right base
    
    COMPUTE end moments (using post.py):
        FOR each element:
            f_local = element_end_forces_local(...)
            M_max = max(|Mi|, |Mj|)
    
    ENGINEERING METRICS:
    -------------------
    1. DRIFT: Critical for serviceability
       - Building codes limit drift to H/400 or H/500
       - Too much drift = building feels "wobbly"
       - Our frame: H = 3.0 m, so drift < 7.5 mm (H/400) is acceptable
    
    2. REACTIONS: Needed for foundation design
       - Horizontal reactions resist lateral load
       - Vertical reactions resist gravity load
       - Must sum to applied loads (equilibrium check)
    
    3. MOMENTS: Needed for member sizing
       - Maximum moment determines required beam/column size
       - Higher moment = need stronger (larger) members
    """
    
    print("STEP 7: Postprocessing Results")
    print("-" * 70)
    
    # Extract displacements
    ux_node1 = d[DOF_PER_NODE * 1 + 0]  # Left top, horizontal
    uy_node1 = d[DOF_PER_NODE * 1 + 1]  # Left top, vertical
    ux_node2 = d[DOF_PER_NODE * 2 + 0]  # Right top, horizontal
    uy_node2 = d[DOF_PER_NODE * 2 + 1]  # Right top, vertical
    
    # Calculate drift (maximum horizontal displacement at top)
    drift = max(abs(ux_node1), abs(ux_node2))
    drift_ratio = drift / H  # Drift as fraction of height
    
    # Extract reactions
    Rx_left = R[DOF_PER_NODE * 0 + 0]   # Left base, horizontal
    Ry_left = R[DOF_PER_NODE * 0 + 1]   # Left base, vertical
    Rx_right = R[DOF_PER_NODE * 3 + 0]  # Right base, horizontal
    Ry_right = R[DOF_PER_NODE * 3 + 1]  # Right base, vertical
    
    # Compute end moments for each element
    max_moments = []
    element_moments = []
    
    for i, element in enumerate(elements):
        udl_w = element_udls.get(element.id, None)
        f_local = element_end_forces_local(nodes, element, d, udl_w=udl_w)
        
        Mi = f_local[2]  # Moment at node i
        Mj = f_local[5]  # Moment at node j
        M_max = max(abs(Mi), abs(Mj))
        
        max_moments.append(M_max)
        element_moments.append({
            'element_id': element.id,
            'Mi': Mi,
            'Mj': Mj,
            'M_max': M_max
        })
    
    max_moment_overall = max(max_moments)
    
    # Print results
    print("DISPLACEMENTS:")
    print(f"  Left top (Node 1):")
    print(f"    Horizontal (ux): {ux_node1*1000:.2f} mm")
    print(f"    Vertical (uy): {uy_node1*1000:.2f} mm")
    print(f"  Right top (Node 2):")
    print(f"    Horizontal (ux): {ux_node2*1000:.2f} mm")
    print(f"    Vertical (uy): {uy_node2*1000:.2f} mm")
    print()
    print(f"DRIFT: {drift*1000:.2f} mm ({drift_ratio*1000:.2f}% of height)")
    print(f"  Code limit (H/400): {H/400*1000:.2f} mm")
    print(f"  Within limit: {'✓' if drift < H/400 else '✗'}")
    print()
    print("REACTIONS:")
    print(f"  Left base (Node 0):")
    print(f"    Horizontal (Rx): {Rx_left:.2f} N")
    print(f"    Vertical (Ry): {Ry_left:.2f} N")
    print(f"  Right base (Node 3):")
    print(f"    Horizontal (Rx): {Rx_right:.2f} N")
    print(f"    Vertical (Ry): {Ry_right:.2f} N")
    print()
    print("EQUILIBRIUM CHECK:")
    sum_Fx = Rx_left + Rx_right
    sum_Fy = Ry_left + Ry_right
    print(f"  Sum of horizontal reactions: {sum_Fx:.2f} N")
    print(f"  Applied horizontal load: {P:.2f} N")
    print(f"  Balance: {'✓' if abs(sum_Fx + P) < 1e-6 else '✗'}")
    print(f"  Sum of vertical reactions: {sum_Fy:.2f} N")
    print(f"  Applied vertical load: {abs(w)*L:.2f} N")
    print(f"  Balance: {'✓' if abs(sum_Fy - abs(w)*L) < 1e-6 else '✗'}")
    print()
    print("MAXIMUM MOMENTS:")
    for i, elem_mom in enumerate(element_moments):
        elem_name = ["Left column", "Beam", "Right column"][i]
        print(f"  {elem_name} (Element {elem_mom['element_id']}):")
        print(f"    Moment at i: {elem_mom['Mi']:.2f} N·m")
        print(f"    Moment at j: {elem_mom['Mj']:.2f} N·m")
        print(f"    Max |M|: {elem_mom['M_max']:.2f} N·m")
    print()
    print(f"Overall maximum moment: {max_moment_overall:.2f} N·m")
    print()
    
    # ========================================================================
    # STEP 8: VISUALIZE RESULTS
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 8:
    ---------------------
    
    EXTRACT coordinates:
        xs_undeformed = [node.x for all nodes]
        ys_undeformed = [node.y for all nodes]
    
    COMPUTE deformed coordinates:
        xs_deformed = xs_undeformed + [ux for all nodes]
        ys_deformed = ys_undeformed + [uy for all nodes]
    
    PLOT:
        - Undeformed frame (blue, solid)
        - Deformed frame (red, dashed, scaled)
        - Supports (triangles)
        - Load arrows
    
    SCALING:
    - Deflections are tiny (millimeters)
    - Scale them up visually (e.g., ×100 or ×1000) so we can see the deformation
    - Label the scale factor clearly
    """
    
    print("STEP 8: Creating Visualization")
    print("-" * 70)
    
    # Extract coordinates
    xs = np.array([nodes[i].x for i in range(len(nodes))])
    ys = np.array([nodes[i].y for i in range(len(nodes))])
    
    # Extract displacements
    ux = np.array([d[i*DOF_PER_NODE + 0] for i in range(len(nodes))])
    uy = np.array([d[i*DOF_PER_NODE + 1] for i in range(len(nodes))])
    
    # Scale factor for visualization (deflections are tiny!)
    scale = 50  # Multiply by 50 to make visible
    
    # Deformed coordinates
    xs_deformed = xs + scale * ux
    ys_deformed = ys + scale * uy
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot undeformed frame
    # Left column
    ax.plot([xs[0], xs[1]], [ys[0], ys[1]], 'b-', linewidth=3, label='Undeformed', alpha=0.7)
    # Beam
    ax.plot([xs[1], xs[2]], [ys[1], ys[2]], 'b-', linewidth=3, alpha=0.7)
    # Right column
    ax.plot([xs[2], xs[3]], [ys[2], ys[3]], 'b-', linewidth=3, alpha=0.7)
    
    # Plot deformed frame
    # Left column
    ax.plot([xs_deformed[0], xs_deformed[1]], 
            [ys_deformed[0], ys_deformed[1]], 
            'r--', linewidth=2, label=f'Deformed (×{scale} scale)', alpha=0.9)
    # Beam
    ax.plot([xs_deformed[1], xs_deformed[2]], 
            [ys_deformed[1], ys_deformed[2]], 
            'r--', linewidth=2, alpha=0.9)
    # Right column
    ax.plot([xs_deformed[2], xs_deformed[3]], 
            [ys_deformed[2], ys_deformed[3]], 
            'r--', linewidth=2, alpha=0.9)
    
    # Plot nodes
    ax.plot(xs, ys, 'bo', markersize=10, label='Nodes')
    ax.plot(xs_deformed, ys_deformed, 'ro', markersize=8, alpha=0.7)
    
    # Plot supports (triangles at base)
    support_size = 0.3
    ax.plot([xs[0]-support_size, xs[0], xs[0]+support_size], 
            [ys[0]-support_size, ys[0], ys[0]-support_size], 
            'k-', linewidth=2, label='Supports (pinned)')
    ax.plot([xs[3]-support_size, xs[3], xs[3]+support_size], 
            [ys[3]-support_size, ys[3], ys[3]-support_size], 
            'k-', linewidth=2)
    
    # Plot load arrows
    # UDL on beam (downward arrows)
    num_arrows = 5
    for i in range(num_arrows):
        x_arrow = xs[1] + (xs[2] - xs[1]) * (i + 0.5) / num_arrows
        y_arrow = ys[1] + 0.1
        ax.arrow(x_arrow, y_arrow, 0, -0.2, head_width=0.15, head_length=0.1,
                fc='g', ec='g', linewidth=1.5, alpha=0.7)
    
    # Lateral load (horizontal arrow at left top)
    ax.arrow(xs[1] + 0.2, ys[1], 0.5, 0, head_width=0.15, head_length=0.1,
            fc='orange', ec='orange', linewidth=2, label='Lateral load')
    
    # Add text annotations
    ax.text(L/2, H + 0.5, f'UDL: {abs(w):.0f} N/m', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(xs[1] + 0.8, ys[1] + 0.3, f'P = {P:.0f} N', 
            ha='left', fontsize=10, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # Add results text box
    results_text = f"""Results:
Drift: {drift*1000:.2f} mm
Max Moment: {max_moment_overall:.2f} N·m
Left Rx: {Rx_left:.0f} N
Right Rx: {Rx_right:.0f} N"""
    ax.text(0.02, 0.98, results_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('Portal Frame: Undeformed vs Deformed', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, H + 1.0)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization created!")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("KEY TAKEAWAYS:")
    print("1. Portal frames combine multiple elements and load types")
    print("2. Drift is a critical serviceability metric")
    print("3. Reactions must balance applied loads (equilibrium)")
    print("4. Maximum moments determine member sizing")
    print("5. This solver can now handle real-world structures!")
    print()


if __name__ == "__main__":
    main()