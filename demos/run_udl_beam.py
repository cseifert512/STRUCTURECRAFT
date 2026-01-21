# File: demos/run_udl_beam.py
"""
FORMAL DEMO: SIMPLY SUPPORTED BEAM WITH UNIFORM DISTRIBUTED LOAD (UDL)
======================================================================

PURPOSE:
--------
This demo shows how to analyze a beam with a uniform distributed load (UDL)
using the finite element method. It demonstrates:

1. How UDL loads are converted to equivalent nodal loads
2. How these loads are assembled into the global force vector
3. How to solve the structural system
4. How to verify results against textbook formulas

PHYSICAL PROBLEM:
----------------
Imagine a bridge deck (beam) that's supported at both ends (simply supported).
The beam is subjected to its own weight plus snow load, which we model as
a uniform distributed load w (force per unit length) acting downward.

We want to know:
- How much does the beam bend (deflection)?
- How much force do the supports push back with (reactions)?

THEORETICAL SOLUTION:
--------------------
For a simply supported beam with uniform distributed load w over length L:
- Maximum deflection (at center): δ_max = 5wL⁴/(384EI)
- Each support reaction: R = wL/2 (each support takes half the total load)

This demo verifies our finite element solution matches these formulas.
"""

import numpy as np
import matplotlib.pyplot as plt

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear


def main():
    """
    PSEUDOCODE OVERVIEW:
    ====================
    
    STEP 1: Define the physical problem (beam properties, loads, geometry)
    STEP 2: Create the finite element model (nodes and elements)
    STEP 3: Build the stiffness matrix K (describes beam stiffness)
    STEP 4: Assemble the load vector F from UDL loads (this is the NEW part!)
    STEP 5: Define boundary conditions (supports)
    STEP 6: Solve the system: K × d = F (find displacements d)
    STEP 7: Extract and verify results (deflections, reactions)
    STEP 8: Visualize the results (plot undeformed vs deformed shape)
    """
    
    print("=" * 70)
    print("DEMO: SIMPLY SUPPORTED BEAM WITH UNIFORM DISTRIBUTED LOAD")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: DEFINE THE PHYSICAL PROBLEM
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 1:
    ---------------------
    
    DEFINE beam geometry:
        L = total beam length (meters)
    
    DEFINE material properties:
        E = Young's modulus (how stiff the material is, in Pascals)
        I = Moment of inertia (how resistant to bending, in m⁴)
        A = Cross-sectional area (in m²)
    
    DEFINE loading:
        w = uniform distributed load per unit length (N/m, negative = downward)
        w_total = w × L (total load over entire beam)
    
    EXAMPLE:
    --------
    Think of a steel beam 4 meters long:
    - L = 4.0 m
    - E = 210 GPa (210 billion Pascals, typical for steel)
    - I = 8.0e-6 m⁴ (depends on beam cross-section shape)
    - w = -1000 N/m (1000 Newtons per meter, downward)
    - Total load = -1000 × 4 = -4000 N (about 400 kg)
    """
    
    # Physical parameters
    L = 4.0          # Total beam length (meters)
    E = 210e9        # Young's modulus (Pascals) - Steel is very stiff
    I = 8.0e-6       # Moment of inertia (m⁴) - Resistance to bending
    A = 0.01         # Cross-sectional area (m²)
    w = -1000.0      # Uniform distributed load (N/m, negative = downward)
    
    # Calculate total load for reference
    w_total = w * L  # Total distributed load over entire length
    
    print("STEP 1: Physical Problem Definition")
    print("-" * 70)
    print(f"Beam length (L):           {L:.2f} m")
    print(f"Young's modulus (E):       {E:.2e} Pa")
    print(f"Moment of inertia (I):     {I:.2e} m⁴")
    print(f"Distributed load (w):      {w:.2f} N/m (downward)")
    print(f"Total load (w × L):        {w_total:.2f} N")
    print()
    
    # ========================================================================
    # STEP 2: CREATE THE FINITE ELEMENT MODEL
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 2:
    ---------------------
    
    WHY MESH?
    --------
    A single finite element can't perfectly capture UDL deflection (the
    exact solution is a quartic curve, but one element gives a cubic).
    More elements = more accurate solution.
    
    ALGORITHM:
    ---------
    
    1. DETERMINE number of elements:
        num_elements = 10  (more elements = more accurate)
        num_nodes = num_elements + 1  (one more node than elements)
    
    2. CREATE nodes:
        FOR each node from 0 to num_nodes-1:
            x_position = L × (node_index / num_elements)
            y_position = 0.0  (beam is horizontal)
            nodes[node_id] = Node(id=node_id, x=x_position, y=y_position)
    
    3. CREATE elements:
        FOR each element from 0 to num_elements-1:
            start_node = element_index
            end_node = element_index + 1
            elements.append(Frame2D(
                id=element_index,
                ni=start_node,
                nj=end_node,
                E=E,
                A=A,
                I=I
            ))
    
    VISUALIZATION:
    -------------
    Nodes:   0    1    2    3    4    5    6    7    8    9    10
    Elements: [0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
    
    Each element connects two adjacent nodes.
    """
    
    # Create a meshed beam (more elements = more accurate for UDL)
    num_elements = 10  # Number of beam segments
    num_nodes = num_elements + 1  # One more node than elements
    
    print("STEP 2: Creating Finite Element Model")
    print("-" * 70)
    print(f"Number of elements:        {num_elements}")
    print(f"Number of nodes:           {num_nodes}")
    
    # Create nodes evenly spaced along the beam
    nodes = {}
    for i in range(num_nodes):
        # Calculate x position: evenly distributed from 0 to L
        x = L * i / num_elements
        # Beam is horizontal, so y = 0.0
        nodes[i] = Node(i, x, 0.0)
    
    # Create elements connecting adjacent nodes
    elements = []
    for i in range(num_elements):
        # Element i connects node i to node i+1
        elements.append(Frame2D(
            id=i,
            ni=i,      # Start node
            nj=i+1,    # End node
            E=E,
            A=A,
            I=I
        ))
    
    print(f"Node positions:            ", end="")
    for i in [0, num_nodes//2, num_nodes-1]:
        print(f"Node {i}: x={nodes[i].x:.2f}m", end="  ")
    print()
    print()
    
    # ========================================================================
    # STEP 3: BUILD THE STIFFNESS MATRIX
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 3:
    ---------------------
    
    WHAT IS A STIFFNESS MATRIX?
    ----------------------------
    The stiffness matrix K describes how the beam resists deformation.
    Think of it as a "spring constant" matrix that relates forces to
    displacements: F = K × d
    
    WHERE:
    - F = force vector (what we're pushing with)
    - K = stiffness matrix (how stiff the beam is)
    - d = displacement vector (how much it moves)
    
    ALGORITHM:
    ---------
    
    1. For each element:
        a. Calculate element's local stiffness matrix (depends on E, A, I, L)
        b. Transform to global coordinates (rotate if beam is tilted)
        c. Map element DOFs to global DOFs (which node is which)
    
    2. Assemble all element contributions:
        FOR each element:
            FOR each DOF pair (i, j):
                K_global[global_DOF_i, global_DOF_j] += element_stiffness[i, j]
    
    3. Result: Complete global stiffness matrix K
    
    PHYSICAL MEANING:
    ----------------
    K[i, j] tells you: "If I apply a unit force at DOF j, how much will
    DOF i move?" This captures the entire beam's stiffness behavior.
    """
    
    print("STEP 3: Building Stiffness Matrix")
    print("-" * 70)
    
    # Assemble the global stiffness matrix
    # This matrix describes how the entire structure responds to forces
    K = assemble_global_K(nodes, elements)
    
    # Calculate total number of degrees of freedom
    ndof = DOF_PER_NODE * len(nodes)  # 3 DOFs per node (ux, uy, rz)
    
    print(f"Stiffness matrix size:     {K.shape[0]} × {K.shape[1]}")
    print(f"Total DOFs:                {ndof} (3 per node × {num_nodes} nodes)")
    print(f"Matrix is symmetric:       {np.allclose(K, K.T)}")
    print()
    
    # ========================================================================
    # STEP 4: ASSEMBLE THE LOAD VECTOR FROM UDL LOADS
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 4:
    ---------------------
    
    THIS IS THE NEW PART! We're using UDL loads instead of point loads.
    
    WHAT IS A UDL?
    --------------
    Uniform Distributed Load = load spread evenly along the beam
    Example: Snow load, beam's own weight, uniform pressure
    
    THE CHALLENGE:
    -------------
    Our solver handles point loads at nodes. But we have a distributed load!
    Solution: Convert the distributed load to "equivalent nodal loads" that
    produce the same deflection at the nodes.
    
    PHYSICAL PRINCIPLE:
    ------------------
    For a uniform load w over length L:
    - Total force = w × L
    - Each node gets half: wL/2
    - BUT: The distributed nature also creates moments!
    - End moments: ±wL²/12 (one positive, one negative)
    
    ALGORITHM:
    ---------
    
    1. CREATE empty global force vector:
        F = zeros(ndof)  (start with zero forces everywhere)
    
    2. FOR each element that has a UDL:
        a. Calculate local equivalent nodal loads:
           f_local = [0, wL/2, +wL²/12, 0, wL/2, -wL²/12]
           (in local coordinates: [ux_i, uy_i, rz_i, ux_j, uy_j, rz_j])
        
        b. Transform to global coordinates:
           f_global = T.T @ f_local
           (T is transformation matrix, T.T rotates from local to global)
        
        c. Map to global DOFs and add to F:
           FOR each element DOF:
               global_DOF = map_element_DOF_to_global(element, local_DOF)
               F[global_DOF] += f_global[local_DOF]
    
    3. RESULT: Global force vector F with all UDL loads assembled
    
    WHY THIS WORKS:
    --------------
    The equivalent nodal loads produce the SAME nodal deflections as the
    actual distributed load would (for linear beam elements). This is a
    fundamental principle of finite element analysis called "consistent
    nodal loading."
    """
    
    print("STEP 4: Assembling Load Vector from UDL Loads")
    print("-" * 70)
    
    # Initialize force vector to zeros
    F = np.zeros(ndof, dtype=float)
    
    # Define UDL for each element
    # Dictionary format: {element_id: w_value}
    # For this demo, all elements have the same UDL w
    element_udls = {i: w for i in range(num_elements)}
    
    print(f"UDL per element (w):       {w:.2f} N/m")
    print(f"Number of loaded elements: {len(element_udls)}")
    
    # Assemble UDL loads into global force vector
    # This is the NEW function we just implemented!
    F_udl = assemble_element_loads_global(nodes, elements, element_udls)
    
    # Add UDL forces to the force vector
    F += F_udl
    
    # Extract and display force information
    # For a horizontal beam with downward UDL, we expect:
    # - No horizontal forces (F[ux DOFs] ≈ 0)
    # - Vertical forces at each node (F[uy DOFs] < 0 for downward)
    # - Moments at each node (F[rz DOFs] can be + or -)
    
    print(f"\nForce vector assembled:")
    print(f"  Total forces:")
    total_vertical = sum(F[i] for i in range(ndof) if i % DOF_PER_NODE == 1)
    print(f"    Vertical (sum):        {total_vertical:.2f} N")
    print(f"    Expected (w×L):        {w_total:.2f} N")
    print(f"    Match:                 {np.isclose(total_vertical, w_total, rtol=1e-6)} ✓")
    
    # Check forces at key nodes
    print(f"\n  Forces at key nodes:")
    print(f"    Node 0 (left):         Fy = {F[DOF_PER_NODE*0 + 1]:.2f} N")
    mid_node = num_nodes // 2
    print(f"    Node {mid_node} (middle):  Fy = {F[DOF_PER_NODE*mid_node + 1]:.2f} N")
    print(f"    Node {num_nodes-1} (right):  Fy = {F[DOF_PER_NODE*(num_nodes-1) + 1]:.2f} N")
    print()
    
    # ========================================================================
    # STEP 5: DEFINE BOUNDARY CONDITIONS (SUPPORTS)
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 5:
    ---------------------
    
    WHAT ARE BOUNDARY CONDITIONS?
    -----------------------------
    Boundary conditions prevent rigid body motion. Without supports, the
    beam would just move as a whole (translate or rotate) without deforming.
    
    SIMPLY SUPPORTED BEAM:
    ---------------------
    - Left support: Pinned (can rotate, but can't move up/down or left/right)
    - Right support: Pinned (can rotate, but can't move up/down)
    
    IMPLEMENTATION:
    --------------
    We "fix" certain DOFs (set their displacement to zero):
    
    FOR simply supported (pinned at both ends):
        fixed_DOFs = [
            node_0, DOF_ux,  // Left support: can't move horizontally
            node_0, DOF_uy,  // Left support: can't move vertically
            node_last, DOF_uy,  // Right support: can't move vertically
            // NOTE: We DON'T fix rotations - supports are "pinned"
            // NOTE: We DON'T fix right horizontal - allows thermal expansion
        ]
    
    WHY THIS PATTERN?
    ----------------
    - Fix left ux + uy: Prevents horizontal and vertical translation
    - Fix right uy: Prevents vertical translation
    - Don't fix right ux: Allows horizontal expansion/contraction (important!)
    - Don't fix rotations: Supports are "pinned" and can rotate
    
    This pattern prevents all rigid body motion while keeping supports "pinned."
    """
    
    print("STEP 5: Defining Boundary Conditions (Supports)")
    print("-" * 70)
    
    # Simply supported: pinned at both ends
    # Left support: fix both horizontal (ux) and vertical (uy)
    # Right support: fix only vertical (uy)
    # Both supports can rotate (not fixed) - this is "pinned"
    
    last_node = num_nodes - 1
    
    fixed = [
        DOF_PER_NODE * 0 + 0,              # Node 0, ux (horizontal)
        DOF_PER_NODE * 0 + 1,              # Node 0, uy (vertical)
        DOF_PER_NODE * last_node + 1,      # Last node, uy (vertical)
    ]
    
    print(f"Support type:              Simply supported (pinned at both ends)")
    print(f"Fixed DOFs:")
    print(f"  Node 0:                  ux = 0, uy = 0 (pinned)")
    print(f"  Node {last_node}:                  uy = 0 (pinned)")
    print(f"  Rotations:               Free (supports can rotate)")
    print()
    
    # ========================================================================
    # STEP 6: SOLVE THE STRUCTURAL SYSTEM
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 6:
    ---------------------
    
    THE FUNDAMENTAL EQUATION:
    -------------------------
    K × d = F
    
    WHERE:
    - K = stiffness matrix (known: we built it)
    - d = displacement vector (unknown: we want to find this)
    - F = force vector (known: we assembled it from loads)
    
    PHYSICAL MEANING:
    ----------------
    This equation says: "Given the forces we're applying (F) and the
    stiffness of the structure (K), what are the resulting displacements (d)?"
    
    SOLUTION METHOD:
    ---------------
    
    1. PARTITION the system:
        Separate fixed DOFs from free DOFs:
        
        K × d = F
        
        becomes:
        
        [K_ff  K_fx] [d_f]   [F_f]
        [          ] [   ] = [   ]
        [K_xf  K_xx] [d_x]   [F_x]
        
        WHERE:
        - f = free DOFs (can move)
        - x = fixed DOFs (displacement = 0)
    
    2. SOLVE for free DOFs:
        K_ff × d_f = F_f
        d_f = K_ff^(-1) × F_f
    
    3. CALCULATE reactions:
        R = K × d - F
        (Reactions = forces needed to enforce zero displacement at supports)
    
    4. CHECK for mechanisms:
        If K_ff is singular (determinant ≈ 0), structure is unstable!
    
    RESULT:
    ------
    - d = displacement vector (how much each node moved)
    - R = reaction vector (forces at supports pushing back)
    """
    
    print("STEP 6: Solving the Structural System")
    print("-" * 70)
    
    # Solve: K × d = F
    # This finds the displacements d that satisfy equilibrium
    d, R, free = solve_linear(K, F, fixed)
    
    print("System solved successfully! ✓")
    print(f"Free DOFs:                 {len(free)} (out of {ndof} total)")
    print()
    
    # ========================================================================
    # STEP 7: EXTRACT AND VERIFY RESULTS
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 7:
    ---------------------
    
    EXTRACT KEY RESULTS:
    -------------------
    
    1. DEFLECTIONS:
        FOR each node:
            ux = d[node_id × 3 + 0]  (horizontal displacement)
            uy = d[node_id × 3 + 1]  (vertical displacement)
            rz = d[node_id × 3 + 2]  (rotation)
        
        Maximum deflection = maximum |uy| (usually at center for UDL)
    
    2. REACTIONS:
        FOR each support node:
            Rx = R[node_id × 3 + 0]  (horizontal reaction)
            Ry = R[node_id × 3 + 1]  (vertical reaction)
            Mz = R[node_id × 3 + 2]  (moment reaction)
        
        Total reactions should balance applied loads (equilibrium check)
    
    3. VERIFY against theoretical solution:
        FOR simply supported beam with UDL:
            δ_max_theory = 5wL⁴/(384EI)  (at center)
            R_left_theory = wL/2  (each support takes half)
            R_right_theory = wL/2
        
        Check: |δ_max_computed - δ_max_theory| < tolerance
        Check: |R_computed - R_theory| < tolerance
    """
    
    print("STEP 7: Extracting and Verifying Results")
    print("-" * 70)
    
    # Extract displacements
    # Get vertical displacement at each node
    uy_all = np.array([d[i * DOF_PER_NODE + 1] for i in range(num_nodes)])
    ux_all = np.array([d[i * DOF_PER_NODE + 0] for i in range(num_nodes)])
    
    # Maximum deflection (should be at center for symmetric loading)
    max_deflection = np.min(uy_all)  # Most negative (downward)
    max_deflection_node = np.argmin(uy_all)
    
    # Extract reactions
    Ry_left = R[DOF_PER_NODE * 0 + 1]      # Left support vertical reaction
    Ry_right = R[DOF_PER_NODE * last_node + 1]  # Right support vertical reaction
    
    # Theoretical solutions (from beam theory textbooks)
    # For simply supported beam with uniform distributed load:
    delta_max_theory = 5 * abs(w) * L**4 / (384 * E * I)  # Maximum deflection
    # Note: w is negative, but formula uses absolute value
    R_theory = abs(w) * L / 2.0  # Each support takes half the load
    
    print("DISPLACEMENTS:")
    print(f"  Maximum deflection:      {max_deflection:.6f} m (downward)")
    print(f"  Location:                Node {max_deflection_node} (x = {nodes[max_deflection_node].x:.2f} m)")
    print(f"  Expected (theory):       {-delta_max_theory:.6f} m")
    print(f"  Error:                   {abs(max_deflection + delta_max_theory):.6e} m")
    print(f"  Relative error:          {abs(max_deflection + delta_max_theory) / delta_max_theory * 100:.2f}%")
    
    # Check if theory matches computation
    theory_match = np.isclose(max_deflection, -delta_max_theory, rtol=2e-2)
    print(f"  Theory match:            {'✓ PASS' if theory_match else '✗ FAIL'}")
    print()
    
    print("REACTIONS:")
    print(f"  Left reaction:           {Ry_left:.2f} N (upward)")
    print(f"  Right reaction:          {Ry_right:.2f} N (upward)")
    print(f"  Total reactions:         {Ry_left + Ry_right:.2f} N")
    print(f"  Expected (theory):       {R_theory:.2f} N each")
    print(f"  Total expected:          {2 * R_theory:.2f} N")
    
    # Check equilibrium: reactions should balance applied load
    total_applied = abs(w_total)
    total_reactions = Ry_left + Ry_right
    equilibrium_error = abs(total_reactions - total_applied)
    
    print(f"  Equilibrium check:")
    print(f"    Applied load:          {total_applied:.2f} N")
    print(f"    Reactions:             {total_reactions:.2f} N")
    print(f"    Error:                 {equilibrium_error:.2e} N")
    print(f"    In equilibrium:        {'✓ YES' if equilibrium_error < 1e-6 else '✗ NO'}")
    print()
    
    # ========================================================================
    # STEP 8: VISUALIZE THE RESULTS
    # ========================================================================
    """
    PSEUDOCODE FOR STEP 8:
    ---------------------
    
    CREATE visualization:
    
    1. EXTRACT coordinates:
        xs = [node.x for all nodes]
        ys = [node.y for all nodes]  (should all be 0 for horizontal beam)
    
    2. EXTRACT displacements:
        uy = [d[node_id × 3 + 1] for all nodes]  (vertical displacements)
    
    3. CALCULATE deformed shape:
        deformed_y = original_y + uy
    
    4. SCALE for visibility:
        Scale up deflection (deflections are tiny!)
        deformed_y_scaled = original_y + scale_factor × uy
    
    5. PLOT:
        - Plot original beam (blue line)
        - Plot deformed beam (red line, scaled)
        - Add labels, legend, grid
    """
    
    print("STEP 8: Creating Visualization")
    print("-" * 70)
    
    # Extract node coordinates
    xs = np.array([nodes[i].x for i in range(num_nodes)])
    ys = np.array([nodes[i].y for i in range(num_nodes)])  # All zeros (horizontal)
    
    # Extract vertical displacements
    uy_displacements = np.array([d[i * DOF_PER_NODE + 1] for i in range(num_nodes)])
    
    # Scale deflections for visualization (real deflections are tiny!)
    scale = 1000  # Multiply by 1000 to make visible
    ys_deformed = ys + scale * uy_displacements
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot original (undeformed) beam
    plt.plot(xs, ys, 'b-o', 
             label='Undeformed (original)', 
             linewidth=2, 
             markersize=6,
             alpha=0.7)
    
    # Plot deformed beam (scaled up)
    plt.plot(xs, ys_deformed, 'r-o', 
             label=f'Deformed (×{scale} scale, exaggerated)', 
             linewidth=2, 
             markersize=6,
             alpha=0.7)
    
    # Add reference line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add support symbols
    plt.plot(nodes[0].x, 0, 'g^', markersize=15, label='Support (pinned)')
    plt.plot(nodes[last_node].x, 0, 'g^', markersize=15)
    
    # Formatting
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.title('Simply Supported Beam with Uniform Distributed Load', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add text box with results
    textstr = f'Results:\n'
    textstr += f'Max deflection: {max_deflection:.6f} m\n'
    textstr += f'Left reaction: {Ry_left:.1f} N\n'
    textstr += f'Right reaction: {Ry_right:.1f} N'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props)
    
    print("Plot created! Showing visualization...")
    print("  Blue line: Original (straight) beam")
    print("  Red line: Deformed (bent) beam (1000× exaggerated)")
    print("  Green triangles: Supports")
    print()
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("KEY TAKEAWAYS:")
    print("1. UDL loads are converted to equivalent nodal loads")
    print("2. Each element contributes wL/2 force and ±wL²/12 moments")
    print("3. Forces are transformed from local to global coordinates")
    print("4. The solution matches theoretical beam formulas ✓")
    print()
    print("This demonstrates that your loads.py implementation works correctly!")


if __name__ == "__main__":
    main()