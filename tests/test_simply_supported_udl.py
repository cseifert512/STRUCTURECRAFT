# File: tests/test_simply_supported_udl.py
"""
TEST: SIMPLY SUPPORTED BEAM WITH UNIFORM DISTRIBUTED LOAD (UDL)
================================================================

PURPOSE:
--------
This test validates our UDL load implementation by comparing the finite element
solution against the exact theoretical solution from beam theory.

WHY MESHING?
-----------
A single Euler-Bernoulli beam element can only represent a cubic displacement
field, but the exact solution for UDL is a quartic (4th order) polynomial.
By meshing the beam into multiple elements (e.g., 10 elements), we can
approximate the quartic solution much more accurately.

THEORETICAL SOLUTION:
--------------------
For a simply supported beam with uniform distributed load w over length L:
- Maximum deflection (at center): δ_max = 5wL⁴/(384EI)
- Each support reaction: R = wL/2

This test verifies our finite element solution matches this formula.
"""

import numpy as np

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear


def test_simply_supported_udl_deflection():
    """
    PSEUDOCODE OVERVIEW:
    ====================
    
    STEP 1: Define the physical problem (beam properties, UDL value)
    STEP 2: Create a meshed beam model (10 elements, 11 nodes)
    STEP 3: Build the stiffness matrix K
    STEP 4: Assemble UDL loads into force vector F (NEW!)
    STEP 5: Define boundary conditions (simply supported: pinned both ends)
    STEP 6: Solve the system: K × d = F
    STEP 7: Extract midspan deflection
    STEP 8: Compare to theoretical formula and verify within tolerance
    """
    
    # ========================================================================
    # STEP 1: DEFINE THE PHYSICAL PROBLEM
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    DEFINE beam properties:
        L = total beam length (meters)
        E = Young's modulus (material stiffness, Pascals)
        I = Moment of inertia (bending resistance, m⁴)
        A = Cross-sectional area (m²)
        w0 = uniform distributed load per unit length (N/m, negative = downward)
    
    NOTE: We use w0 (absolute value) for clarity in formulas, but w = -w0
          for downward loading.
    """
    
    # Physical parameters
    L = 4.0          # Total beam length (meters)
    E = 210e9        # Young's modulus (Pascals) - Steel
    I = 8.0e-6       # Moment of inertia (m⁴)
    A = 0.01         # Cross-sectional area (m²)
    w0 = 1000.0      # Uniform distributed load magnitude (N/m)
    w = -w0          # Downward load (negative = downward in our convention)
    
    # ========================================================================
    # STEP 2: CREATE MESHED BEAM MODEL
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    WHY 10 ELEMENTS?
    ----------------
    Single element: cubic displacement field (inaccurate for UDL)
    Multiple elements: can approximate quartic solution (accurate for UDL)
    Rule of thumb: 10 elements gives ~2% accuracy for UDL
    
    ALGORITHM:
    ---------
    
    1. DETERMINE mesh parameters:
        num_elements = 10
        num_nodes = num_elements + 1  # 11 nodes
    
    2. CREATE nodes evenly spaced:
        FOR i from 0 to num_nodes-1:
            x_position = L × (i / num_elements)
            y_position = 0.0  (beam is horizontal)
            nodes[i] = Node(id=i, x=x_position, y=y_position)
    
    3. CREATE elements connecting adjacent nodes:
        FOR i from 0 to num_elements-1:
            start_node = i
            end_node = i + 1
            elements.append(Frame2D(
                id=i,
                ni=start_node,
                nj=end_node,
                E=E, A=A, I=I
            ))
    
    VISUALIZATION:
    -------------
    Nodes:   0    1    2    3    4    5    6    7    8    9    10
    Elements: [0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
    
    Midspan node = node 5 (at x = L/2 = 2.0 m)
    """
    
    # Create meshed beam
    num_elements = 10  # Number of beam segments
    num_nodes = num_elements + 1  # 11 nodes (one more than elements)
    
    # Create nodes evenly spaced along the beam
    nodes = {}
    for i in range(num_nodes):
        x = L * i / num_elements  # Evenly distributed from 0 to L
        nodes[i] = Node(i, x, 0.0)  # Beam is horizontal (y = 0)
    
    # Create elements connecting adjacent nodes
    elements = []
    for i in range(num_elements):
        elements.append(Frame2D(
            id=i,
            ni=i,      # Start node
            nj=i+1,    # End node
            E=E,
            A=A,
            I=I
        ))
    
    # Identify midspan node (closest to L/2)
    mid_node = num_nodes // 2  # Node 5 (at x = 2.0 m)
    
    # ========================================================================
    # STEP 3: BUILD THE STIFFNESS MATRIX
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    ALGORITHM:
    ---------
    K = assemble_global_K(nodes, elements)
    
    This assembles all element stiffness matrices into a global stiffness
    matrix that describes how the entire structure responds to forces.
    """
    
    # Build global stiffness matrix
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)  # Total degrees of freedom
    
    # ========================================================================
    # STEP 4: ASSEMBLE UDL LOADS INTO FORCE VECTOR
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    THIS IS THE KEY STEP! We're using our new loads.py functionality.
    
    ALGORITHM:
    ---------
    
    1. CREATE empty force vector:
        F = zeros(ndof)
    
    2. DEFINE UDL for each element:
        element_udls = {element_id: w_value for each element}
        OR: element_udls = {i: w for i in range(num_elements)}
    
    3. ASSEMBLE UDL loads:
        F_udl = assemble_element_loads_global(nodes, elements, element_udls)
        F += F_udl
    
    WHAT HAPPENS INSIDE assemble_element_loads_global:
    --------------------------------------------------
    FOR each element:
        - Calculate local equivalent nodal loads: [0, wL/2, +wL²/12, 0, wL/2, -wL²/12]
        - Transform to global coordinates
        - Add to global force vector F
    
    RESULT: Force vector F with all UDL loads properly distributed across nodes
    """
    
    # Initialize force vector to zeros
    F = np.zeros(ndof, dtype=float)
    
    # Define UDL for each element (all elements have same UDL w)
    element_udls = {i: w for i in range(num_elements)}
    
    # Assemble UDL loads into global force vector
    # This uses the function we implemented in loads.py!
    F_udl = assemble_element_loads_global(nodes, elements, element_udls)
    F += F_udl
    
    # ========================================================================
    # STEP 5: DEFINE BOUNDARY CONDITIONS (SUPPORTS)
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    SIMPLY SUPPORTED (PINNED AT BOTH ENDS):
    ---------------------------------------
    - Left support: fix ux and uy (prevents horizontal and vertical movement)
    - Right support: fix uy only (prevents vertical movement, allows horizontal expansion)
    - Rotations: free at both ends (supports are "pinned" and can rotate)
    
    ALGORITHM:
    ---------
    
    fixed_DOFs = [
        node_0 × DOF_PER_NODE + 0,  # Node 0, ux (horizontal)
        node_0 × DOF_PER_NODE + 1,  # Node 0, uy (vertical)
        node_last × DOF_PER_NODE + 1,  # Last node, uy (vertical)
    ]
    
    This prevents rigid body motion while keeping supports "pinned" (rotatable).
    """
    
    # Simply supported: pinned at both ends
    last_node = num_nodes - 1
    
    fixed = [
        DOF_PER_NODE * 0 + 0,              # Node 0, ux (horizontal)
        DOF_PER_NODE * 0 + 1,              # Node 0, uy (vertical)
        DOF_PER_NODE * last_node + 1,      # Last node, uy (vertical)
    ]
    
    # ========================================================================
    # STEP 6: SOLVE THE STRUCTURAL SYSTEM
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    THE FUNDAMENTAL EQUATION:
    -------------------------
    K × d = F
    
    WHERE:
    - K = stiffness matrix (known: we built it)
    - d = displacement vector (unknown: we want to find this)
    - F = force vector (known: we assembled it from UDL loads)
    
    ALGORITHM:
    ---------
    d, R, free = solve_linear(K, F, fixed)
    
    RESULT:
    - d = displacement vector (how much each node moved)
    - R = reaction vector (forces at supports)
    - free = list of free DOF indices
    """
    
    # Solve: K × d = F
    d, R, _ = solve_linear(K, F, fixed)
    
    # ========================================================================
    # STEP 7: EXTRACT MIDSPAN DEFLECTION
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    EXTRACT vertical displacement at midspan node:
    
    midspan_deflection = d[mid_node × DOF_PER_NODE + 1]
    
    WHERE:
    - mid_node × DOF_PER_NODE + 1 = vertical DOF (uy) at midspan node
    - Negative value means downward deflection
    """
    
    # Extract midspan vertical deflection
    uy_midspan = d[DOF_PER_NODE * mid_node + 1]  # Vertical displacement at midspan
    
    # ========================================================================
    # STEP 8: COMPARE TO THEORETICAL FORMULA AND VERIFY
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    THEORETICAL FORMULA (from beam theory):
    ---------------------------------------
    For simply supported beam with uniform distributed load w over length L:
    
    δ_max = 5wL⁴/(384EI)
    
    WHERE:
    - w = distributed load per unit length (N/m)
    - L = beam length (m)
    - E = Young's modulus (Pa)
    - I = Moment of inertia (m⁴)
    
    NOTE: Since w is negative (downward), the formula gives negative deflection.
    
    VERIFICATION:
    ------------
    Check: |computed_deflection - theoretical_deflection| / |theoretical_deflection| < tolerance
    
    TOLERANCE:
    ---------
    - For 10 elements: rtol = 2e-2 (2% relative error is acceptable)
    - For more elements (e.g., 20): can use tighter tolerance
    
    WHY 2%?
    ------
    Single element would give much larger error (~10-20%).
    With 10 elements, we should achieve ~1-2% accuracy.
    """
    
    # Theoretical maximum deflection (from beam theory)
    # Formula: δ_max = 5wL⁴/(384EI)
    # Note: w is negative (downward), so deflection is negative
    delta_max_theory = 5 * w * L**4 / (384 * E * I)
    
    # Also check reactions (optional but good validation)
    Ry_left = R[DOF_PER_NODE * 0 + 1]      # Left support reaction
    Ry_right = R[DOF_PER_NODE * last_node + 1]  # Right support reaction
    R_theory = abs(w) * L / 2.0  # Each support takes half the total load
    
    # ========================================================================
    # STEP 9: ASSERTIONS (VERIFY CORRECTNESS)
    # ========================================================================
    """
    PSEUDOCODE:
    ----------
    
    CHECK 1: Midspan deflection matches theory
        ASSERT: |uy_midspan - delta_max_theory| / |delta_max_theory| < rtol
    
    CHECK 2: Reactions are equal and correct (optional but recommended)
        ASSERT: Ry_left ≈ R_theory
        ASSERT: Ry_right ≈ R_theory
    
    CHECK 3: Equilibrium (reactions balance applied load)
        total_applied = abs(w) × L
        total_reactions = Ry_left + Ry_right
        ASSERT: |total_reactions - total_applied| < small_tolerance
    """
    
    # Verify midspan deflection matches theory (within 2% for 10 elements)
    assert np.isclose(uy_midspan, delta_max_theory, rtol=2e-2, atol=1e-9), \
        f"Midspan deflection {uy_midspan:.6f} m != expected {delta_max_theory:.6f} m " \
        f"(relative error: {abs(uy_midspan - delta_max_theory) / abs(delta_max_theory) * 100:.2f}%)"
    
    # Verify reactions (each support takes half the load)
    assert np.isclose(Ry_left, R_theory, rtol=1e-3, atol=1e-6), \
        f"Left reaction {Ry_left:.2f} N != expected {R_theory:.2f} N"
    assert np.isclose(Ry_right, R_theory, rtol=1e-3, atol=1e-6), \
        f"Right reaction {Ry_right:.2f} N != expected {R_theory:.2f} N"
    
    # Verify equilibrium: reactions should balance applied load
    total_applied = abs(w) * L  # Total applied load
    total_reactions = Ry_left + Ry_right
    equilibrium_error = abs(total_reactions - total_applied)
    assert equilibrium_error < 1e-6, \
        f"Equilibrium check failed: reactions {total_reactions:.2f} N != applied {total_applied:.2f} N"
    
    # Print success message with results
    print(f"✓ UDL Beam Test PASSED")
    print(f"  Midspan deflection: {uy_midspan:.6f} m (theory: {delta_max_theory:.6f} m)")
    print(f"  Relative error: {abs(uy_midspan - delta_max_theory) / abs(delta_max_theory) * 100:.2f}%")
    print(f"  Left reaction: {Ry_left:.2f} N (expected: {R_theory:.2f} N)")
    print(f"  Right reaction: {Ry_right:.2f} N (expected: {R_theory:.2f} N)")
    print(f"  Equilibrium check: ✓ (error: {equilibrium_error:.2e} N)")