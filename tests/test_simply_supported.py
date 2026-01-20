import numpy as np

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.solve import solve_linear

def test_simply_supported_midspan_pointload():
    """
    We're testing a beam that's supported at both ends (like a bridge deck).
    A weight is placed in the middle. We want to know:
    1. How much force each support pushes back with (reactions)
    2. How much the beam bends downward at the middle (deflection)
    
    WHY THIS MATTERS:
    - In real life: bridges, floor beams, shelves
    - We need to verify our code gives the right answer
    - We compare against textbook formulas (closed-form solutions)
    """
    
    # ========================================================================
    # STEP 1: DEFINE THE PHYSICAL PROBLEM
    # ========================================================================
    # Imagine a steel beam, 4 meters long, with a 1000 Newton weight in the middle
    
    L = 4.0  # Length of beam in meters
    E = 210e9  # Young's Modulus (Pa) - How stiff the material is
               # Steel is very stiff, so E is huge (210 billion Pascals)
    I = 8.0e-6  # Moment of Inertia (m⁴) - How resistant to bending
                # Think: thicker beams have bigger I, bend less
    A = 0.01  # Cross-sectional area (m²) - How much material
    P = 1000.0  # Applied load (N) - The weight pushing down
    
    # ========================================================================
    # STEP 2: CREATE THE COMPUTER MODEL
    # ========================================================================
    # We break the beam into "nodes" (points) and "elements" (segments between points)
    # Think of it like drawing dots on a ruler and connecting them
    
    # Nodes = specific points we care about
    # Node 0: Left support (where beam sits on left wall)
    # Node 1: Middle of beam (where the weight is)
    # Node 2: Right support (where beam sits on right wall)
    nodes = {
        0: Node(0, 0.0, 0.0),    # (id, x-coordinate, y-coordinate)
        1: Node(1, L/2, 0.0),    # Middle is at half the length
        2: Node(2, L, 0.0),      # Right end is at full length
    }
    
    # Elements = the beam segments connecting nodes
    # Element 0: connects node 0 to node 1 (left half of beam)
    # Element 1: connects node 1 to node 2 (right half of beam)
    # Why two elements? More elements = more accurate, but slower
    elements = [
        Frame2D(0, 0, 1, E=E, A=A, I=I),  # (element_id, start_node, end_node, properties)
        Frame2D(1, 1, 2, E=E, A=A, I=I),
    ]
    
    # ========================================================================
    # STEP 3: BUILD THE STIFFNESS MATRIX
    # ========================================================================
    # This is the "magic matrix" that tells us: "If I push here, how much does it move?"
    # Think: stiffer beam = bigger numbers in matrix = less movement for same force
    K = assemble_global_K(nodes, elements)
    
    # ========================================================================
    # STEP 4: SET UP THE FORCES
    # ========================================================================
    # We need to tell the computer: "Apply a 1000N force downward at the middle"
    # DOF = Degrees of Freedom = ways the node can move
    # Each node has 3 DOFs: horizontal (ux), vertical (uy), rotation (rz)
    
    ndof = DOF_PER_NODE * len(nodes)  # Total DOFs = 3 per node × 3 nodes = 9
    F = np.zeros(ndof)  # Start with zero forces everywhere
    
    # Apply the load at node 1 (middle), in the vertical direction (DOF index 1)
    # DOF_PER_NODE*1 + 1 means: "node 1, vertical direction"
    # -P means downward (negative = down in our coordinate system)
    F[DOF_PER_NODE*1 + 1] = -P
    
    # ========================================================================
    # STEP 5: DEFINE BOUNDARY CONDITIONS (SUPPORTS)
    # ========================================================================
    # "Simply supported" means:
    # - Beam can't move up/down at supports (pinned)
    # - Beam CAN rotate at supports (unlike fixed end)
    # - BUT we need to prevent rigid body motion!
    
    # To prevent the beam from sliding/rotating as a whole:
    # - Left support: fix BOTH horizontal (ux) AND vertical (uy) movement
    # - Right support: fix only vertical (uy) movement
    # This prevents rigid body motion while keeping supports "pinned" (rotatable)

    fixed = [
        DOF_PER_NODE*0 + 0,  # Node 0, horizontal (ux) = 0 (can't slide left/right)
        DOF_PER_NODE*0 + 1,  # Node 0, vertical (uy) = 0 (can't move up/down)
        DOF_PER_NODE*2 + 1,  # Node 2, vertical (uy) = 0 (can't move up/down)
        # Note: We DON'T fix rotations (rz) - supports are "pinned" and can rotate
        # Note: We DON'T fix horizontal at right support - it can expand/contract
    ]

    # ========================================================================
    # STEP 6: SOLVE THE SYSTEM
    # ========================================================================
    # This solves the equation: K × d = F
    # Where:
    #   K = stiffness matrix (how stiff)
    #   d = displacements (how much it moves) ← THIS IS WHAT WE WANT
    #   F = forces (what we're pushing with)
    # 
    # The computer does: d = K^(-1) × F  (matrix inversion)
    d, R, _ = solve_linear(K, F, fixed)
    # d = displacements (movements at each node)
    # R = reactions (forces at supports pushing back)
    
    # ========================================================================
    # STEP 7: EXTRACT THE RESULTS WE CARE ABOUT
    # ========================================================================
    # Get the reaction force at left support (vertical direction)
    Ry_left = R[DOF_PER_NODE*0 + 1]   # Reaction at node 0, vertical
    
    # Get the reaction force at right support (vertical direction)
    Ry_right = R[DOF_PER_NODE*2 + 1]  # Reaction at node 2, vertical
    
    # Get how much the middle of the beam moved down
    uy_midspan = d[DOF_PER_NODE*1 + 1]  # Displacement at node 1, vertical
    
    # ========================================================================
    # STEP 8: COMPARE TO TEXTBOOK ANSWER
    # ========================================================================
    # These are formulas from structural engineering textbooks
    # If our code matches these, we know it's working correctly!
    
    # For simply supported beam with center load:
    # - Each support takes half the load (symmetry)
    R_expected = P / 2.0  # 1000N / 2 = 500N at each support
    
    # - Maximum deflection (bending) at center:
    #   Formula: δ = PL³/(48EI)
    #   P = load, L = length, E = stiffness, I = resistance to bending
    #   Negative means downward
    delta_max_expected = -P * L**3 / (48 * E * I)
    
    # ========================================================================
    # STEP 9: VERIFY OUR ANSWER IS CORRECT
    # ========================================================================
    # Check if our computed values match the expected values (within small tolerance)
    assert np.isclose(Ry_left, R_expected, rtol=1e-3, atol=1e-6), \
        f"Left reaction {Ry_left} != expected {R_expected}"
    assert np.isclose(Ry_right, R_expected, rtol=1e-3, atol=1e-6), \
        f"Right reaction {Ry_right} != expected {R_expected}"
    assert np.isclose(uy_midspan, delta_max_expected, rtol=1e-3, atol=1e-9), \
        f"Midspan deflection {uy_midspan} != expected {delta_max_expected}"
    
    # Print success message
    print(f"✓ Left reaction: {Ry_left:.2f} N (expected: {R_expected:.2f} N)")
    print(f"✓ Right reaction: {Ry_right:.2f} N (expected: {R_expected:.2f} N)")
    print(f"✓ Midspan deflection: {uy_midspan:.6f} m (expected: {delta_max_expected:.6f} m)")