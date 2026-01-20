import numpy as np

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.solve import solve_linear

def test_stiffness_matrix_symmetry():
    """
    WHAT IS THIS TEST?
    ==================
    We check that the stiffness matrix is symmetric.
    
    WHY DOES THIS MATTER?
    ====================
    In physics: "If I push at point A and it moves at point B,
    then pushing at point B should move point A the same amount."
    This is called "reciprocity" or "Maxwell's reciprocal theorem."
    
    Mathematically: K[i,j] = K[j,i] for all i, j
    This means K = K^T (matrix equals its transpose)
    
    If this fails, our code has a bug!
    """
    # Create a simple beam
    L = 3.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, L, 0.0),
    }
    elements = [Frame2D(0, 0, 1, E=E, A=A, I=I)]
    
    # Build the stiffness matrix
    K = assemble_global_K(nodes, elements)
    
    # Check: K should equal K^T (transpose)
    # If they're equal, the matrix is symmetric
    np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-12,
                               err_msg="Stiffness matrix is not symmetric!")
    print("✓ Stiffness matrix is symmetric (physics is preserved)")

def test_equilibrium_vertical_forces():
    """
    WHAT IS THIS TEST?
    ==================
    We check that forces balance: Σ(upward forces) = Σ(downward forces)
    
    WHY DOES THIS MATTER?
    ====================
    Newton's First Law: "An object at rest stays at rest"
    For the beam to not accelerate up or down:
    - Sum of all upward forces = Sum of all downward forces
    - This is called "vertical equilibrium"
    
    In our case:
    - Applied load: P downward (negative)
    - Reactions: R_upward (positive)
    - Equilibrium: R_upward + P_downward = 0
    """
    # Create a cantilever beam with a tip load
    L = 3.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    P = 1000.0  # Load pushing down
    
    nodes = {
        0: Node(0, 0.0, 0.0),  # Fixed end
        1: Node(1, L, 0.0),    # Free end (where load is)
    }
    elements = [Frame2D(0, 0, 1, E=E, A=A, I=I)]
    
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    F[DOF_PER_NODE*1 + 1] = -P  # Apply load downward
    
    fixed = [0, 1, 2]  # Fix the left end
    d, R, _ = solve_linear(K, F, fixed)
    
    # ========================================================================
    # CALCULATE FORCE BALANCE
    # ========================================================================
    # Sum all vertical applied forces (DOF index 1, 4, 7, ... = 1 mod 3)
    # In our case: just the -P at node 1
    sum_applied_vertical = sum(F[i] for i in range(ndof) if i % DOF_PER_NODE == 1)
    # This equals -P (negative = downward)
    
    # Sum all vertical reactions (forces at supports)
    sum_reactions_vertical = sum(R[i] for i in range(ndof) if i % DOF_PER_NODE == 1)
    # This should equal +P (positive = upward, pushing back)
    
    # Equilibrium check: ΣR_y + ΣF_applied = 0
    # If beam is in equilibrium, this should be zero
    equilibrium_error = sum_reactions_vertical + sum_applied_vertical
    
    # Verify it's zero (within numerical tolerance)
    assert np.isclose(equilibrium_error, 0.0, rtol=1e-6, atol=1e-9), \
        f"Vertical force equilibrium violated: ΣR_y + ΣF = {equilibrium_error:.2e}"
    
    print(f"✓ Vertical equilibrium: ΣR_y = {-sum_applied_vertical:.2f} N, "
          f"ΣF_applied = {sum_applied_vertical:.2f} N")
    print(f"  (They balance: {equilibrium_error:.2e} ≈ 0)")

def test_equilibrium_moments():
    """
    WHAT IS THIS TEST?
    ==================
    We check that moments (rotational forces) balance.
    
    WHY DOES THIS MATTER?
    ====================
    For the beam to not rotate, the sum of all moments must be zero.
    Think: if you push on one side of a door, it rotates.
    To keep it still, you need equal push on the other side.
    
    Moment = Force × Distance
    We check: Σ(all moments) = 0
    """
    # Same setup as before
    L = 3.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    P = 1000.0
    
    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, L, 0.0),
    }
    elements = [Frame2D(0, 0, 1, E=E, A=A, I=I)]
    
    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    F[DOF_PER_NODE*1 + 1] = -P
    
    fixed = [0, 1, 2]
    d, R, _ = solve_linear(K, F, fixed)
    
    # ========================================================================
    # CALCULATE MOMENT BALANCE
    # ========================================================================
    # Sum all direct moments (DOF index 2, 5, 8, ... = 2 mod 3)
    # These are rotational forces applied directly
    sum_applied_moments = sum(F[i] for i in range(ndof) if i % DOF_PER_NODE == 2)
    sum_reaction_moments = sum(R[i] for i in range(ndof) if i % DOF_PER_NODE == 2)
    
    # Also calculate moments from forces (Force × Distance)
    # Moment from a force = force × perpendicular distance
    # In 2D: M_z = x*F_y - y*F_x (but y=0 for horizontal beam, so M = x*F_y)
    moment_from_forces = 0.0
    for node_id, node in nodes.items():
        dof_y = DOF_PER_NODE * node_id + 1  # Vertical force DOF
        # Moment = distance × force
        moment_from_forces += node.x * (F[dof_y] + R[dof_y])
    
    # Total moment should be zero
    total_moment = sum_reaction_moments + sum_applied_moments + moment_from_forces
    
    assert np.isclose(total_moment, 0.0, rtol=1e-3, atol=1e-6), \
        f"Moment equilibrium violated: ΣM = {total_moment:.2e}"
    
    print(f"✓ Moment equilibrium satisfied (ΣM ≈ 0)")
    print(f"  (Beam doesn't rotate: all moments balance)")