import numpy as np

from mini_branch.model import Node, Frame2D
from mini_branch.assemble import assemble_global_K, DOF_PER_NODE
from mini_branch.solve import solve_linear

def test_cantilever_tip_load_deflection():
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

    uy_tip = d[DOF_PER_NODE*1 + 1]
    rz_tip = d[DOF_PER_NODE*1 + 2]

    uy_expected = -P * L**3 / (3 * E * I)
    rz_expected = -P * L**2 / (2 * E * I)

    assert np.isclose(uy_tip, uy_expected, rtol=1e-3, atol=1e-9)
    assert np.isclose(rz_tip, rz_expected, rtol=1e-3, atol=1e-9)

    # Reaction sanity: fixed-end Fy should be +P (within tolerance)
    Fy_fixed = R[1]
    assert np.isclose(Fy_fixed, +P, rtol=1e-6, atol=1e-6)
