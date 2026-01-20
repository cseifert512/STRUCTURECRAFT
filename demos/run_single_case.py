import numpy as np
import matplotlib.pyplot as plt

from mini_branch.model import Node, Frame2D
from mini_branch.assembly import assemble_global_K, DOF_PER_NODE
from mini_branch.solve import solve_linear

def main():
    # Units: SI here (N, m, Pa). Keep consistent.
    L = 3.0
    E = 210e9
    I = 8.0e-6
    A = 0.01
    P = 1000.0  # N downward

    nodes = {
        0: Node(0, 0.0, 0.0),
        1: Node(1, L, 0.0),
    }
    elements = [Frame2D(0, 0, 1, E=E, A=A, I=I)]

    K = assemble_global_K(nodes, elements)
    ndof = DOF_PER_NODE * len(nodes)
    F = np.zeros(ndof)
    # Apply tip load at node 1, Fy = -P
    F[DOF_PER_NODE*1 + 1] = -P

    fixed = [0, 1, 2]  # node 0: ux, uy, rz fixed
    d, R, _ = solve_linear(K, F, fixed)

    uy_tip = d[DOF_PER_NODE*1 + 1]
    rz_tip = d[DOF_PER_NODE*1 + 2]
    print("Tip uy (m):", uy_tip)
    print("Tip rz (rad):", rz_tip)
    print("Reactions at fixed (Fx,Fy,M):", R[0:3])

    # Closed-form checks
    uy_expected = -P * L**3 / (3 * E * I)
    rz_expected = -P * L**2 / (2 * E * I)
    print("Expected uy (m):", uy_expected)
    print("Expected rz (rad):", rz_expected)

    # crude deflection plot (just two nodes)
    scale = 50
    xs = np.array([nodes[0].x, nodes[1].x])
    ys = np.array([nodes[0].y, nodes[1].y])
    uy = np.array([d[0*DOF_PER_NODE+1], d[1*DOF_PER_NODE+1]])

    plt.figure()
    plt.plot(xs, ys, marker="o", label="undeformed")
    plt.plot(xs, ys + scale*uy, marker="o", label=f"deformed x{scale}")
    plt.legend()
    plt.title("Cantilever deflection (scaled)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()

if __name__ == "__main__":
    main()
