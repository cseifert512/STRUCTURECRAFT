#global K assembly

import numpy as np
from .model import Node, Frame2D
from .elements import frame2d_global_stiffness

DOF_PER_NODE = 3  # ux, uy, rz

def dof_index(node_id: int, local_dof: int) -> int:
    return DOF_PER_NODE * node_id + local_dof

def assemble_global_K(nodes: dict[int, Node], elements: list[Frame2D]) -> np.ndarray:
    nnode = len(nodes)
    ndof = DOF_PER_NODE * nnode
    K = np.zeros((ndof, ndof), dtype=float)

    for e in elements:
        ke = frame2d_global_stiffness(nodes, e)
        # element DOF map
        map_ = [
            dof_index(e.ni, 0), dof_index(e.ni, 1), dof_index(e.ni, 2),
            dof_index(e.nj, 0), dof_index(e.nj, 1), dof_index(e.nj, 2),
        ]
        for a in range(6):
            ia = map_[a]
            for b in range(6):
                ib = map_[b]
                K[ia, ib] += ke[a, b]
    return K
