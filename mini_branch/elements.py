# Frame2D element stiffness + transformation + fixed-end forces for UDL

import numpy as np
from .model import Node, Frame2D

def element_geometry(nodes: dict[int, Node], e: Frame2D):
    ni = nodes[e.ni]
    nj = nodes[e.nj]
    dx = nj.x - ni.x
    dy = nj.y - ni.y
    L = float(np.hypot(dx, dy))
    if L <= 0.0:
        raise ValueError(f"Element {e.id} has zero length.")
    c = dx / L
    s = dy / L
    return L, c, s

def frame2d_local_stiffness(E: float, A: float, I: float, L: float) -> np.ndarray:
    """
    Local stiffness matrix in element local coords (x along member).
    DOF order: [uix, uiy, rzi, ujx, ujy, rzj]
    """
    EA_L = E * A / L
    EI = E * I
    L2 = L * L
    L3 = L2 * L

    k = np.array([
        [ EA_L,      0.0,        0.0,    -EA_L,      0.0,        0.0],
        [  0.0,  12*EI/L3,   6*EI/L2,      0.0, -12*EI/L3,   6*EI/L2],
        [  0.0,   6*EI/L2,    4*EI/L,      0.0,  -6*EI/L2,    2*EI/L],
        [-EA_L,      0.0,        0.0,     EA_L,      0.0,        0.0],
        [  0.0, -12*EI/L3,  -6*EI/L2,      0.0,  12*EI/L3,  -6*EI/L2],
        [  0.0,   6*EI/L2,    2*EI/L,      0.0,  -6*EI/L2,    4*EI/L],
    ], dtype=float)
    return k

def frame2d_transform(c: float, s: float) -> np.ndarray:
    """
    6x6 transform from global DOFs to local DOFs.
    """
    T = np.array([
        [ c,  s, 0,  0, 0, 0],
        [-s,  c, 0,  0, 0, 0],
        [ 0,  0, 1,  0, 0, 0],
        [ 0,  0, 0,  c, s, 0],
        [ 0,  0, 0, -s, c, 0],
        [ 0,  0, 0,  0, 0, 1],
    ], dtype=float)
    return T

def frame2d_global_stiffness(nodes: dict[int, Node], e: Frame2D) -> np.ndarray:
    L, c, s = element_geometry(nodes, e)
    k_local = frame2d_local_stiffness(e.E, e.A, e.I, L)
    T = frame2d_transform(c, s)
    k_global = T.T @ k_local @ T
    return k_global
