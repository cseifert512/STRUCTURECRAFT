# Node, Element, Material, Section, LoadCase (dataclasses/pydantic)

from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float

@dataclass(frozen=True)
class Frame2D:
    """
    2D frame element (Euler–Bernoulli): 2 nodes, 3 DOF per node: (ux, uy, rz)
    """
    id: int
    ni: int
    nj: int
    E: float
    A: float
    I: float
