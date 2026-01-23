# mini_branch/v3d/model.py
"""
3D MODEL DEFINITIONS: Node3D and Truss3D
========================================

PURPOSE:
--------
This module defines the basic data structures for 3D structural analysis:
- Node3D: A point in 3D space with x, y, z coordinates
- Truss3D: An axial-only bar connecting two nodes

ENGINEERING CONTEXT:
--------------------
A 3D TRUSS is a structural system where:
- Members carry only AXIAL forces (tension or compression)
- Connections are idealized as frictionless pins (no moment transfer)
- Each node has 3 DOFs: ux, uy, uz (translations only)

This is simpler than a 3D FRAME (which includes bending/torsion) but is
perfect for spaceframes, canopies, and lattice structures where members
are primarily loaded axially.

WHY TRUSS FIRST?
----------------
1. Simpler math: 6×6 element stiffness vs 12×12 for frames
2. Faster validation: easier equilibrium checks
3. Perfect for canopy/spaceframe use case
4. Still impressive: "3D structural analysis" is not undergrad-level
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node3D:
    """
    A node (joint) in 3D space.
    
    Each node represents a connection point where structural members meet.
    In a truss analysis, each node has 3 degrees of freedom:
    - ux: displacement in x direction
    - uy: displacement in y direction  
    - uz: displacement in z direction
    
    Parameters:
    -----------
    id : int
        Unique identifier for this node (used for DOF mapping)
        
    x : float
        X-coordinate in global coordinate system (meters)
        
    y : float
        Y-coordinate in global coordinate system (meters)
        
    z : float
        Z-coordinate in global coordinate system (meters)
    
    Examples:
    ---------
    >>> n0 = Node3D(0, 0.0, 0.0, 0.0)  # Origin
    >>> n1 = Node3D(1, 1.0, 0.0, 0.0)  # 1m along x-axis
    >>> n2 = Node3D(2, 0.0, 0.0, 1.0)  # 1m along z-axis (up)
    
    Notes:
    ------
    - frozen=True makes nodes immutable (can't accidentally modify coordinates)
    - The id is used by DOFManager to compute global DOF indices
    - Coordinate system: typically x=east, y=north, z=up (right-handed)
    """
    id: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Truss3D:
    """
    A 3D truss element (axial-only bar) connecting two nodes.
    
    A truss element:
    - Carries only axial force (tension/compression)
    - Has no bending or torsional stiffness
    - Connects two nodes (ni, nj)
    - Has uniform cross-section properties
    
    The element stiffness matrix is 6×6 (3 DOFs at each of 2 nodes):
        [ux_i, uy_i, uz_i, ux_j, uy_j, uz_j]
    
    Parameters:
    -----------
    id : int
        Unique identifier for this element
        
    ni : int
        ID of start node (node i)
        
    nj : int
        ID of end node (node j)
        
    E : float
        Young's modulus (Pa = N/m²)
        - Steel: ~210 GPa = 210e9 Pa
        - Aluminum: ~70 GPa = 70e9 Pa
        - Timber: ~10-15 GPa
        
    A : float
        Cross-sectional area (m²)
        - Determines axial stiffness: k = EA/L
        - Also used for volume/mass calculations
    
    Examples:
    ---------
    >>> bar = Truss3D(id=0, ni=0, nj=1, E=210e9, A=0.001)  # Steel, 10cm² area
    >>> bar = Truss3D(id=1, ni=1, nj=2, E=70e9, A=0.0005)  # Aluminum, 5cm² area
    
    Notes:
    ------
    - The element direction (ni → nj) affects sign conventions but not stiffness
    - For trusses, we don't need moment of inertia (I) — that's for frames
    - Self-weight can be computed as: density × A × L
    """
    id: int
    ni: int  # Start node ID
    nj: int  # End node ID
    E: float  # Young's modulus (Pa)
    A: float  # Cross-sectional area (m²)

