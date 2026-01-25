# mini_branch/kernel/dof.py
"""
DOF MANAGER: Dimension-Agnostic Degree of Freedom Indexing
==========================================================

PURPOSE:
--------
This module handles the mapping from (node_id, local_dof) to global DOF indices.
This is the ONE thing that changes between 2D and 3D analysis:

    2D Frame:  3 DOF/node (ux, uy, rz)
    3D Truss:  3 DOF/node (ux, uy, uz)
    3D Frame:  6 DOF/node (ux, uy, uz, rx, ry, rz)

By abstracting this into a DOFManager, our assembly and solve code
becomes dimension-agnostic.

USAGE:
------
    # For 2D frame analysis (your existing code)
    dof = DOFManager(dof_per_node=3)
    
    # For 3D truss analysis (new)
    dof = DOFManager(dof_per_node=3)  # Same! ux, uy, uz
    
    # For 3D frame analysis (future)
    dof = DOFManager(dof_per_node=6)
    
    # Get global index for node 2, local DOF 1
    global_idx = dof.idx(node_id=2, local_dof=1)  # → 7 (for dof_per_node=3)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DOFManager:
    """
    Manages degree-of-freedom indexing for structural analysis.
    
    This is the bridge between "node 5, y-displacement" and "global DOF index 16".
    The same interface works for 2D frames, 3D trusses, and 3D frames.
    
    Attributes:
    -----------
    dof_per_node : int
        Number of DOFs per node:
        - 3 for 2D frame (ux, uy, rz)
        - 3 for 3D truss (ux, uy, uz)
        - 6 for 3D frame (ux, uy, uz, rx, ry, rz)
    
    Examples:
    ---------
    >>> dof = DOFManager(dof_per_node=3)
    >>> dof.idx(0, 0)  # Node 0, DOF 0 (ux)
    0
    >>> dof.idx(0, 1)  # Node 0, DOF 1 (uy)
    1
    >>> dof.idx(1, 0)  # Node 1, DOF 0 (ux)
    3
    >>> dof.ndof(4)    # Total DOFs for 4 nodes
    12
    """
    dof_per_node: int
    
    def idx(self, node_id: int, local_dof: int) -> int:
        """
        Get the global DOF index for a node's local DOF.
        
        Parameters:
        -----------
        node_id : int
            The node identifier (0-indexed)
        local_dof : int
            The local DOF index within the node (0 to dof_per_node-1)
            - For 2D frame: 0=ux, 1=uy, 2=rz
            - For 3D truss: 0=ux, 1=uy, 2=uz
            - For 3D frame: 0=ux, 1=uy, 2=uz, 3=rx, 4=ry, 5=rz
        
        Returns:
        --------
        int
            Global DOF index in the system matrices
        """
        return self.dof_per_node * node_id + local_dof
    
    def ndof(self, n_nodes: int) -> int:
        """
        Calculate total DOFs for a system with n_nodes.
        
        Parameters:
        -----------
        n_nodes : int
            Number of nodes in the model
        
        Returns:
        --------
        int
            Total number of DOFs (size of K matrix)
        """
        return self.dof_per_node * n_nodes
    
    def node_dofs(self, node_id: int) -> List[int]:
        """
        Get all global DOF indices for a single node.
        
        Useful for applying boundary conditions or extracting
        all displacements at a node.
        
        Parameters:
        -----------
        node_id : int
            The node identifier
        
        Returns:
        --------
        List[int]
            List of global DOF indices [first_dof, ..., last_dof]
        
        Examples:
        ---------
        >>> dof = DOFManager(dof_per_node=3)
        >>> dof.node_dofs(2)
        [6, 7, 8]
        """
        base = self.dof_per_node * node_id
        return list(range(base, base + self.dof_per_node))
    
    def element_dof_map(self, node_ids: List[int]) -> List[int]:
        """
        Get the DOF map for an element connecting multiple nodes.
        
        This returns the indices needed to scatter/gather element
        matrices into/from the global matrices.
        
        Parameters:
        -----------
        node_ids : List[int]
            Node IDs that the element connects (e.g., [ni, nj] for 2-node element)
        
        Returns:
        --------
        List[int]
            Flattened list of global DOF indices for the element
        
        Examples:
        ---------
        >>> dof = DOFManager(dof_per_node=3)
        >>> dof.element_dof_map([0, 1])  # 2-node element
        [0, 1, 2, 3, 4, 5]
        >>> dof.element_dof_map([2, 5])  # Element from node 2 to node 5
        [6, 7, 8, 15, 16, 17]
        """
        result = []
        for node_id in node_ids:
            result.extend(self.node_dofs(node_id))
        return result


# Convenience: pre-configured managers for common analysis types
DOF_2D_FRAME = DOFManager(dof_per_node=3)   # ux, uy, rz
DOF_3D_TRUSS = DOFManager(dof_per_node=3)   # ux, uy, uz
DOF_3D_FRAME = DOFManager(dof_per_node=6)   # ux, uy, uz, rx, ry, rz




