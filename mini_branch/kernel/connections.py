# mini_branch/kernel/connections.py
"""Connection modeling: semi-rigid joints with rotational springs."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class ConnectionType(Enum):
    """Standard connection types with typical stiffness values."""
    RIGID = "rigid"           # Fully moment-resisting
    PINNED = "pinned"         # Zero moment transfer
    SEMI_RIGID_LOW = "semi_rigid_low"     # Clip angle, single bolt
    SEMI_RIGID_MED = "semi_rigid_med"     # End plate, multiple bolts
    SEMI_RIGID_HIGH = "semi_rigid_high"   # Extended end plate


# Typical rotational stiffness values (N·m/rad)
# Based on research and design guides
CONNECTION_STIFFNESS = {
    ConnectionType.RIGID: float('inf'),
    ConnectionType.PINNED: 0.0,
    ConnectionType.SEMI_RIGID_LOW: 1e6,    # ~1 MN·m/rad
    ConnectionType.SEMI_RIGID_MED: 5e6,    # ~5 MN·m/rad
    ConnectionType.SEMI_RIGID_HIGH: 20e6,  # ~20 MN·m/rad
}


@dataclass
class Connection:
    """Connection definition at element end."""
    node_id: int
    element_id: int
    end: str           # 'i' or 'j' (start or end of element)
    conn_type: ConnectionType
    k_rot: Optional[float] = None  # Custom stiffness (overrides conn_type)
    
    def get_stiffness(self) -> float:
        """Get rotational stiffness, using custom if provided."""
        if self.k_rot is not None:
            return self.k_rot
        return CONNECTION_STIFFNESS.get(self.conn_type, float('inf'))


def fixity_factor(k_rot: float, EI: float, L: float) -> float:
    """
    Calculate fixity factor for semi-rigid connection.
    
    Fixity factor ranges from 0 (pinned) to 1 (rigid).
    Based on the relationship between connection stiffness and member stiffness.
    
    f = 1 / (1 + 3EI / (k_rot * L))
    
    Args:
        k_rot: Rotational stiffness of connection (N·m/rad)
        EI: Flexural rigidity of member (N·m²)
        L: Member length (m)
    
    Returns:
        f: Fixity factor [0, 1]
    """
    if k_rot == 0.0:
        return 0.0
    if k_rot == float('inf') or not np.isfinite(k_rot):
        return 1.0
    
    member_stiffness = 3.0 * EI / L  # Stiffness coefficient for rotation at end
    return 1.0 / (1.0 + member_stiffness / k_rot)


def modify_frame_stiffness_for_connections(
    ke: np.ndarray,
    E: float,
    I: float,
    L: float,
    conn_i: Optional[Connection] = None,
    conn_j: Optional[Connection] = None
) -> np.ndarray:
    """
    Modify frame element stiffness matrix for semi-rigid connections.
    
    For 2D frame element with 6 DOFs: [u1, v1, θ1, u2, v2, θ2]
    Modifies rotational terms based on connection fixity.
    
    This uses the simplified approach of scaling rotational stiffness terms
    by the fixity factor. More accurate methods use condensation.
    
    Args:
        ke: Local element stiffness matrix (6x6 for 2D frame)
        E: Elastic modulus
        I: Moment of inertia
        L: Element length
        conn_i: Connection at node i (start)
        conn_j: Connection at node j (end)
    
    Returns:
        ke_mod: Modified stiffness matrix
    """
    ke_mod = ke.copy()
    EI = E * I
    
    # Rotational DOF indices (for 2D frame: DOF 2 at node i, DOF 5 at node j)
    rot_i = 2
    rot_j = 5
    
    # Apply fixity factor at node i
    if conn_i is not None:
        k_rot_i = conn_i.get_stiffness()
        f_i = fixity_factor(k_rot_i, EI, L)
        
        # Scale rotational stiffness terms
        ke_mod[rot_i, :] *= f_i
        ke_mod[:, rot_i] *= f_i
        ke_mod[rot_i, rot_i] /= f_i if f_i > 0 else 1.0  # Correct diagonal
    
    # Apply fixity factor at node j
    if conn_j is not None:
        k_rot_j = conn_j.get_stiffness()
        f_j = fixity_factor(k_rot_j, EI, L)
        
        ke_mod[rot_j, :] *= f_j
        ke_mod[:, rot_j] *= f_j
        ke_mod[rot_j, rot_j] /= f_j if f_j > 0 else 1.0
    
    return ke_mod


def add_rotational_spring(
    K: np.ndarray,
    dof_idx: int,
    k_rot: float
) -> np.ndarray:
    """
    Add rotational spring to global stiffness matrix.
    
    Alternative approach: model connection flexibility as explicit spring element.
    
    Args:
        K: Global stiffness matrix
        dof_idx: Rotational DOF index to add spring to
        k_rot: Rotational spring stiffness (N·m/rad)
    
    Returns:
        K_mod: Modified global stiffness matrix
    """
    K_mod = K.copy()
    if np.isfinite(k_rot) and k_rot > 0:
        K_mod[dof_idx, dof_idx] += k_rot
    return K_mod


def compute_connection_moments(
    d_global: np.ndarray,
    connections: List[Connection],
    dof_manager: Any
) -> Dict[int, Dict[str, float]]:
    """
    Compute moments at semi-rigid connections.
    
    M = k_rot * θ
    
    Args:
        d_global: Global displacement vector
        connections: List of connection definitions
        dof_manager: DOFManager instance
    
    Returns:
        Dictionary mapping node_id to connection moment data
    """
    results = {}
    
    for conn in connections:
        node_id = conn.node_id
        k_rot = conn.get_stiffness()
        
        # Get rotational DOF index
        # Assumes rotation is the 3rd DOF (index 2) per node for 2D frames
        dof_per_node = dof_manager.dof_per_node
        if dof_per_node >= 3:
            rot_dof = dof_manager.idx(node_id, 2)  # θz for 2D frame
        else:
            continue  # Truss elements don't have rotational DOFs
        
        # Get rotation at node
        theta = d_global[rot_dof]
        
        # Compute moment (if not pinned or rigid)
        if np.isfinite(k_rot) and k_rot > 0:
            moment = k_rot * theta
        else:
            moment = 0.0
        
        if node_id not in results:
            results[node_id] = {}
        
        results[node_id][conn.element_id] = {
            'rotation_rad': theta,
            'moment_Nm': moment,
            'stiffness': k_rot,
            'conn_type': conn.conn_type.value,
        }
    
    return results


# =============================================================================
# For 3D Truss Elements (simplified connection model)
# =============================================================================

def release_truss_end(
    ke: np.ndarray,
    end: str,
    release_axial: bool = False
) -> np.ndarray:
    """
    Apply end release to 3D truss element.
    
    Truss elements only have axial DOFs, so "releases" mean
    the member doesn't connect to that node (pin in all directions).
    
    This is useful for modeling mechanisms or partial attachments.
    
    Args:
        ke: Local element stiffness (6x6 for 3D truss)
        end: 'i' or 'j'
        release_axial: If True, release axial connection (makes element inactive)
    
    Returns:
        ke_mod: Modified stiffness matrix
    """
    ke_mod = ke.copy()
    
    if release_axial:
        if end == 'i':
            # Zero out node i contributions
            ke_mod[:3, :] = 0.0
            ke_mod[:, :3] = 0.0
        else:
            # Zero out node j contributions
            ke_mod[3:, :] = 0.0
            ke_mod[:, 3:] = 0.0
    
    return ke_mod


def create_connection_matrix(
    nodes: dict,
    bars: list,
    connections: Dict[Tuple[int, int], Connection]
) -> Dict[int, List[Connection]]:
    """
    Build connection lookup matrix.
    
    Args:
        nodes: Node dictionary
        bars: List of bars
        connections: Dict mapping (node_id, element_id) to Connection
    
    Returns:
        Dict mapping element_id to list of connections at that element
    """
    conn_matrix = {}
    
    for bar in bars:
        conn_matrix[bar.id] = []
        
        # Check for connection at node i
        key_i = (bar.ni, bar.id)
        if key_i in connections:
            conn_matrix[bar.id].append(connections[key_i])
        
        # Check for connection at node j
        key_j = (bar.nj, bar.id)
        if key_j in connections:
            conn_matrix[bar.id].append(connections[key_j])
    
    return conn_matrix


def default_rigid_connections(bars: list) -> Dict[Tuple[int, int], Connection]:
    """Create default fully-rigid connections for all bar ends."""
    connections = {}
    
    for bar in bars:
        connections[(bar.ni, bar.id)] = Connection(
            node_id=bar.ni,
            element_id=bar.id,
            end='i',
            conn_type=ConnectionType.RIGID
        )
        connections[(bar.nj, bar.id)] = Connection(
            node_id=bar.nj,
            element_id=bar.id,
            end='j',
            conn_type=ConnectionType.RIGID
        )
    
    return connections


def default_pinned_connections(bars: list) -> Dict[Tuple[int, int], Connection]:
    """Create default pinned connections for all bar ends (for truss structures)."""
    connections = {}
    
    for bar in bars:
        connections[(bar.ni, bar.id)] = Connection(
            node_id=bar.ni,
            element_id=bar.id,
            end='i',
            conn_type=ConnectionType.PINNED
        )
        connections[(bar.nj, bar.id)] = Connection(
            node_id=bar.nj,
            element_id=bar.id,
            end='j',
            conn_type=ConnectionType.PINNED
        )
    
    return connections




