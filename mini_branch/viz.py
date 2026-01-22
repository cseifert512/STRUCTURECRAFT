"""
VISUALIZATION: PLOTTING DESIGN RESULTS
=======================================

PURPOSE:
--------
This module creates visualizations of design exploration results. Visualizations
are critical for understanding trade-offs, identifying optimal designs, and
communicating results to stakeholders (engineers, clients, managers).

WHY THIS MATTERS:
-----------------
1. **Human Understanding**: Numbers in a CSV are hard to interpret. A plot
   instantly shows patterns, trends, and outliers.

2. **Pareto Frontier Visualization**: The Pareto plot shows the "efficient
   frontier" - the boundary of optimal designs. Engineers can see the trade-off
   curve and choose based on priorities.

3. **Deformed Shape Visualization**: Shows how the structure actually moves
   under load. This helps engineers understand structural behavior and verify
   that results make physical sense.

4. **Communication**: Plots are essential for reports, presentations, and
   design reviews. A good plot tells a story.

ENGINEERING CONTEXT:
--------------------
- **Pareto Plot**: Shows volume vs drift trade-off. Lower-left corner = best
  (low volume AND low drift). The frontier shows the best possible trade-offs.

- **Deformed Shape**: Shows structural response. Engineers can verify:
  - Does it bend in the expected direction?
  - Are deflections reasonable?
  - Does the structure look stable?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from .model import Node, Frame2D
from .elements import element_geometry
from .assembly import DOF_PER_NODE, dof_index


def plot_pareto(
    df,
    mask,
    outpath: str,
    x: str = "volume",
    y: str = "drift",
    title: str = "Pareto Frontier: Volume vs Drift",
) -> None:
    """
    Create a scatter plot showing the Pareto frontier of designs.
    
    WHY THIS FUNCTION?
    ------------------
    After running a design search, we have hundreds of designs. We need to
    visualize them to understand the trade-offs. This function creates a
    scatter plot where:
    - Each point is one design
    - X-axis = volume (material)
    - Y-axis = drift (stiffness)
    - Pareto-optimal designs are highlighted
    - The frontier shows the best possible trade-offs
    
    HOW IT WORKS (PSEUDOCODE):
    --------------------------
    
    STEP 1: UNDERSTAND THE DATA
    ---------------------------
    We have:
    - df: DataFrame with all designs (columns: volume, drift, ok, etc.)
    - mask: Boolean Series (True for Pareto-optimal designs)
    - x, y: Column names for the two objectives
    
    STEP 2: FILTER DATA
    -------------------
    - Only plot successful designs (ok=True)
    - Separate Pareto designs from dominated designs
    - This makes the plot clearer
    
    STEP 3: CREATE THE PLOT
    -----------------------
    - Scatter plot: all designs (gray, small)
    - Highlight Pareto designs (red, larger, with edge)
    - Add labels, grid, legend
    - Save to file
    
    STEP 4: INTERPRETATION
    ----------------------
    The plot shows:
    - Lower-left = best (low volume AND low drift)
    - Upper-right = worst (high volume AND high drift)
    - Pareto frontier = boundary of optimal designs
    - Moving along frontier = trade-off (improve one, worsen other)
    
    ENGINEERING INTERPRETATION:
    ---------------------------
    - **Ideal design**: Bottom-left corner (low volume, low drift)
    - **Pareto frontier**: The curve connecting optimal designs
    - **Trade-off**: Can't improve both objectives simultaneously
    - **Design choice**: Engineer picks based on priorities
      - "I care more about cost" → choose low volume
      - "I care more about stiffness" → choose low drift
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with design results
        Must have columns: x, y, ok (at minimum)
    
    mask : pd.Series
        Boolean mask: True for Pareto-optimal designs
        Same length as df, can be used as: df[mask]
    
    outpath : str
        File path to save the plot (e.g., "artifacts/pareto.png")
        - Will create directory if needed
        - Supports .png, .pdf, .svg formats
    
    x : str
        Column name for x-axis (default: "volume")
        - Represents first objective (material/cost)
    
    y : str
        Column name for y-axis (default: "drift")
        - Represents second objective (stiffness/performance)
    
    title : str
        Plot title (default: "Pareto Frontier: Volume vs Drift")
    
    Returns:
    --------
    None
        Saves plot to file (doesn't display)
    
    Example:
    --------
    >>> from mini_branch.pareto import pareto_mask
    >>> mask = pareto_mask(df)
    >>> plot_pareto(df, mask, "artifacts/pareto.png")
    """
    
    # ========================================================================
    # STEP 1: FILTER TO SUCCESSFUL DESIGNS ONLY
    # ========================================================================
    """
    WHY FILTER?
    -----------
    - Failed designs (ok=False) have NaN for metrics
    - We can't plot NaN values (they're not numbers)
    - Only successful designs are meaningful for comparison
    """
    
    # Only plot designs that solved successfully
    successful = df[df['ok'] == True].copy()
    
    if len(successful) == 0:
        raise ValueError("No successful designs to plot (all ok=False)")
    
    # Separate Pareto designs from dominated designs
    pareto_designs = successful[mask[successful.index]]
    dominated_designs = successful[~mask[successful.index]]
    
    # ========================================================================
    # STEP 2: EXTRACT DATA FOR PLOTTING
    # ========================================================================
    """
    WHY EXTRACT TO ARRAYS?
    ----------------------
    - Faster plotting with NumPy arrays
    - Easier to handle missing values
    - Cleaner code
    """
    
    # Pareto designs
    x_pareto = pareto_designs[x].values
    y_pareto = pareto_designs[y].values
    
    # Dominated designs
    x_dominated = dominated_designs[x].values
    y_dominated = dominated_designs[y].values
    
    # ========================================================================
    # STEP 3: CREATE THE FIGURE
    # ========================================================================
    """
    WHY MATPLOTLIB?
    ---------------
    - Standard Python plotting library
    - Good control over appearance
    - Easy to save to file
    - Works well with pandas DataFrames
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ========================================================================
    # STEP 4: PLOT DOMINATED DESIGNS (BACKGROUND)
    # ========================================================================
    """
    WHY PLOT DOMINATED DESIGNS?
    ----------------------------
    - Shows the full design space (not just optimal)
    - Helps understand where Pareto designs sit
    - Shows how many designs were explored
    - Makes Pareto designs stand out by contrast
    """
    
    if len(dominated_designs) > 0:
        ax.scatter(
            x_dominated, y_dominated,
            c='lightgray',
            s=30,  # Small points
            alpha=0.5,
            label=f'Dominated designs ({len(dominated_designs)})',
            edgecolors='none',
            zorder=1,  # Behind Pareto points
        )
    
    # ========================================================================
    # STEP 5: PLOT PARETO DESIGNS (HIGHLIGHTED)
    # ========================================================================
    """
    WHY HIGHLIGHT PARETO DESIGNS?
    ------------------------------
    - These are the designs worth considering
    - Make them visually prominent
    - Use color, size, and edge to stand out
    - These are the "winners" of the optimization
    """
    
    if len(pareto_designs) > 0:
        ax.scatter(
            x_pareto, y_pareto,
            c='red',
            s=100,  # Larger points
            alpha=0.8,
            label=f'Pareto-optimal ({len(pareto_designs)})',
            edgecolors='darkred',
            linewidths=2,
            zorder=2,  # In front of dominated points
        )
    
    # ========================================================================
    # STEP 6: ADD FORMATTING AND LABELS
    # ========================================================================
    """
    WHY FORMATTING MATTERS?
    ------------------------
    - Clear labels help interpretation
    - Grid makes it easier to read values
    - Title explains what the plot shows
    - Legend identifies what each color means
    """
    
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # ========================================================================
    # STEP 7: ADD ANNOTATIONS (OPTIONAL)
    # ========================================================================
    """
    WHY ANNOTATIONS?
    ----------------
    - Help identify key designs
    - Show design counts
    - Add context (e.g., "Lower is better")
    """
    
    # Add text box with summary
    textstr = f'Total designs: {len(successful)}\n'
    textstr += f'Pareto-optimal: {len(pareto_designs)}\n'
    textstr += f'Dominated: {len(dominated_designs)}\n'
    textstr += '\nLower values are better'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=props)
    
    # ========================================================================
    # STEP 8: SAVE THE PLOT
    # ========================================================================
    """
    WHY SAVE TO FILE?
    -----------------
    - Can be included in reports
    - Can be shared with others
    - Can be version controlled
    - Doesn't require display (works on servers)
    """
    
    import os
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Pareto plot saved to: {outpath}")


def plot_frame_deformed(
    nodes: Dict[int, Node],
    elements: List[Frame2D],
    d: np.ndarray,
    outpath: str,
    scale: float = 50.0,
    title: str = "Portal Frame: Deformed Shape",
) -> None:
    """
    Plot a portal frame showing both undeformed and deformed shapes.
    
    WHY THIS FUNCTION?
    ------------------
    After solving a structure, we want to visualize how it deforms under load.
    This helps engineers:
    - Verify results make physical sense
    - Understand structural behavior
    - Communicate findings to others
    - Identify potential issues (excessive deflections, unexpected shapes)
    
    HOW IT WORKS (PSEUDOCODE):
    --------------------------
    
    STEP 1: EXTRACT COORDINATES
    ---------------------------
    For each node:
        x_undeformed = node.x (original position)
        y_undeformed = node.y (original position)
        ux = displacement in x direction (from d vector)
        uy = displacement in y direction (from d vector)
        x_deformed = x_undeformed + scale × ux
        y_deformed = y_undeformed + scale × uy
    
    STEP 2: DRAW ELEMENTS
    ---------------------
    For each element (connects node i to node j):
        Draw line from (xi, yi) to (xj, yj) for undeformed
        Draw line from (xi_def, yi_def) to (xj_def, yj_def) for deformed
    
    STEP 3: ADD DETAILS
    -------------------
    - Plot nodes (circles)
    - Plot supports (triangles at base)
    - Add labels, grid, legend
    - Save to file
    
    STEP 4: SCALING
    ---------------
    Real deflections are tiny (millimeters). We scale them up visually
    (e.g., ×50 or ×100) so we can see the deformation. This is called
    "exaggerated deformation" - it's for visualization only, not the
    actual magnitude.
    
    ENGINEERING INTERPRETATION:
    ---------------------------
    - **Undeformed shape**: Original structure (before loading)
    - **Deformed shape**: Structure after loading (how it actually moves)
    - **Scale factor**: How much we exaggerate deflections for visibility
    - **Physical meaning**: 
      - Beam sags down (gravity load)
      - Frame sways to the right (lateral load)
      - Columns bend (combined axial + bending)
    
    Parameters:
    -----------
    nodes : Dict[int, Node]
        Dictionary mapping node IDs to Node objects
        Example: {0: Node(0, 0.0, 0.0), 1: Node(1, 0.0, 3.0), ...}
    
    elements : List[Frame2D]
        List of frame elements
        Example: [Frame2D(0, 0, 1, E=..., A=..., I=...), ...]
    
    d : np.ndarray
        Global displacement vector from solve_linear
        Shape: (ndof,) where ndof = 3 × number_of_nodes
        Contains: [ux_0, uy_0, rz_0, ux_1, uy_1, rz_1, ...]
    
    outpath : str
        File path to save the plot (e.g., "artifacts/winner_frame.png")
        - Will create directory if needed
        - Supports .png, .pdf, .svg formats
    
    scale : float
        Scale factor for deformation visualization (default: 50.0)
        - Real deflections are tiny (millimeters)
        - Scale them up so we can see them
        - Example: scale=50 means 1mm deflection → 50mm on plot
        - This is for visualization only (not actual magnitude)
    
    title : str
        Plot title (default: "Portal Frame: Deformed Shape")
    
    Returns:
    --------
    None
        Saves plot to file (doesn't display)
    
    Example:
    --------
    >>> nodes, elements, fixed, udls, loads = make_portal(params, material, sections)
    >>> K = assemble_global_K(nodes, elements)
    >>> F = assemble_element_loads_global(nodes, elements, udls)
    >>> d, R, _ = solve_linear(K, F, fixed)
    >>> plot_frame_deformed(nodes, elements, d, "artifacts/frame.png")
    """
    
    # ========================================================================
    # STEP 1: EXTRACT UNDEFORMED COORDINATES
    # ========================================================================
    """
    WHY EXTRACT COORDINATES?
    ------------------------
    - Need node positions to draw elements
    - Store in arrays for easy plotting
    - Separate x and y for matplotlib
    """
    
    node_ids = sorted(nodes.keys())
    xs_undeformed = np.array([nodes[i].x for i in node_ids])
    ys_undeformed = np.array([nodes[i].y for i in node_ids])
    
    # ========================================================================
    # STEP 2: EXTRACT DISPLACEMENTS
    # ========================================================================
    """
    WHY EXTRACT DISPLACEMENTS?
    -------------------------
    - Displacements are stored in d vector
    - Need to map from global DOF index to node
    - Extract ux and uy for each node
    - rz (rotation) is not used for plotting (only translation matters)
    """
    
    ux = np.array([d[dof_index(node_id, 0)] for node_id in node_ids])
    uy = np.array([d[dof_index(node_id, 1)] for node_id in node_ids])
    
    # ========================================================================
    # STEP 3: COMPUTE DEFORMED COORDINATES
    # ========================================================================
    """
    WHY SCALE DEFLECTIONS?
    ----------------------
    - Real deflections are tiny (millimeters)
    - Can't see them at true scale
    - Scale them up visually (exaggerate)
    - This is standard practice in structural visualization
    
    PHYSICAL EXAMPLE:
    -----------------
    If a 3m tall column deflects 5mm horizontally:
    - True deflection: 5mm = 0.005m
    - At scale=50: 5mm × 50 = 250mm = 0.25m on plot
    - Now we can see it clearly!
    """
    
    xs_deformed = xs_undeformed + scale * ux
    ys_deformed = ys_undeformed + scale * uy
    
    # ========================================================================
    # STEP 4: CREATE THE FIGURE
    # ========================================================================
    """
    WHY CREATE FIGURE?
    -----------------
    - matplotlib requires a figure and axes
    - Can set size, style, etc.
    - Standard plotting workflow
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # ========================================================================
    # STEP 5: DRAW UNDEFORMED ELEMENTS
    # ========================================================================
    """
    WHY DRAW UNDEFORMED FIRST?
    ---------------------------
    - Shows original structure (reference)
    - Makes deformed shape stand out
    - Helps understand how much it moved
    - Standard visualization practice
    """
    
    for element in elements:
        # Get node indices in our sorted list
        i_idx = node_ids.index(element.ni)
        j_idx = node_ids.index(element.nj)
        
        # Draw line from node i to node j
        ax.plot(
            [xs_undeformed[i_idx], xs_undeformed[j_idx]],
            [ys_undeformed[i_idx], ys_undeformed[j_idx]],
            'b-',  # Blue, solid line
            linewidth=3,
            alpha=0.7,
            label='Undeformed' if element == elements[0] else '',  # Label only once
            zorder=1,  # Behind deformed
        )
    
    # ========================================================================
    # STEP 6: DRAW DEFORMED ELEMENTS
    # ========================================================================
    """
    WHY DRAW DEFORMED?
    ------------------
    - Shows how structure actually moves
    - Helps verify results make sense
    - Shows structural behavior
    - Critical for understanding response
    """
    
    for element in elements:
        # Get node indices
        i_idx = node_ids.index(element.ni)
        j_idx = node_ids.index(element.nj)
        
        # Draw line from deformed node i to deformed node j
        ax.plot(
            [xs_deformed[i_idx], xs_deformed[j_idx]],
            [ys_deformed[i_idx], ys_deformed[j_idx]],
            'r--',  # Red, dashed line
            linewidth=2,
            alpha=0.9,
            label=f'Deformed (×{scale} scale)' if element == elements[0] else '',
            zorder=2,  # In front of undeformed
        )
    
    # ========================================================================
    # STEP 7: DRAW NODES
    # ========================================================================
    """
    WHY DRAW NODES?
    ---------------
    - Shows connection points
    - Helps identify structure
    - Makes plot clearer
    - Standard visualization element
    """
    
    # Undeformed nodes
    ax.plot(xs_undeformed, ys_undeformed, 'bo', markersize=10, label='Nodes', zorder=3)
    
    # Deformed nodes
    ax.plot(xs_deformed, ys_deformed, 'ro', markersize=8, alpha=0.7, zorder=3)
    
    # ========================================================================
    # STEP 8: DRAW SUPPORTS
    # ========================================================================
    """
    WHY DRAW SUPPORTS?
    ------------------
    - Shows boundary conditions
    - Helps understand structure
    - Standard engineering visualization
    - Makes plot complete
    """
    
    # Find base nodes (y = 0 or minimum y)
    base_y = min(ys_undeformed)
    base_nodes = [i for i in node_ids if abs(nodes[i].y - base_y) < 1e-6]
    
    # Draw support triangles
    support_size = 0.3  # Size of triangle
    for node_id in base_nodes:
        idx = node_ids.index(node_id)
        x_base = xs_undeformed[idx]
        y_base = ys_undeformed[idx]
        
        # Triangle pointing up (pinned support symbol)
        triangle_x = [x_base - support_size, x_base, x_base + support_size]
        triangle_y = [y_base - support_size, y_base, y_base - support_size]
        
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2, 
                label='Support (pinned)' if node_id == base_nodes[0] else '',
                zorder=4)
    
    # ========================================================================
    # STEP 9: ADD FORMATTING AND LABELS
    # ========================================================================
    """
    WHY FORMATTING?
    ---------------
    - Clear labels help interpretation
    - Grid makes it easier to read
    - Title explains what we're showing
    - Legend identifies elements
    """
    
    ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')  # Equal aspect ratio (important for structures)
    
    # ========================================================================
    # STEP 10: ADD ANNOTATIONS
    # ========================================================================
    """
    WHY ANNOTATIONS?
    ----------------
    - Show scale factor (important!)
    - Add context
    - Help interpretation
    """
    
    # Add text box with scale info
    textstr = f'Deformation scale: ×{scale}\n'
    textstr += f'(Deflections exaggerated for visibility)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=props)
    
    # ========================================================================
    # STEP 11: SAVE THE PLOT
    # ========================================================================
    """
    WHY SAVE TO FILE?
    -----------------
    - Can be included in reports
    - Can be shared with others
    - Can be version controlled
    - Doesn't require display (works on servers)
    """
    
    import os
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Frame plot saved to: {outpath}")
