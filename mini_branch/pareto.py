"""
PARETO FRONTIER: IDENTIFYING OPTIMAL TRADE-OFFS
===============================================

PURPOSE:
--------
This module identifies the "Pareto frontier" - a set of designs that represent
the best possible trade-offs between competing objectives. In structural
engineering, we often want to minimize BOTH material (cost/carbon) AND deflection
(drift), but these goals conflict: more material = less drift, less material = more drift.

WHY THIS MATTERS:
-----------------
1. **Design Decision Support**: The Pareto frontier shows engineers which designs
   are "non-dominated" - meaning you can't improve one objective without making
   the other worse. These are the only designs worth considering.

2. **Multi-Objective Optimization**: Instead of picking a single "best" design,
   we identify a set of optimal trade-offs. The engineer can then choose based on
   their priorities (e.g., "I care more about cost than stiffness").

3. **Visualization**: The Pareto frontier creates a clear visual boundary in
   the design space, making trade-offs obvious.

ENGINEERING CONTEXT:
--------------------
In our portal frame problem:
- **Objective 1: Minimize Volume** (less material = lower cost/carbon)
- **Objective 2: Minimize Drift** (stiffer structure = better performance)

A design is "Pareto-optimal" (non-dominated) if:
- No other design has BOTH lower volume AND lower drift
- At least one other design must be strictly worse in one objective

Example:
- Design A: volume=0.5 m³, drift=0.01 m
- Design B: volume=0.6 m³, drift=0.01 m  → Dominated by A (same drift, more volume)
- Design C: volume=0.5 m³, drift=0.02 m  → Dominated by A (same volume, more drift)
- Design D: volume=0.7 m³, drift=0.005 m → NOT dominated (better drift, worse volume)

Designs A and D are both Pareto-optimal (they represent different trade-offs).
"""

import numpy as np
import pandas as pd


def pareto_mask(
    df: pd.DataFrame,
    x: str = "volume",
    y: str = "drift",
    lower_is_better: bool = True,
) -> pd.Series:
    """
    Identify Pareto-optimal (non-dominated) designs in a DataFrame.
    
    WHY THIS FUNCTION?
    ------------------
    After running a design search, we have hundreds of designs with different
    volume and drift values. We need to identify which ones are "Pareto-optimal"
    - meaning they represent the best possible trade-offs. This function returns
    a boolean mask (True for Pareto points, False for dominated points).
    
    HOW IT WORKS (PSEUDOCODE):
    --------------------------
    
    STEP 1: UNDERSTAND THE PROBLEM
    -------------------------------
    We have a DataFrame with columns:
    - x: First objective (e.g., "volume")
    - y: Second objective (e.g., "drift")
    - Each row is one design variant
    
    We want to find designs where:
    - No other design has BOTH lower x AND lower y
    - These are the "non-dominated" or "Pareto-optimal" designs
    
    STEP 2: THE ALGORITHM (for each design)
    ---------------------------------------
    For design i:
        is_pareto = True  (assume it's optimal)
        
        For every other design j:
            If design j has:
                - x_j <= x_i  AND  y_j <= y_i
                - AND at least one is strictly better (< not <=)
            Then:
                design i is dominated (not Pareto)
                is_pareto = False
                break (no need to check more)
        
        If we checked all others and none dominated i:
            design i is Pareto-optimal
    
    STEP 3: HANDLE EDGE CASES
    -------------------------
    - Missing values (NaN): Skip designs with NaN in x or y
    - Only successful designs: Filter to ok=True first (optional)
    - Multiple objectives: Can extend to 3+ objectives (Day 4)
    
    STEP 4: RETURN BOOLEAN MASK
    ---------------------------
    Return a pandas Series (boolean array) where:
    - True = Pareto-optimal design
    - False = Dominated design
    
    This mask can be used to filter the DataFrame:
        pareto_designs = df[pareto_mask(df)]
    
    ENGINEERING INTERPRETATION:
    ---------------------------
    The Pareto frontier represents the "efficient frontier" of designs:
    - Moving along the frontier: improving one objective worsens the other
    - Designs off the frontier: can be improved in both objectives
    - Engineers choose based on priorities: "I'll accept 10% more volume
      if it reduces drift by 20%"
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with design results
        Must have columns specified by x and y parameters
    
    x : str
        Column name for first objective (default: "volume")
        - Lower values are better (minimize)
    
    y : str
        Column name for second objective (default: "drift")
        - Lower values are better (minimize)
    
    lower_is_better : bool
        If True, lower values are better (minimize both objectives)
        If False, higher values are better (maximize both objectives)
        - Default: True (minimize volume and drift)
    
    Returns:
    --------
    pd.Series
        Boolean mask: True for Pareto-optimal designs, False for dominated
        - Same length as input DataFrame
        - Can be used to filter: df[pareto_mask(df)]
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'volume': [0.5, 0.6, 0.5, 0.7],
    ...     'drift': [0.01, 0.01, 0.02, 0.005],
    ...     'ok': [True, True, True, True]
    ... })
    >>> mask = pareto_mask(df)
    >>> pareto_designs = df[mask]
    >>> # Returns designs with volume=0.5,drift=0.01 and volume=0.7,drift=0.005
    """
    
    # ========================================================================
    # STEP 1: VALIDATE INPUTS
    # ========================================================================
    """
    WHY VALIDATE?
    -------------
    - Catch errors early (before processing)
    - Give clear error messages
    - Prevent crashes from missing columns
    """
    
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame. Available: {df.columns.tolist()}")
    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in DataFrame. Available: {df.columns.tolist()}")
    
    # ========================================================================
    # STEP 2: HANDLE MISSING VALUES
    # ========================================================================
    """
    WHY HANDLE NaN?
    ---------------
    - Designs that failed (ok=False) have NaN for metrics
    - We can't compare NaN values (they're not numbers)
    - Strategy: Mark designs with NaN as NOT Pareto (they're invalid)
    """
    
    # Find rows with valid (non-NaN) values in both objectives
    valid_mask = df[x].notna() & df[y].notna()
    
    # Initialize result mask: all False (assume dominated)
    pareto_mask_result = pd.Series(False, index=df.index)
    
    # If no valid designs, return all False
    if not valid_mask.any():
        return pareto_mask_result
    
    # Get valid designs only
    valid_df = df[valid_mask].copy()
    valid_indices = valid_df.index
    
    # ========================================================================
    # STEP 3: EXTRACT OBJECTIVE VALUES
    # ========================================================================
    """
    WHY EXTRACT TO ARRAYS?
    ----------------------
    - Faster to work with NumPy arrays than pandas Series
    - Can use vectorized operations
    - Easier to compare designs pairwise
    """
    
    x_values = valid_df[x].values  # Convert to NumPy array
    y_values = valid_df[y].values  # Convert to NumPy array
    
    # ========================================================================
    # STEP 4: IDENTIFY PARETO-OPTIMAL DESIGNS
    # ========================================================================
    """
    THE CORE ALGORITHM:
    -------------------
    For each design i, check if any other design j dominates it.
    
    Design j dominates design i if:
    - j is better or equal in BOTH objectives (x_j <= x_i AND y_j <= y_i)
    - AND j is strictly better in at least one (x_j < x_i OR y_j < y_i)
    
    If no design dominates i, then i is Pareto-optimal.
    
    WHY THIS WORKS:
    ---------------
    - If j dominates i, then i is not optimal (we can use j instead)
    - If no j dominates i, then i is on the frontier (can't improve both objectives)
    - This is the standard definition of Pareto optimality
    """
    
    n = len(valid_df)
    is_pareto = np.ones(n, dtype=bool)  # Start with all True (assume Pareto)
    
    # Compare each design to all others
    for i in range(n):
        # Skip if already marked as dominated
        if not is_pareto[i]:
            continue
        
        # Compare design i to all other designs j
        for j in range(n):
            if i == j:
                continue  # Don't compare design to itself
            
            # Check if design j dominates design i
            if lower_is_better:
                # Lower is better: j dominates i if j <= i in both, and < in at least one
                j_better_x = x_values[j] <= x_values[i]
                j_better_y = y_values[j] <= y_values[i]
                j_strictly_better_x = x_values[j] < x_values[i]
                j_strictly_better_y = y_values[j] < y_values[i]
            else:
                # Higher is better: j dominates i if j >= i in both, and > in at least one
                j_better_x = x_values[j] >= x_values[i]
                j_better_y = y_values[j] >= y_values[i]
                j_strictly_better_x = x_values[j] > x_values[i]
                j_strictly_better_y = y_values[j] > y_values[i]
            
            # j dominates i if:
            # - j is better or equal in BOTH objectives
            # - AND j is strictly better in at least one
            if (j_better_x and j_better_y) and (j_strictly_better_x or j_strictly_better_y):
                # Design i is dominated by design j
                is_pareto[i] = False
                break  # No need to check more designs against i
    
    # ========================================================================
    # STEP 5: MAP RESULTS BACK TO ORIGINAL DATAFRAME
    # ========================================================================
    """
    WHY MAP BACK?
    -------------
    - We only processed valid designs (no NaN)
    - But we need to return a mask for ALL rows in original DataFrame
    - Map the Pareto results back to original indices
    """
    
    # Set True for Pareto-optimal designs
    for idx, is_pareto_val in zip(valid_indices, is_pareto):
        pareto_mask_result[idx] = is_pareto_val
    
    # ========================================================================
    # STEP 6: RETURN RESULTS
    # ========================================================================
    """
    RETURN FORMAT:
    --------------
    - Boolean Series with same index as input DataFrame
    - True = Pareto-optimal design
    - False = Dominated design or invalid (NaN)
    
    USAGE:
    ------
    pareto_designs = df[pareto_mask(df)]
    """
    
    return pareto_mask_result


