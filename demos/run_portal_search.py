"""
PORTAL FRAME DESIGN SEARCH DEMO
================================

PURPOSE:
--------
This script demonstrates the complete Day 3 workflow:
1. Generate hundreds of portal frame design variants
2. Evaluate each variant (solve, extract metrics)
3. Identify Pareto-optimal designs
4. Visualize results (Pareto plot, deformed shapes)
5. Export results (CSV, markdown summary)

WHY THIS SCRIPT?
----------------
This is the "main event" of Day 3 - it ties together all the pieces:
- explore.py: Generates and evaluates variants
- pareto.py: Identifies optimal designs
- viz.py: Creates visualizations
- Results: CSV for analysis, plots for communication

ENGINEERING CONTEXT:
--------------------
This script performs "design space exploration":
- We don't know the "best" design upfront
- We generate many variants and evaluate them
- We identify the Pareto frontier (best trade-offs)
- Engineers can then choose based on priorities

This is how real engineering optimization works:
1. Define design space (parameters and ranges)
2. Sample many designs
3. Evaluate performance
4. Identify optimal designs
5. Make decision based on priorities
"""

import argparse
import os
import pandas as pd

from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL
from mini_branch.explore import run_search, make_portal, evaluate_variant
from mini_branch.pareto import pareto_mask
from mini_branch.viz import plot_pareto, plot_frame_deformed
from mini_branch.assembly import assemble_global_K
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear


def main():
    """
    MAIN WORKFLOW (PSEUDOCODE):
    ===========================
    
    STEP 1: PARSE COMMAND LINE ARGUMENTS
    -------------------------------------
    - Get number of variants to generate (--n)
    - Get random seed for reproducibility (--seed)
    - These control the search process
    
    STEP 2: RUN THE DESIGN SEARCH
    ------------------------------
    - Call run_search(n, seed)
    - This generates variants, evaluates them, returns DataFrame
    - DataFrame has one row per variant with all metrics
    
    STEP 3: SAVE RESULTS TO CSV
    ---------------------------
    - Save DataFrame to artifacts/results.csv
    - This is the "raw data" - can be analyzed later
    - Day 4 will use this for ML training
    
    STEP 4: IDENTIFY PARETO FRONTIER
    --------------------------------
    - Call pareto_mask(df) to find optimal designs
    - Filter DataFrame to Pareto designs only
    - These are the designs worth considering
    
    STEP 5: CREATE PARETO PLOT
    --------------------------
    - Call plot_pareto(df, mask, "artifacts/pareto.png")
    - Shows volume vs drift trade-off
    - Highlights Pareto-optimal designs
    - Visual representation of design space
    
    STEP 6: CREATE TOP 10 SUMMARY
    -----------------------------
    - Sort Pareto designs by volume (or other metric)
    - Write top 10 to artifacts/top10.md
    - Includes parameters and metrics for each design
    - Human-readable summary for engineers
    
    STEP 7: VISUALIZE WINNER DESIGN
    ---------------------------------
    - Pick one "winner" design (e.g., lowest volume Pareto design)
    - Re-evaluate to get full model (nodes, elements, displacements)
    - Call plot_frame_deformed() to show how it deforms
    - Save to artifacts/winner_frame.png
    
    STEP 8: PRINT SUMMARY
    ---------------------
    - Print statistics (total designs, Pareto count, etc.)
    - Print file locations
    - Confirm completion
    """
    
    # ========================================================================
    # STEP 1: PARSE COMMAND LINE ARGUMENTS
    # ========================================================================
    """
    WHY COMMAND LINE ARGUMENTS?
    ----------------------------
    - Makes script flexible (can run with different parameters)
    - Easy to automate (can call from scripts)
    - Reproducible (same seed = same results)
    - Standard practice for command-line tools
    
    EXAMPLE USAGE:
    --------------
    python demos/run_portal_search.py --n 500 --seed 42
    """
    
    parser = argparse.ArgumentParser(
        description='Run portal frame design search and generate results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demos/run_portal_search.py --n 500 --seed 42
  python demos/run_portal_search.py --n 1000 --seed 123
  
This will generate:
  - artifacts/results.csv (all designs)
  - artifacts/pareto.png (Pareto plot)
  - artifacts/top10.md (top designs summary)
  - artifacts/winner_frame.png (deformed shape)
        """
    )
    
    parser.add_argument(
        '--n',
        type=int,
        default=500,
        help='Number of design variants to generate and evaluate (default: 500)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PORTAL FRAME DESIGN SEARCH")
    print("=" * 70)
    print(f"Generating {args.n} design variants (seed={args.seed})")
    print()
    
    # ========================================================================
    # STEP 2: RUN THE DESIGN SEARCH
    # ========================================================================
    """
    WHY run_search()?
    -----------------
    - Encapsulates the entire search loop
    - Handles variant generation, evaluation, error handling
    - Returns clean DataFrame ready for analysis
    - This is the core of Day 3's exploration workflow
    
    WHAT IT DOES:
    -------------
    1. Generates n random design variants (using sample_params)
    2. Evaluates each variant (using evaluate_variant)
    3. Collects all results into DataFrame
    4. Returns DataFrame with columns:
       - All PortalParams fields (span, height, brace, etc.)
       - Metrics (drift, max_abs_M, volume, carbon)
       - Status (ok, reason)
    """
    
    print("STEP 1: Running design search...")
    print("-" * 70)
    
    df = run_search(n=args.n, seed=args.seed)
    
    print(f"Search complete: {len(df)} designs evaluated")
    print(f"  Successful: {df['ok'].sum()}")
    print(f"  Failed: {(~df['ok']).sum()}")
    print()
    
    # ========================================================================
    # STEP 3: SAVE RESULTS TO CSV
    # ========================================================================
    """
    WHY SAVE TO CSV?
    ----------------
    - Permanent record of all designs
    - Can be analyzed later (Excel, Python, etc.)
    - Day 4 will use this for ML training
    - Can be version controlled
    - Easy to share with others
    
    CSV FORMAT:
    ----------
    - One row per design variant
    - Columns: span, height, brace, sec_col, sec_beam, sec_brace,
               udl_w, wind_P, shipping_limit, drift, max_abs_M,
               volume, carbon, ok, reason
    - Can be opened in Excel, pandas, etc.
    """
    
    print("STEP 2: Saving results to CSV...")
    print("-" * 70)
    
    os.makedirs('artifacts', exist_ok=True)
    csv_path = 'artifacts/results.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {csv_path}")
    print(f"  Total designs: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print()
    
    # ========================================================================
    # STEP 4: IDENTIFY PARETO FRONTIER
    # ========================================================================
    """
    WHY PARETO ANALYSIS?
    --------------------
    - We have hundreds of designs, but only some are optimal
    - Pareto frontier identifies the "best" trade-offs
    - Engineers only need to consider Pareto designs
    - Makes decision-making easier
    
    WHAT IS PARETO-OPTIMAL?
    ------------------------
    A design is Pareto-optimal if:
    - No other design has BOTH lower volume AND lower drift
    - These designs represent the best possible trade-offs
    - Moving along the frontier: improve one objective, worsen the other
    
    ENGINEERING INTERPRETATION:
    ---------------------------
    - Pareto designs = "efficient frontier"
    - These are the only designs worth considering
    - Engineer picks based on priorities:
      - "I care more about cost" → choose low volume
      - "I care more about stiffness" → choose low drift
    """
    
    print("STEP 3: Identifying Pareto-optimal designs...")
    print("-" * 70)
    
    # Only consider successful designs
    successful_df = df[df['ok'] == True].copy()
    
    if len(successful_df) == 0:
        print("ERROR: No successful designs to analyze!")
        return
    
    # Find Pareto-optimal designs
    mask = pareto_mask(successful_df, x="volume", y="drift")
    pareto_df = successful_df[mask].copy()
    
    print(f"Pareto analysis complete:")
    print(f"  Total successful designs: {len(successful_df)}")
    print(f"  Pareto-optimal designs: {len(pareto_df)}")
    print(f"  Dominated designs: {len(successful_df) - len(pareto_df)}")
    print()
    
    # ========================================================================
    # STEP 5: CREATE PARETO PLOT
    # ========================================================================
    """
    WHY VISUALIZE?
    --------------
    - Numbers in CSV are hard to interpret
    - Plot shows patterns instantly
    - Pareto frontier is clearly visible
    - Helps engineers understand trade-offs
    
    WHAT THE PLOT SHOWS:
    ----------------
    - X-axis: Volume (material/cost)
    - Y-axis: Drift (stiffness/performance)
    - Gray points: Dominated designs (can be improved)
    - Red points: Pareto-optimal designs (best trade-offs)
    - Lower-left corner = ideal (low volume AND low drift)
    - Frontier shows the best possible trade-offs
    """
    
    print("STEP 4: Creating Pareto plot...")
    print("-" * 70)
    
    # Create mask for full DataFrame (including failed designs)
    full_mask = pd.Series(False, index=df.index)
    full_mask[successful_df.index[mask]] = True
    
    plot_path = 'artifacts/pareto.png'
    plot_pareto(df, full_mask, plot_path)
    
    print(f"Pareto plot saved to: {plot_path}")
    print()
    
    # ========================================================================
    # STEP 6: CREATE TOP 10 SUMMARY
    # ========================================================================
    """
    WHY TOP 10 SUMMARY?
    -------------------
    - Engineers need a quick summary of best designs
    - Markdown is human-readable
    - Can be included in reports
    - Shows key parameters and metrics
    
    WHAT TO INCLUDE:
    ----------------
    - Design parameters (span, height, sections, etc.)
    - Performance metrics (drift, volume, carbon)
    - Why it's good (Pareto-optimal, low volume, etc.)
    - Easy to scan and compare
    """
    
    print("STEP 5: Creating top 10 summary...")
    print("-" * 70)
    
    # Sort Pareto designs by volume (lowest first)
    top_designs = pareto_df.nsmallest(10, 'volume')
    
    md_path = 'artifacts/top10.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Top 10 Pareto-Optimal Designs\n\n")
        f.write("These designs represent the best trade-offs between volume (material/cost) ")
        f.write("and drift (stiffness/performance).\n\n")
        f.write("**Lower values are better for both objectives.**\n\n")
        f.write("---\n\n")
        
        for i, (idx, row) in enumerate(top_designs.iterrows(), 1):
            f.write(f"## Design #{i}\n\n")
            f.write("### Parameters\n")
            f.write(f"- **Span**: {row['span']:.2f} m\n")
            f.write(f"- **Height**: {row['height']:.2f} m\n")
            f.write(f"- **Bracing**: {'X-brace' if row['brace'] == 1 else 'None'}\n")
            f.write(f"- **Column section**: {TIMBER_SECTIONS[int(row['sec_col'])].name}\n")
            f.write(f"- **Beam section**: {TIMBER_SECTIONS[int(row['sec_beam'])].name}\n")
            if row['brace'] == 1:
                f.write(f"- **Brace section**: {TIMBER_SECTIONS[int(row['sec_brace'])].name}\n")
            f.write(f"- **UDL**: {row['udl_w']:.0f} N/m\n")
            f.write(f"- **Wind load**: {row['wind_P']:.0f} N\n")
            f.write("\n### Performance Metrics\n")
            f.write(f"- **Drift**: {row['drift']*1000:.2f} mm\n")
            f.write(f"- **Max moment**: {row['max_abs_M']:.0f} N·m\n")
            f.write(f"- **Volume**: {row['volume']:.4f} m³\n")
            f.write(f"- **Carbon**: {row['carbon']:.2f} kg CO2\n")
            f.write("\n---\n\n")
    
    print(f"Top 10 summary saved to: {md_path}")
    print()
    
    # ========================================================================
    # STEP 7: VISUALIZE WINNER DESIGN
    # ========================================================================
    """
    WHY VISUALIZE ONE DESIGN?
    -------------------------
    - Shows how the structure actually deforms
    - Helps verify results make physical sense
    - Useful for reports and presentations
    - Engineers can see structural behavior
    
    WHICH DESIGN TO PICK?
    ---------------------
    - Pick a Pareto-optimal design (it's optimal)
    - Could pick lowest volume, lowest drift, or balanced
    - For demo, we'll pick the lowest volume Pareto design
    - This shows a "cost-optimized" solution
    """
    
    print("STEP 6: Visualizing winner design...")
    print("-" * 70)
    
    if len(pareto_df) == 0:
        print("WARNING: No Pareto designs to visualize!")
    else:
        # Pick the lowest volume Pareto design as "winner"
        winner = pareto_df.nsmallest(1, 'volume').iloc[0]
        
        # Re-evaluate to get full model (we need nodes, elements, displacements)
        from mini_branch.explore import PortalParams
        
        winner_params = PortalParams(
            span=winner['span'],
            height=winner['height'],
            brace=int(winner['brace']),
            sec_col=int(winner['sec_col']),
            sec_beam=int(winner['sec_beam']),
            sec_brace=int(winner['sec_brace']),
            udl_w=winner['udl_w'],
            wind_P=winner['wind_P'],
            shipping_limit=winner['shipping_limit'],
        )
        
        # Build model
        nodes, elements, fixed_dofs, element_udl_map, nodal_loads = make_portal(
            winner_params, DEFAULT_MATERIAL, TIMBER_SECTIONS
        )
        
        # Solve
        K = assemble_global_K(nodes, elements)
        ndof = 3 * len(nodes)
        F = assemble_element_loads_global(nodes, elements, element_udl_map)
        
        from mini_branch.assembly import dof_index
        for node_id, load_vec in nodal_loads.items():
            F[dof_index(node_id, 0)] += load_vec[0]
            F[dof_index(node_id, 1)] += load_vec[1]
            F[dof_index(node_id, 2)] += load_vec[2]
        
        d, R, _ = solve_linear(K, F, fixed_dofs)
        
        # Plot
        frame_path = 'artifacts/winner_frame.png'
        plot_frame_deformed(nodes, elements, d, frame_path, 
                           title=f"Winner Design: Volume={winner['volume']:.4f} m³, Drift={winner['drift']*1000:.2f} mm")
        
        print(f"Winner design visualization saved to: {frame_path}")
        print(f"  Volume: {winner['volume']:.4f} m³")
        print(f"  Drift: {winner['drift']*1000:.2f} mm")
        print()
    
    # ========================================================================
    # STEP 8: PRINT SUMMARY
    # ========================================================================
    """
    WHY PRINT SUMMARY?
    ------------------
    - Confirms completion
    - Shows what was generated
    - Helps user find results
    - Standard practice for scripts
    """
    
    print("=" * 70)
    print("DESIGN SEARCH COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  [CSV] {csv_path} ({len(df)} designs)")
    print(f"  [PLOT] {plot_path} (Pareto frontier)")
    print(f"  [MD] {md_path} (Top 10 designs)")
    if len(pareto_df) > 0:
        print(f"  [FRAME] {frame_path} (Winner design)")
    print()
    print("Next steps:")
    print("  - Review Pareto plot to understand trade-offs")
    print("  - Check top10.md for best designs")
    print("  - Use results.csv for further analysis")
    print("  - Day 4: Train ML surrogate on results.csv")
    print()


if __name__ == "__main__":
    main()

