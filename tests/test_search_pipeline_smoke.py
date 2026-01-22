"""
SMOKE TEST: DESIGN SEARCH PIPELINE
===================================

PURPOSE:
--------
This test verifies that the entire Day 3 design search pipeline works end-to-end.
It's called a "smoke test" because it checks that the system doesn't "catch fire"
(i.e., crash) when run - it's a basic sanity check.

WHY THIS TEST?
-------------
1. **Regression Prevention**: If we break something, this test will catch it
2. **Integration Check**: Verifies all pieces work together (explore, pareto, viz)
3. **Fast Feedback**: Runs quickly (only 10 designs) so we get immediate feedback
4. **Documentation**: Shows how to use the pipeline

TEST PHILOSOPHY:
---------------
- **Smoke test = basic functionality check**
  - Does it run without crashing?
  - Does it produce expected outputs?
  - Are results in the right format?
  
- **NOT a comprehensive test**
  - Doesn't verify correctness of results (that's for unit tests)
  - Doesn't test edge cases (that's for integration tests)
  - Just checks that the pipeline works

ENGINEERING CONTEXT:
--------------------
This test simulates what an engineer would do:
1. Run a design search
2. Get results
3. Check that results make sense
4. Verify outputs exist

If this test passes, we know the Day 3 pipeline is functional.
"""

import pytest
import pandas as pd
import os

from mini_branch.explore import run_search
from mini_branch.pareto import pareto_mask


def test_search_pipeline_smoke():
    """
    Test that the design search pipeline runs successfully.
    
    WHY THIS TEST?
    --------------
    This is a "smoke test" - it verifies the basic functionality:
    - Can we generate variants?
    - Can we evaluate them?
    - Do we get results in the right format?
    - Are there any successful designs?
    
    It does NOT verify:
    - That results are correct (that's for unit tests)
    - That metrics are accurate (that's for validation tests)
    - That Pareto analysis is correct (that's for pareto tests)
    
    HOW IT WORKS (PSEUDOCODE):
    --------------------------
    
    STEP 1: RUN SEARCH
    ------------------
    - Call run_search(n=10, seed=1)
    - This should complete without crashing
    - Returns a DataFrame
    
    STEP 2: CHECK DATAFRAME STRUCTURE
    ---------------------------------
    - Verify it's a pandas DataFrame
    - Check it has expected columns
    - Verify it has some rows (at least 1)
    
    STEP 3: CHECK FOR SUCCESSFUL DESIGNS
    -----------------------------------
    - At least one design should have ok=True
    - This means at least one variant solved successfully
    - If all fail, something is wrong
    
    STEP 4: CHECK METRICS FOR SUCCESSFUL DESIGNS
    --------------------------------------------
    - For designs with ok=True, metrics should not be NaN
    - drift, max_abs_M, volume, carbon should all be numbers
    - This verifies evaluation worked correctly
    
    STEP 5: CHECK PARETO ANALYSIS
    -----------------------------
    - Can we identify Pareto-optimal designs?
    - Does pareto_mask() work on the results?
    - This verifies the full pipeline works
    
    EXPECTED BEHAVIOR:
    -----------------
    - Search completes in reasonable time (< 10 seconds for 10 designs)
    - DataFrame has expected columns
    - At least one successful design
    - Metrics are valid numbers (not NaN)
    - Pareto analysis works
    """
    
    # ========================================================================
    # STEP 1: RUN THE SEARCH
    # ========================================================================
    """
    WHY SMALL N?
    ------------
    - Smoke test should be fast (< 10 seconds)
    - n=10 is enough to verify functionality
    - Don't need hundreds of designs for smoke test
    - Can test with more designs in integration tests
    """
    
    print("\n" + "=" * 70)
    print("SMOKE TEST: Running design search pipeline")
    print("=" * 70)
    print("Generating 10 design variants (seed=1)...")
    
    df = run_search(n=10, seed=1)
    
    print(f"Search complete: {len(df)} designs evaluated")
    
    # ========================================================================
    # STEP 2: VERIFY DATAFRAME STRUCTURE
    # ========================================================================
    """
    WHY CHECK STRUCTURE?
    --------------------
    - Ensures we got the right type of object
    - Verifies expected columns exist
    - Catches errors early
    - Makes test failures clear
    """
    
    print("\nChecking DataFrame structure...")
    
    # Should be a DataFrame
    assert isinstance(df, pd.DataFrame), "run_search() should return a pandas DataFrame"
    
    # Should have some rows
    assert len(df) > 0, "DataFrame should have at least one row"
    
    # Should have expected columns
    expected_columns = [
        'span', 'height', 'brace', 'sec_col', 'sec_beam', 'sec_brace',
        'udl_w', 'wind_P', 'shipping_limit',
        'drift', 'max_abs_M', 'volume', 'carbon',
        'ok', 'reason'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"DataFrame should have column '{col}'"
    
    print(f"✓ DataFrame structure is correct ({len(df.columns)} columns, {len(df)} rows)")
    
    # ========================================================================
    # STEP 3: CHECK FOR SUCCESSFUL DESIGNS
    # ========================================================================
    """
    WHY CHECK FOR SUCCESS?
    ---------------------
    - If all designs fail, something is wrong
    - At least one should solve successfully
    - This verifies the solver works
    - This verifies evaluation works
    """
    
    print("\nChecking for successful designs...")
    
    successful = df[df['ok'] == True]
    failed = df[df['ok'] == False]
    
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    # At least one should succeed
    assert len(successful) > 0, "At least one design should solve successfully"
    
    print("✓ At least one design solved successfully")
    
    # ========================================================================
    # STEP 4: CHECK METRICS FOR SUCCESSFUL DESIGNS
    # ========================================================================
    """
    WHY CHECK METRICS?
    -----------------
    - Successful designs should have valid metrics
    - NaN values indicate evaluation failed
    - This verifies metrics extraction works
    - This verifies the full pipeline works
    """
    
    print("\nChecking metrics for successful designs...")
    
    # Key metrics that should be numbers (not NaN)
    key_metrics = ['drift', 'max_abs_M', 'volume', 'carbon']
    
    for metric in key_metrics:
        # Check that successful designs have valid (non-NaN) values
        valid_values = successful[metric].notna()
        assert valid_values.all(), f"All successful designs should have valid '{metric}'"
        
        # Check that values are reasonable (not infinite, not negative for these)
        if metric in ['volume', 'carbon']:
            assert (successful[metric] >= 0).all(), f"'{metric}' should be non-negative"
        
        print(f"  ✓ {metric}: all successful designs have valid values")
    
    print("✓ All metrics are valid for successful designs")
    
    # ========================================================================
    # STEP 5: CHECK PARETO ANALYSIS
    # ========================================================================
    """
    WHY CHECK PARETO?
    ----------------
    - Verifies the full pipeline works (search → pareto)
    - Ensures pareto_mask() works on real data
    - This is part of the Day 3 workflow
    - Catches integration issues
    """
    
    print("\nChecking Pareto analysis...")
    
    if len(successful) > 0:
        # Should be able to compute Pareto mask
        mask = pareto_mask(successful, x="volume", y="drift")
        
        # Should be a boolean Series
        assert isinstance(mask, pd.Series), "pareto_mask() should return a pandas Series"
        assert len(mask) == len(successful), "Mask should have same length as DataFrame"
        
        # Should have at least one Pareto design (if we have successful designs)
        pareto_count = mask.sum()
        print(f"  Pareto-optimal designs: {pareto_count}")
        
        # At least one should be Pareto (the best one)
        assert pareto_count > 0, "At least one design should be Pareto-optimal"
        
        print("✓ Pareto analysis works correctly")
    else:
        print("  Skipping Pareto check (no successful designs)")
    
    # ========================================================================
    # STEP 6: SUMMARY
    # ========================================================================
    """
    WHY PRINT SUMMARY?
    ------------------
    - Makes test output clear
    - Shows what was checked
    - Confirms everything passed
    - Helps debugging if test fails
    """
    
    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print(f"✓ Generated {len(df)} designs")
    print(f"✓ {len(successful)} designs solved successfully")
    print(f"✓ All metrics are valid")
    print(f"✓ Pareto analysis works")
    print("\nThe design search pipeline is functional!")
    print()


def test_search_pipeline_reproducibility():
    """
    Test that the search is reproducible (same seed = same results).
    
    WHY THIS TEST?
    --------------
    - Reproducibility is critical for scientific work
    - Same seed should produce same variants
    - This verifies random number generation works
    - This is important for Day 4 (ML training needs reproducible data)
    
    HOW IT WORKS:
    -------------
    - Run search twice with same seed
    - Results should be identical
    - This verifies reproducibility
    """
    
    print("\n" + "=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)
    
    # Run search twice with same seed
    df1 = run_search(n=5, seed=42)
    df2 = run_search(n=5, seed=42)
    
    # Results should be identical
    # (We compare the parameter columns, not metrics, since metrics depend on solving)
    param_cols = ['span', 'height', 'brace', 'sec_col', 'sec_beam', 'sec_brace', 
                  'udl_w', 'wind_P', 'shipping_limit']
    
    # Reset index for comparison
    df1_params = df1[param_cols].reset_index(drop=True)
    df2_params = df2[param_cols].reset_index(drop=True)
    
    # Should be identical
    pd.testing.assert_frame_equal(df1_params, df2_params, 
                                  "Results should be identical with same seed")
    
    print("✓ Search is reproducible (same seed = same results)")
    print()


if __name__ == "__main__":
    # Run tests manually
    test_search_pipeline_smoke()
    test_search_pipeline_reproducibility()
    print("\n✅ All smoke tests passed!")


