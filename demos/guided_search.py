"""
ML-GUIDED DESIGN SEARCH
========================

This script demonstrates how the surrogate model accelerates design exploration.
Instead of randomly sampling and evaluating everything with the expensive solver,
we use the ML model to screen candidates cheaply, then verify only the promising ones.

Philosophy: "Model proposes; physics verifies."
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _ml_utils import load_clean_df, make_features, params_to_features
from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL
from mini_branch.explore import PortalParams, sample_params, evaluate_variant, check_constraints


def main():
    print("=" * 70)
    print("ML-GUIDED DESIGN SEARCH")
    print("=" * 70)
    print()
    
    # =========================================================================
    # STEP 1: Load trained surrogate model
    # =========================================================================
    print("Step 1: Loading surrogate model...")
    model_path = "artifacts/model.joblib"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Run train_surrogate.py first!")
        return
    
    model = joblib.load(model_path)
    print(f"  Loaded model from {model_path}")
    print()
    
    # =========================================================================
    # STEP 2: Generate candidate pool (cheap - no solver)
    # =========================================================================
    print("Step 2: Generating candidate pool...")
    N_CANDIDATES = 5000
    rng = np.random.default_rng(seed=123)
    
    candidates = sample_params(rng, TIMBER_SECTIONS, N_CANDIDATES)
    print(f"  Generated {len(candidates)} valid candidates")
    print()
    
    # =========================================================================
    # STEP 3: Score candidates with surrogate (instant)
    # =========================================================================
    print("Step 3: Scoring candidates with surrogate...")
    
    # Convert PortalParams to feature matrix
    candidate_dicts = []
    for p in candidates:
        candidate_dicts.append({
            'span': p.span,
            'height': p.height,
            'brace': p.brace,
            'sec_col': p.sec_col,
            'sec_beam': p.sec_beam,
            'sec_brace': p.sec_brace,
            'udl_w': p.udl_w,
            'wind_P': p.wind_P,
        })
    
    X_candidates = params_to_features(candidate_dicts)
    predicted_drift = model.predict(X_candidates)
    
    print(f"  Predicted drift range: {predicted_drift.min()*1000:.1f} - {predicted_drift.max()*1000:.1f} mm")
    print()
    
    # =========================================================================
    # STEP 4: Select top K by predicted drift
    # =========================================================================
    print("Step 4: Selecting top candidates...")
    K = 100  # Number of candidates to verify with real solver
    
    # Get indices of K lowest predicted drift
    top_indices = np.argsort(predicted_drift)[:K]
    top_candidates = [candidates[i] for i in top_indices]
    top_predicted = predicted_drift[top_indices]
    
    print(f"  Selected top {K} candidates with predicted drift < {top_predicted.max()*1000:.1f} mm")
    print()
    
    # =========================================================================
    # STEP 5: Verify top candidates with real solver
    # =========================================================================
    print("Step 5: Verifying top candidates with FEA solver...")
    
    guided_results = []
    for i, params in enumerate(top_candidates):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{K}")
        result = evaluate_variant(params, DEFAULT_MATERIAL, TIMBER_SECTIONS)
        result['predicted_drift'] = top_predicted[i]
        guided_results.append(result)
    
    guided_df = pd.DataFrame(guided_results)
    guided_df = guided_df[guided_df['ok'] == True]
    print(f"  Verified {len(guided_df)} successful designs")
    print()
    
    # =========================================================================
    # STEP 6: Random baseline (for comparison)
    # =========================================================================
    print("Step 6: Running random baseline for comparison...")
    
    rng_random = np.random.default_rng(seed=456)
    random_candidates = sample_params(rng_random, TIMBER_SECTIONS, K)
    
    random_results = []
    for i, params in enumerate(random_candidates):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{K}")
        result = evaluate_variant(params, DEFAULT_MATERIAL, TIMBER_SECTIONS)
        random_results.append(result)
    
    random_df = pd.DataFrame(random_results)
    random_df = random_df[random_df['ok'] == True]
    print(f"  Evaluated {len(random_df)} random designs")
    print()
    
    # =========================================================================
    # STEP 7: Compare results
    # =========================================================================
    print("Step 7: Comparing guided vs random...")
    print()
    
    print("  GUIDED SEARCH (ML-screened):")
    print(f"    Best drift:  {guided_df['drift'].min()*1000:.2f} mm")
    print(f"    Mean drift:  {guided_df['drift'].mean()*1000:.2f} mm")
    print(f"    Best volume: {guided_df['volume'].min():.4f} m³")
    print()
    
    print("  RANDOM SEARCH (baseline):")
    print(f"    Best drift:  {random_df['drift'].min()*1000:.2f} mm")
    print(f"    Mean drift:  {random_df['drift'].mean()*1000:.2f} mm")
    print(f"    Best volume: {random_df['volume'].min():.4f} m³")
    print()
    
    improvement = (random_df['drift'].mean() - guided_df['drift'].mean()) / random_df['drift'].mean() * 100
    print(f"  Guided search found designs with {improvement:.1f}% lower mean drift!")
    print()
    
    # =========================================================================
    # STEP 8: Save results
    # =========================================================================
    print("Step 8: Saving results...")
    os.makedirs("artifacts", exist_ok=True)
    
    guided_df.to_csv("artifacts/guided_results.csv", index=False)
    random_df.to_csv("artifacts/random_results.csv", index=False)
    print("  Saved artifacts/guided_results.csv")
    print("  Saved artifacts/random_results.csv")
    print()
    
    # =========================================================================
    # STEP 9: Create comparison plot
    # =========================================================================
    print("Step 9: Creating guided_vs_random.png...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot random baseline
    ax.scatter(random_df['volume'], random_df['drift'] * 1000,
               alpha=0.5, s=40, c='gray', label=f'Random ({len(random_df)} designs)')
    
    # Plot guided results
    ax.scatter(guided_df['volume'], guided_df['drift'] * 1000,
               alpha=0.7, s=60, c='#2563eb', label=f'ML-Guided ({len(guided_df)} designs)')
    
    # Highlight best 10 from guided
    best_guided = guided_df.nsmallest(10, 'drift')
    ax.scatter(best_guided['volume'], best_guided['drift'] * 1000,
               s=120, c='#dc2626', marker='*', label='Best 10 (guided)', zorder=5)
    
    ax.set_xlabel('Volume (m³)', fontsize=12)
    ax.set_ylabel('Drift (mm)', fontsize=12)
    ax.set_title('ML-Guided vs Random Search\n'
                 f'Guided mean drift: {guided_df["drift"].mean()*1000:.1f}mm vs '
                 f'Random: {random_df["drift"].mean()*1000:.1f}mm',
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "artifacts/guided_vs_random.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved {plot_path}")
    print()
    
    # =========================================================================
    # STEP 10: Print explanation card for best design
    # =========================================================================
    print("=" * 70)
    print("BEST DESIGN (ML-GUIDED)")
    print("=" * 70)
    
    best = guided_df.loc[guided_df['drift'].idxmin()]
    
    print(f"\nParameters:")
    print(f"  Span:       {best['span']:.2f} m")
    print(f"  Height:     {best['height']:.2f} m")
    print(f"  Bracing:    {'X-brace' if best['brace'] == 1 else 'None'}")
    print(f"  Column:     {TIMBER_SECTIONS[int(best['sec_col'])].name}")
    print(f"  Beam:       {TIMBER_SECTIONS[int(best['sec_beam'])].name}")
    if best['brace'] == 1:
        print(f"  Brace:      {TIMBER_SECTIONS[int(best['sec_brace'])].name}")
    print(f"  UDL:        {best['udl_w']:.0f} N/m")
    print(f"  Wind:       {best['wind_P']:.0f} N")
    
    print(f"\nPerformance:")
    print(f"  Predicted drift: {best['predicted_drift']*1000:.2f} mm")
    print(f"  Actual drift:    {best['drift']*1000:.2f} mm")
    print(f"  Max moment:      {best['max_abs_M']:.0f} N·m")
    print(f"  Volume:          {best['volume']:.4f} m³")
    print(f"  Carbon:          {best['carbon']:.2f} kg CO₂")
    
    # Feature importances from model
    feature_names = ['span', 'height', 'brace', 'sec_col', 'sec_beam', 'sec_brace', 'udl_w', 'wind_P']
    importances = model.feature_importances_
    top_3_idx = np.argsort(importances)[-3:][::-1]
    
    print(f"\nTop 3 influential features (from model):")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"  {i}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    print()
    print("=" * 70)
    print("GUIDED SEARCH COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


