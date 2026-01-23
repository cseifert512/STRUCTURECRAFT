"""
Train Surrogate Model for Portal Frame Drift Prediction
========================================================

This script trains a RandomForest regressor to predict drift from
design parameters. The model can then screen thousands of candidates
cheaply before running the expensive FEA solver.

Philosophy: "Model proposes; physics verifies."
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Import our ML utils
from _ml_utils import load_clean_df, make_features, make_target


def main():
    print("=" * 70)
    print("SURROGATE MODEL TRAINING")
    print("=" * 70)
    print()
    
    # =========================================================================
    # STEP 1: Load and prepare data
    # =========================================================================
    print("Step 1: Loading data...")
    df = load_clean_df("artifacts/results.csv")
    
    X, feature_names = make_features(df)
    y = make_target(df, target="drift")
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Features: {feature_names}")
    print()
    
    # =========================================================================
    # STEP 2: Train/test split
    # =========================================================================
    print("Step 2: Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print()
    
    # =========================================================================
    # STEP 3: Train model
    # =========================================================================
    print("Step 3: Training RandomForest...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete!")
    print()
    
    # =========================================================================
    # STEP 4: Evaluate on test set
    # =========================================================================
    print("Step 4: Evaluating on test set...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"  Train MAE: {mae_train:.6f} m ({mae_train*1000:.3f} mm)")
    print(f"  Test MAE:  {mae_test:.6f} m ({mae_test*1000:.3f} mm)")
    print(f"  Train R²:  {r2_train:.4f}")
    print(f"  Test R²:   {r2_test:.4f}")
    print()
    
    # =========================================================================
    # STEP 5: Save model
    # =========================================================================
    print("Step 5: Saving model...")
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")
    print()
    
    # =========================================================================
    # STEP 6: Create pred vs actual plot
    # =========================================================================
    print("Step 6: Creating pred_vs_actual.png...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot test set points
    ax.scatter(y_test * 1000, y_pred_test * 1000, 
               alpha=0.7, s=60, c='#2563eb', label='Test set')
    
    # Perfect prediction line
    lims = [0, max(y.max() * 1000 * 1.1, 10)]
    ax.plot(lims, lims, 'k--', lw=2, alpha=0.7, label='Perfect prediction')
    
    ax.set_xlabel('Actual Drift (mm)', fontsize=12)
    ax.set_ylabel('Predicted Drift (mm)', fontsize=12)
    ax.set_title(f'Surrogate Model: Predicted vs Actual Drift\n'
                 f'Test MAE = {mae_test*1000:.3f} mm, R² = {r2_test:.3f}', 
                 fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pred_path = "artifacts/pred_vs_actual.png"
    plt.savefig(pred_path, dpi=150)
    plt.close()
    print(f"  Saved to: {pred_path}")
    print()
    
    # =========================================================================
    # STEP 7: Create feature importance plot
    # =========================================================================
    print("Step 7: Creating feature_importance.png...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(feature_names)))
    bars = ax.barh(range(len(feature_names)), 
                   importances[indices], 
                   color=colors[::-1])
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('RandomForest Feature Importances for Drift Prediction', fontsize=14)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importances[indices])):
        ax.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    imp_path = "artifacts/feature_importance.png"
    plt.savefig(imp_path, dpi=150)
    plt.close()
    print(f"  Saved to: {imp_path}")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Artifacts generated:")
    print(f"  - {model_path}")
    print(f"  - {pred_path}")
    print(f"  - {imp_path}")
    print()
    print(f"Model performance:")
    print(f"  - The surrogate predicts drift with MAE ≈ {mae_test*1000:.2f} mm")
    print(f"  - R² ≈ {r2_test:.3f} on held-out data")
    print()
    print("Next step: Run guided_search.py to use this model!")


if __name__ == "__main__":
    main()