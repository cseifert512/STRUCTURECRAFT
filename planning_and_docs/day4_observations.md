# Day 4 Observations

## What We Built

Day 4 added an **ML-guided design exploration layer** on top of the validated FEA kernel from Days 1-3.

### Core Components

1. **`demos/_ml_utils.py`** — Data preparation helpers
   - `load_clean_df()`: Loads CSV, filters to ok=True, drops NaNs
   - `make_features()`: Extracts 8 design parameters as feature matrix
   - `make_target()`: Extracts drift (or other metric) as target vector
   - `params_to_features()`: Converts PortalParams to features for scoring

2. **`demos/train_surrogate.py`** — Model training pipeline
   - Loads `artifacts/results.csv` from Day 3
   - 80/20 train/test split
   - RandomForest with 300 trees
   - Outputs: `model.joblib`, `pred_vs_actual.png`, `feature_importance.png`

3. **`demos/guided_search.py`** — The "money demo"
   - Generates 5000 candidate designs (cheap, no solver)
   - Scores all with surrogate model (instant)
   - Selects top 100 by predicted drift
   - Verifies with real FEA solver
   - Compares against random baseline

---

## Key Results

### Training Size Experiment

| Samples | Train MAE | Test MAE | Train R² | Test R² |
|---------|-----------|----------|----------|---------|
| 100     | 228 mm    | 521 mm   | 0.92     | 0.52    |
| 500     | 125 mm    | 346 mm   | 0.95     | **0.69**|
| 1000    | 75 mm     | 347 mm   | 0.97     | 0.54    |

**Observation**: 500 samples was the sweet spot. With 1000 samples, the model **overfit** — Train R² increased but Test R² dropped. The gap between train and test performance widened.

**Lesson**: More data isn't always better. Need to either:
- Regularize the model (limit depth, fewer trees)
- Use cross-validation to tune hyperparameters
- Try simpler models that don't overfit as easily

### Guided vs Random Search

| Metric | Guided | Random |
|--------|--------|--------|
| Mean drift | 0.18 mm | 441.8 mm |
| Best drift | 0.09 mm | 0.09 mm |
| Best volume | 0.26 m³ | 0.07 m³ |

**Observation**: Guided search found designs with **~100% lower mean drift**. However, the guided designs use more material (higher volume) because the model learned "bigger sections = less drift" and optimized purely for drift.

**Insight**: The model optimizes for what you train it on. To find balanced designs (low drift AND low volume), would need a multi-objective surrogate or composite objective.

### Feature Importance (Consistent Across Runs)

1. **sec_col** (~0.21) — Column section size dominates
2. **height** (~0.21) — Taller frames drift more
3. **udl_w** (~0.16) — Gravity loads affect stress distribution
4. **span** (~0.11) — Wider spans mean more bending
5. **sec_beam** (~0.11) — Beam size has moderate effect
6. **brace** (~0.08) — Bracing helps less than expected
7. **wind_P** (~0.08) — Lateral load magnitude
8. **sec_brace** (~0.04) — Brace size matters least

**Engineering Insight**: The model confirms physical intuition — column stiffness and frame height are the dominant factors for lateral drift. Bracing (brace=1) helps but isn't as impactful as having stiffer columns.

---

## What Worked Well

1. **Clean separation of concerns**
   - Day 3 generates data, Day 4 consumes it
   - ML layer never touches the solver
   - Philosophy: "Model proposes; physics verifies"

2. **Fast iteration**
   - Training takes ~2 seconds on 500 samples
   - Can experiment with different features/models quickly
   - Guided search runs in <30 seconds total

3. **Interpretable model**
   - RandomForest provides feature importances
   - Can explain "why" the model makes predictions
   - Engineers can sanity-check the rankings

4. **Visualization pipeline**
   - `pred_vs_actual.png` — Trust check for model
   - `feature_importance.png` — Interpretability
   - `guided_vs_random.png` — Value demonstration

---

## What Could Be Improved

1. **Overfitting with more data**
   - Would benefit from cross-validation
   - Could try gradient boosting (better regularization)
   - Could limit tree depth or min samples per leaf

2. **Single-objective optimization**
   - Currently only optimizes for drift
   - Pareto-optimal designs need multi-objective approach
   - Could train separate models for drift, volume, carbon

3. **No uncertainty quantification**
   - Model gives point estimates
   - Don't know confidence intervals
   - Could use ensemble variance or Bayesian approach

4. **Limited to portal frames**
   - Model is topology-specific
   - Can't generalize to other structures
   - Would need retraining for each new topology

---

## Technical Decisions

### Why RandomForest?

- **Simple to implement**: One import, one fit() call
- **No hyperparameter tuning needed**: Works well out of the box
- **Feature importances built-in**: Great for interpretability
- **Handles mixed features**: Categorical (brace) and continuous (span)
- **Fast inference**: Can score 5000 candidates instantly

### Why Not Neural Networks?

- **Overkill for 500-1000 samples**: Would overfit badly
- **Black box**: Hard to interpret
- **Slow training**: Not worth it for small data
- **Day 4 scope**: Needed fast, reliable results

### Why 80/20 Split?

- Standard for small-to-medium datasets
- 20% test set gives reasonable variance estimate
- Could use k-fold CV but adds complexity

---

## Commands to Reproduce

```powershell
# Generate training data (500 designs)
py demos/run_portal_search.py --n 500 --seed 42

# Train surrogate model
py demos/train_surrogate.py

# Run guided search
py demos/guided_search.py

# Run all tests
py -m pytest tests/ -v
```

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `artifacts/model.joblib` | Trained RandomForest model |
| `artifacts/pred_vs_actual.png` | Predicted vs actual scatter plot |
| `artifacts/feature_importance.png` | Feature importance bar chart |
| `artifacts/guided_vs_random.png` | Comparison plot |
| `artifacts/guided_results.csv` | ML-guided search results |
| `artifacts/random_results.csv` | Random baseline results |

---

## Narrative

> "I built and validated a 2D frame analysis kernel, wrapped it in a parametric design explorer with fabrication-aware constraints and Pareto outputs, then trained a surrogate model to guide search—using ML for proposal and the solver for verification. The workflow is reproducible, tested, and produces decision-grade tradeoff visuals."

This is the exact combination of **technical + design + AI + paranoia** that structural engineering teams value.

---

*Day 4 complete. 45 tests passing.*



