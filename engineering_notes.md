# Engineering Notes

Technical documentation for the mini_branch structural analysis framework.

---

## Table of Contents

1. [Assumptions & Simplifications](#assumptions--simplifications)
2. [Validation Summary](#validation-summary)
3. [UDL Implementation](#udl-implementation)
4. [Mechanism Detection](#mechanism-detection)
5. [ML Surrogate Model](#ml-surrogate-model)
6. [Limitations](#limitations)
7. [Next Steps](#next-steps)

---

## Assumptions & Simplifications

### Analysis Type
- **2D frame analysis** (plane stress/strain)
- **Linear elastic** material behavior
- **Small deflection** theory (geometric linearity)
- No second-order (P-Δ) effects

### Element Formulation
- **Euler-Bernoulli beam theory** (no shear deformation)
- Constant cross-section along element length
- Axial-bending coupling through transformation matrices

### Material Model
- Isotropic, homogeneous material
- Single Young's modulus (E) for all elements
- No yield/failure criteria implemented

### Boundary Conditions
- Pinned supports: restrain ux, uy (allow rotation)
- Fixed supports: restrain ux, uy, θz
- No partial fixity or spring supports

### Loading
- Point loads at nodes
- Uniformly distributed loads (UDL) on elements
- No thermal, settlement, or dynamic loads

---

## Validation Summary

### Day 1: Basic Elements

| Test Case | Analytical Formula | FEM Result | Error |
|-----------|-------------------|------------|-------|
| Cantilever tip deflection | δ = PL³/(3EI) | Matches | < 1e-10 |
| Cantilever tip rotation | θ = PL²/(2EI) | Matches | < 1e-10 |
| Simply supported midspan | δ = PL³/(48EI) | Matches | < 1e-10 |
| Reaction equilibrium | ΣF = 0 | Verified | Exact |

### Day 2: UDL and Portal Frames

| Test Case | Analytical Formula | FEM Result | Error |
|-----------|-------------------|------------|-------|
| SS beam UDL deflection | δ = 5wL⁴/(384EI) | Matches | < 1e-6 |
| SS beam UDL reactions | R = wL/2 | Matches | Exact |
| Portal frame equilibrium | ΣF = 0, ΣM = 0 | Verified | < 1e-10 |

### Day 3: Design Exploration

| Metric | Validation |
|--------|------------|
| Pareto dominance | Verified by definition |
| Volume calculation | Σ(A × L) for all elements |
| Carbon calculation | Volume × carbon_factor |

---

## UDL Implementation

### Equivalent Nodal Loads

For a uniformly distributed load w (N/m) on an element of length L:

```
Fixed-end forces (local coordinates):
    Fy_i = wL/2      (shear at node i)
    Fy_j = wL/2      (shear at node j)
    Mz_i = wL²/12    (moment at node i)
    Mz_j = -wL²/12   (moment at node j)
```

### Sign Convention
- Positive w: load in local +y direction
- For gravity loads: w is negative (downward)
- Moments follow right-hand rule

### Transformation
Equivalent nodal loads are transformed to global coordinates using:
```
F_global = T^T × F_local
```
Where T is the same transformation matrix used for stiffness.

---

## Mechanism Detection

### What is a Mechanism?
A mechanism occurs when the structure has insufficient restraints to prevent rigid body motion. Mathematically, the reduced stiffness matrix becomes singular.

### Detection Method
```python
# In solve.py
cond = np.linalg.cond(K_ff)
if cond > 1e12:
    raise MechanismError("Structure is unstable")
```

### Common Causes
1. **Insufficient supports**: Missing reaction DOFs
2. **Collinear elements**: All elements in a straight line with no rotational restraint
3. **Disconnected nodes**: Elements not properly connected

### Portal Frame Stability
Portal frames with pinned bases require:
- At least 4 DOFs restrained (ux, uy at each base)
- Connected elements forming a closed loop
- Or: bracing elements to provide lateral stability

---

## ML Surrogate Model

### Purpose
The surrogate model provides fast approximate predictions of drift, enabling:
- Screening thousands of candidate designs
- Identifying promising regions of design space
- Reducing computational cost of optimization

### Philosophy: "Model Proposes; Physics Verifies"
The ML model is NEVER used for final design decisions. Its role is to:
1. **Propose** candidates that might be good
2. **Filter** obviously bad designs
3. **Accelerate** exploration

The physics solver ALWAYS verifies the final results.

### Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | RandomForestRegressor |
| n_estimators | 300 |
| Training split | 80% train / 20% test |
| Target | drift (meters) |

### Features (8 inputs)

| Feature | Description | Range |
|---------|-------------|-------|
| span | Bay width | 4-12 m |
| height | Column height | 2.5-6 m |
| brace | Bracing config | 0 (none) or 1 (X-brace) |
| sec_col | Column section index | 0-10 |
| sec_beam | Beam section index | 0-10 |
| sec_brace | Brace section index | 0-10 |
| udl_w | Gravity load | -5000 to -1000 N/m |
| wind_P | Lateral load | 2000-10000 N |

### Performance (500 training samples)

| Metric | Train | Test |
|--------|-------|------|
| MAE | 125 mm | 346 mm |
| R² | 0.95 | 0.69 |

### Feature Importance (typical)

1. **sec_col** (~0.21): Column size is most influential
2. **height** (~0.21): Taller frames drift more
3. **udl_w** (~0.16): Heavier loads increase stress
4. **span** (~0.11): Wider spans increase bending
5. **sec_beam** (~0.11): Beam size matters moderately
6. **brace** (~0.08): Bracing helps but less than expected
7. **wind_P** (~0.08): Lateral load magnitude
8. **sec_brace** (~0.04): Brace size matters least

### Guided Search Workflow

```
1. Generate N_CANDIDATES = 5000 random designs (no solver)
2. Score all with surrogate model (instant)
3. Select top K = 100 by predicted drift
4. Verify those K with physics solver (expensive but targeted)
5. Compare against random baseline
```

### When NOT to Trust the Surrogate

- **Extrapolation**: Parameters outside training range
- **Edge cases**: Very small or very large structures
- **Novel topologies**: The model only knows portal frames
- **Final design**: Always verify with physics solver

---

## Limitations

### Current Implementation
1. **2D only**: No out-of-plane behavior
2. **Linear elastic**: No plasticity or buckling
3. **No code compliance**: Results are not code-checked
4. **Single material**: No composite sections
5. **Static loads only**: No dynamic/seismic analysis

### ML Model Limitations
1. **Interpolation only**: Cannot extrapolate beyond training data
2. **Portal frames only**: Not generalizable to other topologies
3. **Single objective**: Trained on drift; doesn't predict volume
4. **No uncertainty**: Provides point estimates, not confidence intervals

### Numerical Precision
- Matrix operations use numpy float64
- Condition number check prevents ill-conditioning
- Tolerance for equilibrium checks: 1e-6

---

## Next Steps

### Planned Enhancements
- [ ] Multi-story frames
- [ ] Dynamic analysis (modal, time-history)
- [ ] Geometric nonlinearity (P-Δ effects)
- [ ] Material nonlinearity (plastic hinges)
- [ ] Code compliance (AISC/NDS checks)
- [ ] 3D frame analysis
- [ ] CAD export (DXF, IFC)

### ML Improvements
- [ ] Multi-output model (predict drift + volume + carbon)
- [ ] Uncertainty quantification (ensemble variance)
- [ ] Active learning (sample where model is uncertain)
- [ ] Neural network surrogate (for larger design spaces)

---

## References

1. **Matrix Structural Analysis**: McGuire, Gallagher, Ziemian
2. **Finite Element Method**: Bathe, K.J.
3. **Timber Design**: NDS (National Design Specification)
4. **Multi-Objective Optimization**: Deb, K. (NSGA-II)
5. **Surrogate Modeling**: Forrester, Sóbester, Keane

---

*Last updated: Day 4 - ML-guided design exploration*





