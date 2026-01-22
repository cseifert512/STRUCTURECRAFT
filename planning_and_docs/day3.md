# Day 3: Design Space Exploration

## Overview

Day 3 transformed the structural analysis solver from a **single-case analysis tool** into a **design exploration system**. Instead of analyzing one design at a time, it's now possible to generate hundreds of design variants, evaluate them all, and identify the optimal trade-offs between competing objectives (cost vs. performance).

## What We Built

### The Big Picture

**Before Day 3**: 
- Analyze one portal frame design manually
- Hardcode geometry, sections, and loads
- Get one answer: "This design has drift X and volume Y"

**After Day 3**:
- Generate 500+ design variants automatically
- Evaluate all of them in a batch
- Identify the **Pareto frontier** - the set of optimal designs
- Visualize trade-offs: "If I want lower drift, I need more material"
- Export results for further analysis (Day 4: ML training)

### Key Achievement

We built a **parametric design exploration pipeline** that:
1. Generates design variants (different geometries, sections, loads)
2. Evaluates each variant (solves structure, extracts metrics)
3. Identifies optimal designs (Pareto frontier)
4. Visualizes results (plots, summaries)
5. Exports data (CSV for analysis, plots for communication)

This is the foundation for **optimization** - instead of guessing what design is best, we systematically explore the design space and find the optimal trade-offs.

---

## Engineering Context: Why This Matters

### The Real-World Problem

In structural engineering, there's always a balance between competing objectives:

- **Cost/Material**: Use less material = lower cost, lower carbon footprint
- **Performance/Stiffness**: Use more material = stiffer structure, less deflection

**The Challenge**: Both objectives can't be optimized simultaneously! More material = better performance, but higher cost. Less material = lower cost, but worse performance.

**The Solution**: Find the **Pareto frontier** - the set of designs where improving one objective necessarily makes the other worse. These are the only designs worth considering.

### Example

Imagine designing a portal frame for a warehouse:

- **Design A**: Volume = 0.5 m³, Drift = 10 mm
- **Design B**: Volume = 0.6 m³, Drift = 10 mm  → **Dominated** (same drift, more material)
- **Design C**: Volume = 0.5 m³, Drift = 15 mm  → **Dominated** (same material, more drift)
- **Design D**: Volume = 0.7 m³, Drift = 5 mm   → **Pareto-optimal** (better drift, worse volume)

Designs A and D are both Pareto-optimal - they represent different trade-offs. The engineer chooses based on priorities:
- "I care more about cost" → Choose Design A
- "I care more about stiffness" → Choose Design D

### Fabrication Constraints

Real designs must satisfy **fabrication constraints**:
- **Shipping limit**: Members can't exceed 12 m (40 ft) - can't ship longer pieces
- **Section variety**: Too many unique sections = expensive fabrication
- **Brace angle**: Shallow braces (< 25°) are ineffective and hard to construct

Our system enforces these constraints during variant generation, ensuring all designs are **fabrication-feasible**.

---

## What We Implemented

### 1. Material and Section Catalog (`catalog.py`)

**Why?** Instead of hardcoding section properties (A, I, S) in every demo, we created a reusable catalog.

**What?**
- `Material` dataclass: E (stiffness), density, carbon_factor
- `Section` dataclass: A (area), I (moment of inertia), S (section modulus)
- `TIMBER_SECTIONS`: 12 predefined timber sections (2x4, 2x6, 2x8, ..., GL6x12)
- `DEFAULT_MATERIAL`: Douglas Fir (common structural timber)

**Engineering Impact:**
- **Volume calculation**: `volume = Σ(A × L)` for all elements
- **Carbon calculation**: `carbon = volume × carbon_factor`
- **Design flexibility**: Easy to try different section combinations
- **ML features**: Section indices (0-11) become features for Day 4

**Example:**
```python
from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL

sec = TIMBER_SECTIONS[3]  # 2x10 section
volume = sec.A * length  # Material volume
carbon = volume * DEFAULT_MATERIAL.carbon_factor  # Embodied carbon
```

---

### 2. Parametric Portal Generator (`explore.py` - Part 1)

**Why?** Turn the portal frame into a **function** - input parameters, get a complete model.

**What?**
- `PortalParams` dataclass: All design parameters (span, height, brace, sections, loads)
- `make_portal()` function: Converts parameters → complete model (nodes, elements, loads, constraints)

**Engineering Process:**
1. **Input**: Parameters (span=6m, height=3m, sec_col=3, sec_beam=4, ...)
2. **Output**: Complete analysis-ready model:
   - Nodes at key points (base, top, corners)
   - Elements with correct section properties
   - Boundary conditions (pinned bases)
   - Loads (UDL on beam, lateral at top)

**Why This Matters:**
- **Reproducibility**: Same parameters = same model
- **Automation**: Can generate models programmatically
- **Exploration**: Easy to try different combinations
- **ML features**: Parameters become features for Day 4

**Example:**
```python
params = PortalParams(
    span=6.0, height=3.0, brace=0,
    sec_col=3, sec_beam=4, sec_brace=1,
    udl_w=-2000.0, wind_P=5000.0
)
nodes, elements, fixed, udls, loads = make_portal(params, material, sections)
# Now ready for analysis!
```

---

### 3. Variant Sampling (`explore.py` - Part 2)

**Why?** Generate many design variants to explore the design space.

**What?**
- `sample_params()` function: Generates random design variants
- `check_constraints()` function: Validates fabrication constraints

**Sampling Strategy:**
- Random uniform sampling (Day 3)
- Parameter ranges: span (4-12m), height (2.5-6m), sections (0-11), loads
- Constraint checking: shipping limit, section variety, brace angle

**Engineering Impact:**
- **Broad exploration**: Samples entire design space
- **Constraint enforcement**: Only generates feasible designs
- **Reproducibility**: Same seed = same sequence of variants

**Example:**
```python
rng = np.random.default_rng(seed=42)
variants = sample_params(rng, TIMBER_SECTIONS, n=500)
# Generates 500 valid design variants
```

---

### 4. Batch Evaluation (`explore.py` - Part 3)

**Why?** Evaluate hundreds of designs automatically.

**What?**
- `evaluate_variant()` function: Evaluates one design variant
- `run_search()` function: Batch evaluation loop

**Evaluation Pipeline:**
1. Build model from parameters (`make_portal`)
2. Assemble stiffness matrix (`assemble_global_K`)
3. Assemble load vector (`assemble_element_loads_global` + nodal loads)
4. Solve (`solve_linear`)
5. Extract metrics:
   - **Drift**: Max horizontal displacement at top (mm)
   - **Max moment**: Maximum absolute moment across all elements (N·m)
   - **Volume**: Total material volume (m³)
   - **Carbon**: Embodied carbon (kg CO₂)

**Error Handling:**
- Catches `MechanismError` (unstable structures)
- Returns `ok=False` with reason for failures
- Continues evaluation even if some variants fail

**Engineering Impact:**
- **Automation**: No manual analysis needed
- **Comprehensive**: Evaluates entire design space
- **Robust**: Handles failures gracefully
- **Data export**: Results saved to CSV for analysis

**Example:**
```python
result = evaluate_variant(params, material, sections)
# Returns: {'span': 6.0, 'height': 3.0, ..., 'drift': 0.05, 'volume': 0.12, 'ok': True}

df = run_search(n=500, seed=42)
# Returns DataFrame with 500 design results
```

---

### 5. Pareto Frontier Analysis (`pareto.py`)

**Why?** Identify the optimal designs from hundreds of candidates.

**What?**
- `pareto_mask()` function: Identifies Pareto-optimal designs

**Pareto Optimality:**
A design is **Pareto-optimal** if:
- No other design has BOTH lower volume AND lower drift
- These designs represent the best possible trade-offs
- Moving along the frontier: improve one objective, worsen the other

**Algorithm:**
For each design i:
- Check if any other design j dominates it
- j dominates i if: j ≤ i in both objectives AND j < i in at least one
- If no design dominates i, then i is Pareto-optimal

**Engineering Impact:**
- **Decision support**: Only need to consider Pareto designs
- **Trade-off visualization**: Frontier shows optimal curve
- **Multi-objective optimization**: Foundation for optimization algorithms

**Example:**
```python
mask = pareto_mask(df, x="volume", y="drift")
pareto_designs = df[mask]  # Only Pareto-optimal designs
```

---

### 6. Visualization (`viz.py`)

**Why?** Visualize results to understand trade-offs and structural behavior.

**What?**
- `plot_pareto()`: Scatter plot showing Pareto frontier
- `plot_frame_deformed()`: Deformed shape visualization

**Pareto Plot:**
- X-axis: Volume (material/cost)
- Y-axis: Drift (stiffness/performance)
- Gray points: Dominated designs
- Red points: Pareto-optimal designs
- Shows the "efficient frontier" clearly

**Deformed Shape Plot:**
- Blue: Undeformed structure
- Red: Deformed structure (exaggerated for visibility)
- Shows how structure actually moves under load
- Helps verify results make physical sense

**Engineering Impact:**
- **Communication**: Plots are essential for reports and presentations
- **Understanding**: Visual representation makes trade-offs obvious
- **Verification**: Deformed shapes help verify results

**Example:**
```python
plot_pareto(df, mask, "artifacts/pareto.png")
plot_frame_deformed(nodes, elements, d, "artifacts/frame.png")
```

---

### 7. Complete Demo Script (`run_portal_search.py`)

**Why?** Tie everything together into a complete workflow.

**What?**
Complete 8-step workflow:
1. Parse command-line arguments (`--n`, `--seed`)
2. Run design search (`run_search`)
3. Save results to CSV (`artifacts/results.csv`)
4. Identify Pareto frontier (`pareto_mask`)
5. Create Pareto plot (`plot_pareto`)
6. Create top 10 summary (`artifacts/top10.md`)
7. Visualize winner design (`plot_frame_deformed`)
8. Print summary

**Deliverables:**
- `artifacts/results.csv`: All design results (for analysis)
- `artifacts/pareto.png`: Pareto frontier plot (for visualization)
- `artifacts/top10.md`: Top 10 designs summary (for decision-making)
- `artifacts/winner_frame.png`: Deformed shape (for verification)

**Usage:**
```bash
python demos/run_portal_search.py --n 500 --seed 42
```

---

## Technical Architecture

### Data Flow

```
Parameters → Model → Solve → Metrics → Results
    ↓          ↓        ↓        ↓         ↓
PortalParams → make_portal() → solve_linear() → evaluate_variant() → DataFrame
```

### Key Design Decisions

1. **Separation of Concerns**:
   - `make_portal()`: Model generation only
   - `evaluate_variant()`: Full evaluation pipeline
   - `run_search()`: Batch orchestration
   - Each function has a single, clear responsibility

2. **Error Handling**:
   - Graceful failure: Failed designs return `ok=False` with reason
   - Continues evaluation even if some variants fail
   - All results saved (even failures) for analysis

3. **Reproducibility**:
   - Random seed ensures same sequence of variants
   - Same seed = same results (critical for Day 4 ML training)

4. **Extensibility**:
   - Easy to add new parameters (just update `PortalParams`)
   - Easy to add new metrics (just update `evaluate_variant`)
   - Easy to add new constraints (just update `check_constraints`)

---

## Engineering Validation

### What We Verified

1. **Model Generation**:
   - ✅ `make_portal()` generates correct structures
   - ✅ Nodes, elements, loads, constraints all correct
   - ✅ Works for both braced and unbraced frames

2. **Evaluation Pipeline**:
   - ✅ All successful designs have valid metrics
   - ✅ Failed designs handled gracefully
   - ✅ Metrics match expected ranges

3. **Pareto Analysis**:
   - ✅ Correctly identifies non-dominated designs
   - ✅ Frontier shows clear trade-off curve
   - ✅ Results make engineering sense

4. **Visualization**:
   - ✅ Plots are clear and informative
   - ✅ Deformed shapes match expected behavior
   - ✅ Pareto frontier is visible

### Test Coverage

- ✅ `test_explore.py`: Tests model generation
- ✅ `test_search_pipeline_smoke.py`: Tests full pipeline
- ✅ All tests pass

---

## Results and Insights

### What We Learned

1. **Design Space Exploration**:
   - Hundreds of designs reveal patterns
   - Pareto frontier shows clear trade-offs
   - Optimal designs cluster around frontier

2. **Constraint Impact**:
   - Shipping limit rejects some large designs
   - Section variety constraint encourages standardization
   - Brace angle constraint ensures effectiveness

3. **Performance Metrics**:
   - Drift ranges from ~5mm to ~500mm (depending on design)
   - Volume ranges from ~0.05 m³ to ~0.5 m³
   - Clear correlation: more volume → less drift

### Example Results

From a 500-design search:
- **Total designs**: 500
- **Successful**: ~480 (96%)
- **Pareto-optimal**: ~20-30 (4-6%)
- **Best volume**: ~0.05 m³ (very light, but high drift)
- **Best drift**: ~5 mm (very stiff, but high volume)

The Pareto frontier shows: **Both low volume AND low drift can't be achieved simultaneously - the trade-off must be chosen.**

---

## Connection to Day 4

Day 3 sets up Day 4 perfectly:

1. **Data**: `results.csv` contains all design results
2. **Features**: Parameters (span, height, sections) become ML features
3. **Targets**: Metrics (drift, volume) become ML targets
4. **Pipeline**: Evaluation pipeline is ready for ML-guided search

**Day 4 will**:
- Train ML surrogate on `results.csv`
- Use surrogate to pre-screen thousands of candidates
- Re-evaluate best candidates with true solver
- Show improvement vs. random sampling

---

## Files Created/Modified

### New Files
- `mini_branch/catalog.py`: Material and section catalog
- `mini_branch/pareto.py`: Pareto frontier analysis
- `mini_branch/viz.py`: Visualization functions
- `demos/run_portal_search.py`: Complete demo script
- `tests/test_search_pipeline_smoke.py`: Pipeline smoke test

### Modified Files
- `mini_branch/explore.py`: Added `evaluate_variant()` and `run_search()`
- `pyproject.toml`: Added `pandas` dependency

---

## Key Takeaways

1. **From Single-Case to Exploration**: We transformed a single-case solver into a design exploration system.

2. **Pareto Optimality**: The concept of Pareto-optimal designs is fundamental to multi-objective optimization.

3. **Fabrication Constraints**: Real designs must satisfy practical constraints (shipping, fabrication, construction).

4. **Visualization**: Plots are essential for understanding trade-offs and communicating results.

5. **Data Export**: Clean CSV export enables further analysis (Day 4: ML training).

6. **Reproducibility**: Random seeds ensure reproducible results (critical for scientific work).

---

## Next Steps (Day 4)

Day 4 will build on Day 3:
- **ML Surrogate**: Train model to predict drift/volume from parameters
- **Guided Search**: Use surrogate to find better designs more efficiently
- **Comparison**: Show improvement vs. random sampling

The foundation is solid - Day 3's clean data pipeline and evaluation system make Day 4's ML integration straightforward.

---

**Status**: ✅ Day 3 Complete - Design Space Exploration, Pareto Analysis, and Visualization implemented and validated
