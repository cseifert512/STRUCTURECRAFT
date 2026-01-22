Day 3 objectives

By end of Day 3 the following should run:

py .\demos\run_portal_search.py --n 500 --seed 42


…and the outputs are:

artifacts/results.csv (one row per design)

artifacts/pareto.png (volume proxy vs drift)

artifacts/top10.md (the "best" candidates with parameters + metrics)

artifacts/winner_frame.png (deformed shape for a chosen design)

And pytest still passes.

Scope (Day 3)
IN

A parametric portal frame family generator

Batch evaluation loop (hundreds of variants)

Objective metrics (drift, max |M|, volume/carbon proxy)

Fabrication-aware filters/penalties (shipping length, unique sizes)

Pareto frontier identification + plotting

Clean results export (CSV)

OUT

ML surrogate / guided search (Day 4)

Code compliance (AISC/NDS), detailed connection design

Rhino integration

Architecture rule (how new code interacts with old code)

These modules continue to be used:

assemble_global_K

solve_linear

loads.py to build global load vector from UDLs + nodal loads

post.py to compute element end forces (for max moment)

New additions:

catalog.py (sections/material presets)

explore.py (variant generator + evaluator)

pareto.py (non-dominated sorting / frontier)

viz.py (plots)

one demo script that ties it together

The Day 2 portal demo should basically become "the single-case evaluator" used inside the search loop.

Day 3 deliverable design (what the code should look like)
Data flow per variant

params -> (nodes, elements, fixed_dofs, loads) -> K,F -> solve -> d,R -> metrics -> row

This is the pipeline to be reused in Day 4 (ML training uses results.csv).

Day 3 plan (timeboxed, ruthless)
Block A (1–2 hours): Define the param schema + section catalog
Goal

Stop hardcoding numbers in demos. Build a tiny catalog to index into.

Deliverables

Create src/mini_branch/catalog.py:

Material(name, E, density, carbon_factor) (carbon_factor can be a constant placeholder)

Section(name, A, I, S) (S = section modulus; S can be computed assuming depth, but it's fine to store a rough number)

TIMBER_SECTIONS = [...] (6–12 options)

DEFAULT_MATERIAL = ...

Expectation
Members can be referred to by section index, and volume proxy = Σ(A·L) can be computed.

How it connects
Catalog indices become ML features in Day 4.

Block B (2–3 hours): Build a parametric portal generator
Goal

Turn "portal frame" into a function.

Deliverables

Create src/mini_branch/explore.py with:

PortalParams dataclass:

span: float

height: float

brace: int (0 none, 1 X-brace)

sec_col: int, sec_beam: int, sec_brace: int

udl_w: float

wind_P: float

shipping_limit: float = 12.0 (meters) or 40 ft equivalent

make_portal(params, material, sections) -> (nodes, elements, fixed_dofs, element_udl_map, nodal_loads)

Nodes: 0 (0,0), 1 (0,H), 2 (L,H), 3 (L,0)

Elements: columns + beam (+ two braces if brace=1)

Supports: pinned at bases (fix ux, uy at nodes 0 and 3)

Loads:

Beam gets UDL

Top node gets lateral Fx point load

Expectation
Calling make_portal returns a complete analysis-ready model without touching solver code.

How it connects
Day 3 evaluation loop just samples PortalParams and calls make_portal.

Block C (2–3 hours): Implement variant sampling + constraints
Goal

Generate "design variants" like a product would.

Deliverables

In explore.py add:

sample_params(rng, sections, n) -> list[PortalParams]

span range, height range, brace toggle, random section picks

Constraint filters / flags (don't overcomplicate):

max_member_length <= shipping_limit (else reject)

unique_sections_count <= 3 (else penalty or reject)

brace angle check: if brace=1 and angle < 25° flag or reject

Expectation
500 variants can be generated with reproducibility (seed).

How it connects
These constraints become the "fabrication-aware" story to tell Lucas.

Block D (3–5 hours): Batch evaluation + metrics extraction
Goal

Turn each variant into a clean row of metrics.

Deliverables

In explore.py add:

evaluate_variant(params) -> dict

Build model via make_portal

Assemble K

Assemble global F from nodal loads + element UDLs (using loads.py)

Solve (solve_linear)

Compute metrics:

drift = max(|ux_top|) (nodes 1 and 2)

uy_beam_mid (optional: interpolate or just use node 1/2 not mid; simplest: use top nodes vertical disp)

max_abs_M from post.element_end_forces_local across elements

volume = sum(A*L) across elements

carbon = volume * carbon_factor

ok = True/False plus failure reason if MechanismError

run_search(n, seed) -> pandas.DataFrame

Write results to:

artifacts/results.csv

Expectation

Search completes without crashing even if some variants fail.

Failures become rows with ok=False and reason="unstable".

How it connects
Day 4 uses this CSV for surrogate training. Stability handling is a massive "detail-oriented" flex.

Block E (1–2 hours): Pareto frontier + plots
Goal

Produce the visual that sells the whole thing.

Deliverables

Create src/mini_branch/pareto.py:

pareto_mask(df, x="volume", y="drift") -> boolean mask

"non-dominated" = no other design has <=volume and <=drift with one strict

Create src/mini_branch/viz.py:

plot_pareto(df, mask, outpath)

plot_frame_deformed(nodes, elements, d, outpath) (reuse the Day 2 plotting style)

Demo script demos/run_portal_search.py should:

run search

compute pareto

save plot artifacts/pareto.png

write artifacts/top10.md (top by a chosen score or just list Pareto set sorted by volume)

Expectation
A plot appears that visibly shows tradeoffs. This is the "design-space explorer" artifact.

How it connects
Day 4's guided search will try to find "better Pareto points" more efficiently.

Block F (1–2 hours): Minimal tests that maintain integrity
Deliverables

Add tests/test_search_pipeline_smoke.py:

run run_search(n=10, seed=1) and assert:

dataframe has expected columns

at least one ok=True

no NaNs in key metrics for ok rows

Optional: tests/test_pareto.py with a tiny synthetic dataset to confirm the pareto logic.

Expectation
The pipeline doesn't regress when refactoring for Day 4.

End-of-Day-3 deliverables checklist

New files expected:

src/mini_branch/catalog.py

src/mini_branch/explore.py

src/mini_branch/pareto.py

src/mini_branch/viz.py

demos/run_portal_search.py

tests/test_search_pipeline_smoke.py (+ optional pareto test)

artifacts/ folder created at runtime

Outputs produced:

artifacts/results.csv

artifacts/pareto.png

artifacts/top10.md

artifacts/winner_frame.png (from one chosen design)

Narrative that can be claimed after Day 3

"I implemented a validated 2D frame solver, then wrapped it in a parametric design explorer that generates portal frame variants, enforces fabrication-aware constraints (shipping length, part variety, bracing geometry), and outputs a Pareto frontier of stiffness vs material."

That's Branch energy.

How Day 3 sets up Day 4 (direct)

Day 4 is simply:

load results.csv

train surrogate on params → predict drift/volume/utilization

use surrogate to pre-screen thousands of candidates

re-run best ones with true solver

show improvement vs random sampling

So Day 3's "clean results table + stable parameterization" is the whole enabler.

Working order 

catalog.py

PortalParams + make_portal

sample_params

evaluate_variant

write CSV

Pareto + plots

smoke tests

This order keeps progress moving forward without refactoring hell.

---

## File Structure Mapping & Import Dependencies

### Current Day 2 Structure (mini_branch/)
```
mini_branch/
├── __init__.py
├── model.py          # Node, Frame2D dataclasses
├── elements.py       # element_geometry, frame2d_*_stiffness, frame2d_transform
├── assembly.py       # assemble_global_K, dof_index, DOF_PER_NODE
├── solve.py          # solve_linear, MechanismError
├── loads.py          # assemble_element_loads_global, frame2d_equiv_nodal_load_udl
├── post.py           # element_end_forces_local
├── checks.py         # (existing, not used in Day 3)
├── viz.py            # (placeholder: "# matplotlib plots")
└── explore.py        # (placeholder: "# variant generator + objectives + constraints")
```

### Day 3 New Files & Where They Plug In

#### 1. `mini_branch/catalog.py` (NEW)
**Location:** `mini_branch/catalog.py`

**Purpose:** Material and section property definitions

**Contents:**
- `Material` dataclass (name, E, density, carbon_factor)
- `Section` dataclass (name, A, I, S)
- `TIMBER_SECTIONS` list (6-12 predefined sections)
- `DEFAULT_MATERIAL` constant

**Imports:** None (standalone module)

**Exports:** Material, Section, TIMBER_SECTIONS, DEFAULT_MATERIAL

---

#### 2. `mini_branch/explore.py` (EXPAND)
**Location:** `mini_branch/explore.py` (currently placeholder)

**Purpose:** Parametric portal generator, variant sampling, batch evaluation

**Contents:**
- `PortalParams` dataclass
- `make_portal(params, material, sections)` function
- `sample_params(rng, sections, n)` function
- `evaluate_variant(params)` function
- `run_search(n, seed)` function

**Imports:**
```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .catalog import Material, Section, TIMBER_SECTIONS, DEFAULT_MATERIAL
from .model import Node, Frame2D
from .elements import element_geometry
from .assembly import assemble_global_K, dof_index, DOF_PER_NODE
from .loads import assemble_element_loads_global
from .solve import solve_linear, MechanismError
from .post import element_end_forces_local
```

**Exports:** PortalParams, make_portal, sample_params, evaluate_variant, run_search

**Key Functions:**
- `make_portal()` returns: `(nodes, elements, fixed_dofs, element_udl_map, nodal_loads)`
  - `nodes`: dict[int, Node] - portal frame nodes (0,1,2,3)
  - `elements`: list[Frame2D] - columns, beam, optional braces
  - `fixed_dofs`: list[int] - pinned base constraints (ux, uy at nodes 0, 3)
  - `element_udl_map`: dict[int, float] - UDL values by element ID
  - `nodal_loads`: dict[int, np.ndarray] - point loads by node ID (or dict mapping to DOF indices)

- `evaluate_variant()` workflow:
  1. Call `make_portal()` to get model
  2. Call `assemble_global_K(nodes, elements)` 
  3. Initialize `F = np.zeros(ndof)`
  4. Call `assemble_element_loads_global(nodes, elements, element_udl_map)` and add to F
  5. Add nodal loads: `F[dof_index(node_id, dof)] += load_value`
  6. Call `solve_linear(K, F, fixed_dofs)` → `d, R, free`
  7. Extract metrics from `d` and `R`
  8. Call `element_end_forces_local()` for each element to get moments
  9. Return dict with metrics

---

#### 3. `mini_branch/pareto.py` (NEW)
**Location:** `mini_branch/pareto.py`

**Purpose:** Pareto frontier identification

**Contents:**
- `pareto_mask(df, x="volume", y="drift")` function

**Imports:**
```python
import numpy as np
import pandas as pd
```

**Exports:** pareto_mask

**Function Logic:**
- Takes DataFrame with columns `x` and `y` (default: "volume", "drift")
- Returns boolean mask: True for non-dominated points
- Non-dominated = no other point has `<= x` AND `<= y` with at least one strict inequality

---

#### 4. `mini_branch/viz.py` (EXPAND)
**Location:** `mini_branch/viz.py` (currently placeholder)

**Purpose:** Plotting functions for results visualization

**Contents:**
- `plot_pareto(df, mask, outpath)` function
- `plot_frame_deformed(nodes, elements, d, outpath)` function

**Imports:**
```python
import numpy as np
import matplotlib.pyplot as plt
from .model import Node, Frame2D
from .elements import element_geometry
from .assembly import DOF_PER_NODE, dof_index
```

**Exports:** plot_pareto, plot_frame_deformed

**Function Details:**
- `plot_pareto()`: Scatter plot with Pareto points highlighted, axes labeled
- `plot_frame_deformed()`: Similar to Day 2 portal frame plotting (undeformed + deformed overlay)

---

#### 5. `demos/run_portal_search.py` (NEW)
**Location:** `demos/run_portal_search.py`

**Purpose:** Main demo script that orchestrates the search

**Imports:**
```python
import argparse
import os
import pandas as pd
from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL
from mini_branch.explore import run_search
from mini_branch.pareto import pareto_mask
from mini_branch.viz import plot_pareto, plot_frame_deformed
from mini_branch.model import Node, Frame2D
from mini_branch.explore import make_portal, evaluate_variant
from mini_branch.assembly import assemble_global_K
from mini_branch.loads import assemble_element_loads_global
from mini_branch.solve import solve_linear
```

**Workflow:**
1. Parse CLI args (`--n`, `--seed`)
2. Call `run_search(n, seed)` → DataFrame
3. Save `artifacts/results.csv`
4. Call `pareto_mask(df)` → boolean mask
5. Call `plot_pareto(df, mask, "artifacts/pareto.png")`
6. Filter top 10 from Pareto set, write `artifacts/top10.md`
7. Pick one "winner" design, re-evaluate to get full model, call `plot_frame_deformed()` → `artifacts/winner_frame.png`

---

#### 6. `tests/test_search_pipeline_smoke.py` (NEW)
**Location:** `tests/test_search_pipeline_smoke.py`

**Imports:**
```python
import pytest
import pandas as pd
from mini_branch.explore import run_search
```

**Tests:**
- Run `run_search(n=10, seed=1)`
- Assert DataFrame has expected columns
- Assert at least one `ok=True`
- Assert no NaNs in key metrics for `ok=True` rows

---

#### 7. `tests/test_pareto.py` (OPTIONAL)
**Location:** `tests/test_pareto.py`

**Imports:**
```python
import pandas as pd
import numpy as np
from mini_branch.pareto import pareto_mask
```

**Tests:**
- Synthetic dataset with known Pareto points
- Verify `pareto_mask()` correctly identifies non-dominated points

---

### Import Dependency Graph

```
catalog.py
  └─ (no dependencies)

explore.py
  ├─ catalog.py (Material, Section, TIMBER_SECTIONS)
  ├─ model.py (Node, Frame2D)
  ├─ elements.py (element_geometry)
  ├─ assembly.py (assemble_global_K, dof_index, DOF_PER_NODE)
  ├─ loads.py (assemble_element_loads_global)
  ├─ solve.py (solve_linear, MechanismError)
  └─ post.py (element_end_forces_local)

pareto.py
  └─ (only pandas/numpy)

viz.py
  ├─ model.py (Node, Frame2D)
  ├─ elements.py (element_geometry)
  └─ assembly.py (DOF_PER_NODE, dof_index)

run_portal_search.py
  ├─ explore.py (run_search, make_portal, evaluate_variant)
  ├─ catalog.py (TIMBER_SECTIONS, DEFAULT_MATERIAL)
  ├─ pareto.py (pareto_mask)
  └─ viz.py (plot_pareto, plot_frame_deformed)

test_search_pipeline_smoke.py
  └─ explore.py (run_search)
```

### Key Integration Points

1. **`make_portal()` output format:**
   - Must match what `assemble_global_K()` expects: `dict[int, Node]`, `list[Frame2D]`
   - `element_udl_map` format: `dict[int, float]` (matches `assemble_element_loads_global()` input)
   - `nodal_loads` can be dict mapping `(node_id, dof)` tuples to values, or dict of node_id → array

2. **`evaluate_variant()` uses existing Day 2 pipeline:**
   - Same sequence as `demos/run_portal_frame.py` but parameterized
   - Must handle `MechanismError` gracefully (return `ok=False`)

3. **Metrics extraction:**
   - `drift`: Extract from `d` at top nodes (nodes 1, 2) using `dof_index()`
   - `max_abs_M`: Loop elements, call `element_end_forces_local()`, extract moments (indices 2, 5)
   - `volume`: Sum `A * L` using `element_geometry()` for length

4. **CSV structure:**
   - Columns: All `PortalParams` fields + metrics (drift, max_abs_M, volume, carbon, ok, reason)
   - One row per variant (even failures)

This structure ensures Day 3 code plugs cleanly into Day 2's validated solver without refactoring.
