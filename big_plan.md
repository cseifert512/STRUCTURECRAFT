3D + generative geometry + fabrication-aware evaluation + a designer-facing decision workflow.

Here’s a rock-solid plan that builds directly on what you already have, and turns it into something structural grads almost never ship: a 3D space-structure design product.

The target artifact
3D Canopy / Spaceframe Design Explorer

Input: a design brief (footprint, supports, max height, headroom goals, shipping limits, part variety limits, stiffness targets).
Output: a set of 3D spaceframe options with:

structural performance (displacements, axial utilization, reactions)

architectural metrics (headroom volume, openness)

fabrication metrics (member length clustering, node complexity, unique parts)

Pareto frontier + “why this wins”

export: cut list + geometry (CSV/JSON, optionally OBJ)

This is unmistakably “Branch-y”: design → structure → fabrication truth → selection.

Phase 0 — Freeze what you have (so it becomes your baseline)

You don’t delete your 2D solver. You promote it to “baseline engine” and stop touching it except to generalize interfaces.

Deliverable:

mini_branch/v2d/ (your existing code lives here unchanged)

All existing tests remain green. This protects momentum.

Phase 1 — Refactor into a dimension-agnostic engine (the foundation for 3D)

The only reason most “3D attempts” die is because people hardcode DOFs and element shapes. Fix that once.

Key design decision

Introduce a simple internal “kernel contract”:

dof_per_node is not a constant.

every element implements:

dof_map() → list[int]

ke_global(nodes) → ndarray

(optional) fe_global(nodes) for member loads later

And create a DOF manager:

dof = dofman.idx(node_id, local_dof) where local_dof is 0..(dof_per_node-1)

Deliverables (new files):

src/mini_branch/kernel/dof.py (DOFManager)

src/mini_branch/kernel/assemble.py (dimension-agnostic assembler)

src/mini_branch/kernel/solve.py (your existing partition solve, but no baked-in dof count)

How it interacts with old code:

Your current 2D frame element becomes one implementation of the Element interface.

The assembler/solver stop caring if it’s 2D or 3D.

This is a “real engineering codebase” move. Lucas will feel it.

Phase 2 — Implement Truss3D (fast, real 3D, huge payoff)

This is your first “not undergrad 2D” leap.

Truss3D model

Nodes: (x, y, z)

DOFs per node: 3 (ux, uy, uz)

Element: axial only (EA/L)

Element stiffness: 6×6 built from direction cosines
(simple, stable, and validates cleanly)

Deliverables:

src/mini_branch/v3d/model.py (Node3D, Truss3D)

src/mini_branch/v3d/elements.py (Truss3D.ke_global(...))

demos/run_space_truss.py (one canonical 3D truss demo)

Tests that make it credible:

tests/test_truss3d_tetrahedron.py (symmetry/equilibrium)

tests/test_truss3d_cantilever.py (tip load deflection sanity via convergence or known simple case)

This gets you to “I can solve 3D geometry” immediately.

Phase 3 — Build the “architecture/product twist”: a 3D spaceframe generator

Now you’re out of “solver land” and into “design tool land.”

Start with one canonical typology

Canopy / pavilion spaceframe over a rectangular footprint.

Paramization (keep it tight, expressive):

footprint: width, depth

support layout: corner supports / edge supports / 6-point supports

heightfield type: flat / paraboloid / ridge

grid density: nx, ny

topology: square grid + diagonals OR pure triangulated mesh

Generator outputs:

node positions in 3D

bar connectivity list (elements)

support conditions (fixed translations at support nodes)

loads (gravity as nodal loads or lumped self-weight)

Deliverables:

src/mini_branch/generative/canopy.py:

generate_canopy(params) -> nodes, elements, fixed_dofs, loads

demos/run_canopy_single.py (renders + solves + prints metrics)

This is where you start looking like someone building Branch, not homework.

Phase 4 — Add metrics engineers don’t encode (your differentiator)

Keep core structural metrics:

max displacement

reactions

max axial force utilization (|N| / N_allow)

Add designer + fabrication metrics that are brutally product-relevant:

Architectural metrics

Headroom volume proxy: count grid points where canopy height > min_clear_height (or integrate a coarse voxel grid)

Openness / visual clutter: bar density in the “occupied zone” (penalize members below headroom)

Fabrication metrics (Branch-flavored)

Member length clustering: how many distinct length “bins” (e.g., within ±5mm / ±1/8") → fewer bins = easier fabrication

Node complexity:

node degree distribution (how many members meet at nodes)

“angle spread” at nodes (wider spread = more complex connectors)

Unique parts count: unique (section, length bin) combos

Deliverables:

src/mini_branch/metrics.py

these columns added to results.csv

This is where you beat structural grads: you’re optimizing for buildability + experience, not just stress.

Phase 5 — Product surface: interactive 3D Design Explorer

This is the “no one else can do this” moment.

Build a Streamlit app with Plotly 3D

left: sliders/toggles (span/depth/heightfield/density/support layout/topology)

center: 3D view (members colored by axial force or utilization)

right: metrics card + warnings + export

And crucially:

run batch generation (N variants)

show Pareto frontier (e.g., stiffness vs carbon proxy vs openness)

click a point → loads that design in 3D viewer

Deliverables:

app.py (Streamlit)

viz3d.py using Plotly:

plot_spaceframe(nodes, elements, color_by="force")

This immediately reframes your work from “solver” to “product prototype.”

Phase 6 — AI that’s actually meaningful: active learning for Pareto discovery

You already trained a surrogate once. Now you do it like a product:

Loop:

sample 100 designs → solve (truth)

train surrogate (predict max disp + openness + buildability)

propose next batch using expected improvement / uncertainty

re-solve proposals

watch Pareto frontier improve faster than random

Deliverables:

demos/active_learning.py

plot: “Pareto frontier quality vs #solves”

This is a very “tech lead” move: you’re optimizing compute, not role-playing AI.

Where we begin right now

Do these in order—no detours:

Step 1: Create the new kernel scaffolding

kernel/dof.py, kernel/assemble.py, kernel/solve.py

make your existing 2D code call into the kernel without behavior changes

run all existing tests

Step 2: Implement Truss3D + tetrahedron test

get one 3D demo running

get tetrahedron symmetry/equilibrium test passing

Step 3: Generate one canopy and solve it

no UI yet

just demos/run_canopy_single.py producing:

3D plot

max displacement

cut list (member lengths)

That’s the minimum “we have crossed into 3D product land” milestone.

File structure I’d use (clean, scalable)
src/mini_branch/
  kernel/
    dof.py
    assemble.py
    solve.py
  v2d/                # your existing frame solver (adapted to kernel)
  v3d/
    model.py
    elements.py       # Truss3D first
  generative/
    canopy.py
  metrics.py
  viz/
    viz2d.py
    viz3d.py
demos/
  run_space_truss.py
  run_canopy_single.py
  run_canopy_search.py
tests/
  test_truss3d_tetrahedron.py
  test_canopy_smoke.py

The “tell Lucas in one line” positioning

“I moved beyond 2D textbook solvers by building a 3D spaceframe design explorer: generative canopy structures, structural evaluation, fabrication-aware metrics (part clustering, node complexity), and an interactive selection workflow with active-learning search.”

That’s not undergrad. That’s platform thinking.

If you paste your current repo tree (just file/folder names), I’ll map exactly what to rename/move for the kernel split and give you the first two files (dof.py + Truss3D) with precise code so you can start implementing immediately without refactor thrash.