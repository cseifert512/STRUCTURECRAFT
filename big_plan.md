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



#DETAILED VERSION
Mini-Branch 3D: Scope + Build Spec (Cursor-ready)
0) One sentence

Build a 3D spaceframe design explorer that takes an architectural “brief,” generates many canopy/spaceframe options, evaluates them with a real structural kernel (3D truss), ranks them using architecture + fabrication + engineering metrics, and exports a selectable option as a reproducible package (geometry + cutlist + report).

1) Why this exists

A 2D frame solver is undergrad-level. The goal is to demonstrate the thing engineers rarely ship and architects rarely can: a product-grade design loop that blends:

geometry generation (design intent)

physics verification (structural truth)

fabrication reality (parts, nodes, lengths)

decision tooling (tradeoffs, Pareto, selection)

AI assistance (accelerate search without bypassing physics)

2) The core outcome (what “done” looks like)

From a clean repo you can run:

“Generate one canopy → solve → visualize → export cutlist”

“Generate 200–2000 canopies → compute metrics → Pareto plot → select winner”

“Train a surrogate + guided search → find better designs with fewer solves”

Everything is deterministic with a seed, and tests stay green.

3) Scope boundaries (what’s in / out)
In scope (must)

3D truss structural kernel (axial-only bars)

3D canopy/spaceframe generator (several topologies)

Batch evaluation producing a dataset (CSV/Parquet)

Multi-objective selection (Pareto + ranking)

Fabrication-aware metrics (length clustering, node complexity, unique parts)

Architectural metrics (headroom/openness proxies)

Exports: cut list + model JSON + plots/HTML

Optional but strongly recommended: a small interactive explorer (Streamlit)

Out of scope (for now)

Full 3D frame bending/torsion (12×12 beam element)

Code-compliance (AISC/NDS/Eurocode)

Nonlinear material behavior

Detailed connection design / shop drawings

Rhino integration (nice later, not needed to win the story)

4) The “product twist” (your unfair advantage)

This is not “solve a structure.” This is: help a designer pick a structure.

The system must support:

“I want a light, open, buildable canopy with minimal unique parts.”

“Show me options and tradeoffs.”

“Explain why one is better.”

“Export the one I choose.”

That’s the difference between an engineer and a platform builder.

5) System architecture overview

Everything is modular so 3D work doesn’t destroy your existing 2D baseline.

5.1 Core layers

Kernel (generic solver plumbing)

DOF management, assembly, boundary conditions, linear solve, diagnostics

Elements

2D frame elements (existing) live as baseline

3D truss elements (new) power the canopy system

Generative geometry

canopy generators: grid, triangulation, diagrid, spaceframe variants

Loads + supports

supports selection and nodal loads (gravity + lateral proxy)

(self-weight optional; start with nodal gravity equivalent)

Metrics

engineering: displacements, reactions, axial forces, utilization proxies

fabrication: length bins, unique part count, node degree/angle spread

architectural: headroom, openness, silhouette/height

Explorer

batch sampling, filtering, Pareto, ranking, guided search

Visualization + export

3D preview (Plotly)

exports: cutlist CSV, model JSON, report HTML/PNG

6) Key design decisions
6.1 Start with 3D truss, not 3D frame

3D truss gets you out of “undergrad 2D” while staying stable and shippable. It’s enough to produce compelling canopy structures and the product story.

6.2 “AI proposes; physics verifies”

All AI outputs are suggestions. Every chosen design is re-solved using the true kernel. No exceptions.

6.3 Everything is reproducible

Every batch run takes a seed. Every output is saved with metadata: seed, version, brief parameters.

7) Data model and formats
7.1 Design brief (input)

A single design_brief.json describing intent + constraints + weights.

Must include:

footprint: width, depth

support scheme: corners / edges / custom points

height intent: max height, min headroom, heightfield type

topology: grid/tri/diagrid (selectable)

constraints: shipping max length, max unique parts, drift/displacement cap

weights: stiffness vs material vs openness vs simplicity

7.2 Model package (output)

A selected design exports a folder:

model.json
nodes, elements, sections, supports, loads, metadata

cutlist.csv
member id, node i/j, length, length_bin, force, utilization, section

preview.html (Plotly 3D)

summary.md or report.html
key metrics, warnings, plots

7.3 Batch results (output)

results.csv with one row per design:

params + metrics + flags + file pointer to model package (optional)

8) Structural kernel scope (3D truss)
8.1 DOFs

Node has 3 DOF: (ux, uy, uz)

8.2 Element behavior

Axial only

Stiffness derived from direction cosines (global coordinates)

Output per bar: axial force, axial utilization proxy

8.3 Loads

Nodal loads only (start)

Gravity: lump to nodes (or distribute by tributary area)

Lateral: nodal forces on selected nodes (wind proxy)

8.4 Supports

Translational constraints at supports (fix ux/uy/uz)

Mechanism detection must be strong (clear error outputs)

8.5 Validation requirements

Must include at least:

symmetry / equilibrium test (tetrahedron)

convergence sanity (refine a canopy grid and show stability)

reaction sums match applied loads

9) Generative canopy scope
9.1 Canonical generator: “Spaceframe canopy”

Given footprint and grid density:

create a base grid of nodes

compute a heightfield (flat / paraboloid / ridge)

generate bar connectivity:

Topology A: triangulated surface (triangle mesh edges)

Topology B: grid + diagonals

Topology C: diagrid pattern

supports at corners/edges/custom

9.2 Parameters (the ones that matter)

width, depth

nx, ny (density)

max height, heightfield type

topology selection

support layout

section choice (single size first, later multi-size catalogs)

9.3 Minimum architectural metrics

Headroom score: portion of footprint above min_clear_height

Openness score: penalty for bars entering the “occupied zone” below headroom

Simplicity score: based on bar count + unique parts

10) Metrics (what gets measured)
10.1 Engineering metrics (must)

max displacement magnitude

max vertical displacement (sag)

max lateral displacement (if lateral load included)

max axial force (tension/compression)

utilization proxy (|N| / N_allow)

10.2 Fabrication metrics (must)

max member length (shipping constraint)

length clustering count (bins within tolerance)

unique parts count (section + length bin)

node degree distribution (connectivity)

node complexity score (degree + angle spread proxy)

10.3 Architectural metrics (must)

headroom score

openness score

“visual clutter” proxy (bar density in view corridor)

10.4 Flags/warnings (must)

violates shipping max length

too many unique parts

excessive node degree

displacement over cap

11) Explorer scope (batch + Pareto + selection)
11.1 Batch generation

Generate N designs by sampling parameters within the brief constraints:

density sweep (nx, ny)

height sweep

topology options

support layouts

11.2 Filtering

Hard reject designs that violate hard constraints (shipping length, instability). Keep a reason code.

11.3 Pareto frontier

At minimum:

minimize material proxy (volume/total length)

minimize displacement (stiffness)

maximize openness/headroom

Store Pareto mask and rank candidates.

11.4 Selection logic

Provide:

“Top 10” candidates by weighted score (from brief)

ability to pick a candidate and export package

12) AI scope (meaningful, not gimmicky)
12.1 Surrogate model

Train a model to predict key metrics from parameters:

displacement

material proxy

openness/headroom proxy

12.2 Guided search (active learning-lite)

Loop:

solve a small random batch

train surrogate

propose candidates (best predicted + uncertainty)

re-solve top candidates

measure: “Pareto improvement per solve”

12.3 Rules

all final outputs are solver-verified

always report prediction vs actual to show honesty

13) Visualization scope
13.1 3D view

Plotly 3D lines for members:

color by axial force or utilization

optional node markers for supports

13.2 Plots

Pareto scatter (2D)

histograms: member lengths, node degrees

predicted vs actual (ML)

14) Tooling expectations

One-command scripts that generate artifacts

Clean folder structure

Tests that cover:

kernel sanity

equilibrium

generator smoke

metrics sanity

ML smoke (fast)

15) Milestones (build order)
Milestone A: Kernel generalization (fast)

dimension-agnostic assembly/solve interface

keep 2D baseline working

Milestone B: Truss3D works + validated

tetrahedron test green

space truss demo produces sensible results

Milestone C: One canopy generator + cutlist export

generate canopy

solve

export cutlist + preview

Milestone D: Batch explorer + Pareto

generate hundreds of canopies

results.csv + pareto plot

Milestone E: Product surface

Streamlit explorer with filters + 3D preview + export selection

Milestone F: Guided search AI

surrogate training + guided candidate selection

show Pareto improvement per solve

16) Acceptance criteria (non-negotiable)

Reproducible runs with seed

Clear failure reasons (unstable, violates constraints, etc.)

Exports are complete and re-runnable

Tests stay green

The demo story can be told in < 8 minutes with visuals

17) The demo story (how you’ll present it)

Show the brief: “Design me a light, open canopy with shipping limits.”

Run batch generation: “Here are 500 viable options.”

Show Pareto: “Here’s the trade space of openness vs stiffness vs material.”

Click one option: show 3D view colored by force, plus warnings/metrics.

Export: cutlist + model package.

Run guided search: “AI finds better options with fewer solves.”

Close: “This is design-to-fabrication decision tooling, not a homework solver.”

18) Next technical expansion (future, optional)

Once the product story is strong:

upgrade to 3D frame element (bending/torsion)

introduce member loads

connection family modeling

Rhino bridge and manufacturing outputs

But the canopy explorer alone is already a serious differentiator.