Day 2 is where the solver stops being a cantilever trick and becomes a usable structural kernel: it can handle member loads (UDL), it can solve a portal frame, and it can output engineering-relevant quantities (reactions, drift, end moments). That's the bridge to Day 3 (design exploration).

Below is a ruthless, executable Day-2 plan with file-by-file deliverables, integration points, and expectations.

Day 2 objectives

By end of Day 2 the following should run:

pytest
py .\demos\run_portal_frame.py


…and the outputs are:

portal frame solves under gravity UDL + lateral point load

printed metrics: max drift, support reactions, max |M|

a simple plot: undeformed vs deformed frame

a UDL validation test that passes (with a meshed beam)

Scope (what's in / out)
IN (Day 2)

Uniform Distributed Load (UDL) on frame elements (local transverse direction)

Assembly of element equivalent nodal loads into global F

Portal frame demo (2 columns + beam)

Postprocessing: element end forces (N, V, M) in local coordinates

Sanity tests: equilibrium + stability + UDL beam check

OUT (Day 2)

Code-compliant checks (AISC/NDS/etc.)

Fancy diagrams (full moment curve along member)

Rhino/Grasshopper bridge

Optimization / AI surrogate (Day 4)

Architecture rule (how new code interacts with the existing code)

These remain stable:

assemble_global_K(nodes, elements) ✅

solve_linear(K, F, fixed_dofs, ...) ✅

The Day-1 tests must keep passing unchanged.

New additions:

loads.py (new)

post.py (new)

viz.py (optional new)

new demos/tests

Sign conventions (be aware of these)

Global axes: +y up

Downward gravity: Fy = negative

UDL input: define w as force per length in local +y

so for downward load on a horizontal beam: w = -abs(w_down)

Element local DOFs order stays:
[uix, uiy, rzi, ujx, ujy, rzj]

This keeps the math coherent across transform / assembly / post.

Day 2 plan (timeboxed, ruthless)
Block A (1–2 hours): Add element UDL equivalent nodal load
Goal

Given an element length L and UDL w in local +y, produce the 6-vector of consistent nodal loads in local coordinates.

Deliverable

Create: src/mini_branch/loads.py

Functions to implement:

frame2d_equiv_nodal_load_udl(L: float, w: float) -> np.ndarray

assemble_element_loads_global(nodes, elements, element_udls) -> F_global

Definition
Consistent nodal load vector for uniform transverse load w (local +y) is:

Shears: wL/2 at each node

End moments: wL^2/12 with opposite signs

Local equivalent nodal force vector:
[0, wL/2, wL^2/12, 0, wL/2, -wL^2/12]

Then transform to global via the same T already available:
f_global = T.T @ f_local

Expectation

F can now be built from:

nodal point loads (existing functionality)

plus element UDL loads (new)

Block B (2–3 hours): Validate UDL with a meshed simply supported beam
Why meshed?

A single Euler–Bernoulli element won't match UDL deflection perfectly (UDL produces a quartic curve). Mesh it (e.g., 10 elements) and compare midspan deflection.

Deliverable

Create: tests/test_simply_supported_udl.py

Test case

Beam length L

n = 10 elements, n+1 nodes

Supports: pinned both ends (fix ux, uy at both ends; rotations free)

Apply UDL downward: w = -w0 on each element

Measure midspan node uy_mid

Closed form max deflection:

𝛿
𝑚
𝑎
𝑥
=
5
𝑤
0
𝐿
4
384
𝐸
𝐼
δ
max
​

=
384EI
5w
0
​

L
4
​


(sign negative for downward)

Tolerance
Use something like rtol=2e-2 (2%) for n=10. For tighter tolerance, bump mesh to n=20.

Expectation

This test is the "I don't just code, I validate loads" proof.

Block C (2–3 hours): Postprocessing end forces (N/V/M) per element
Goal

Compute element end forces in local coordinates.

Deliverable

Create: src/mini_branch/post.py

Core function
element_end_forces_local(nodes, element, d_global, udl_w=None) -> np.ndarray

Implementation recipe:

compute L,c,s, k_local, T

d_local = T @ d_elem_global

f_local = k_local @ d_local

if element has a UDL: subtract its equivalent nodal load:
f_local = f_local - f_udl_local

return f_local as [Ni, Vi, Mi, Nj, Vj, Mj]

Expectation

For any solved structure it's now possible to print:

max |M| across all members

reactions are already in R

this is one step away from utilization checks (Day 3/4 optional)

Block D (2–3 hours): Portal frame demo (gravity + lateral)
Deliverable

Create: demos/run_portal_frame.py

Model
Nodes:

0: left base (0,0)

1: left top (0,H)

2: right top (L,H)

3: right base (L,0)

Elements:

left column: (0–1)

beam: (1–2)

right column: (3–2) (or 2–3, consistent)

Supports (keep it simple and stable):

pinned bases: fix (ux, uy) at node 0 and node 3
rotations free

Loads:

Beam UDL downward: w = -w0 in local +y

Lateral point load at node 1 or 2: Fx = +P

Outputs:

drift = max horizontal displacement at top nodes

reactions at supports

element end moments (from post.py)

plot deformed shape (scaled)

Expectation

This demo should run with a single command and produce a plot. This is the "Lucas sees it and immediately gets it" moment.

Block E (1–2 hours): Sanity tests for the portal frame + equilibrium
Deliverables

Create:

tests/test_portal_frame_equilibrium.py

Assertions:

solver does not throw MechanismError

drift finite and nonzero under lateral load

global equilibrium: sum of reactions in x ≈ sum of applied Fx; sum in y ≈ sum of applied Fy (UDL total)

This reinforces "detail-oriented, not vibes-based."

End-of-Day-2 deliverables checklist

These new files are expected:

src/mini_branch/loads.py

src/mini_branch/post.py

demos/run_portal_frame.py

tests/test_simply_supported_udl.py

tests/test_portal_frame_equilibrium.py

And these outcomes:

pytest all green (Day 1 tests still pass)

portal demo prints drift + max moment + reactions

UDL beam validation passes

How Day 2 sets up Day 3 (directly)

Day 3 is "generate variants + Pareto frontier." That requires:

A standardized design family (portal frame) ✅ (Day 2 demo becomes the generator template)

A richer load model (UDL + lateral) ✅

A way to compute objective metrics (drift, moments, volume proxy) ✅ (postprocessing unlocks this)

So Day 2 is the last "pure engineering plumbing" day before moving to full design/product mode.

Recommended working order (don't improvise)

Implement loads.py UDL vector + global assembly

Get UDL beam test passing

Implement post.py end forces

Build portal demo using those modules

Add portal equilibrium test

Following that order will result in minimal thrash and maximum "looks professional" output.
