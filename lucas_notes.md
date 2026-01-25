# Lucas Notes

Feedback and suggestions from Lucas on next steps for StructureCraft.

---

## Suggested Features

### 1. Deflected Shapes, Bending Moments, and Actual Forces

Lucas recommended integrating these three visualization/analysis features:

#### Deflected Shapes (Curved Beam Deformations)
- **Current state**: We show deformed shapes as straight lines between displaced nodes
- **What's needed**: Show the **curved deflected shape** of frame elements using Hermite interpolation
- **Why it matters**: 
  - Visual verification that structure behaves as expected
  - More professional/realistic visualization
  - Shows where bending is most severe

#### Bending Moments (M Diagram)
- **What it is**: Shows how much the beam is "trying to bend" at every point
- **Why it matters**:
  - Moment causes failure in beams - engineers size beams based on max moment
  - Shows where reinforcement is needed
  - Required for code compliance checks
- **Current state**: We compute element END moments in `post.py`, but don't interpolate along the element

#### Actual Forces (N, V, M Diagrams)
- **N (Axial Force)**: Tension/compression along the member
- **V (Shear Force)**: Force trying to "cut" the beam
- **M (Bending Moment)**: Force trying to bend the beam
- **Why it matters**:
  - Required for any real engineering check
  - Shows how forces flow through the structure
  - Critical for design optimization

---

### 2. Real-time 2D Plate FE on GPU

Lucas also mentioned this but noted it's "incredibly difficult."

#### What is Plate FE?
- **Different from our current 1D elements** (beams/trusses)
- **Plate elements** model flat surfaces: floor slabs, wall panels, canopy surfaces
- Each element has 4+ nodes forming a surface mesh
- ~5 DOFs per node (3 translations + 2 rotations)
- Stiffness matrices are much larger and more complex

#### Why GPU?
- For real-time interaction, the solve cycle must happen in <100ms
- For plate meshes with 1000+ elements, CPU solvers are too slow
- GPU parallelization can accelerate assembly and potentially solving

#### Why It's Difficult
1. **Element formulation**: Plate elements (Mindlin-Reissner or Kirchhoff) are complex
2. **Meshing**: Need automatic mesh generation for arbitrary shapes
3. **GPU programming**: CUDA/WebGPU is specialized
4. **Solver on GPU**: Linear solvers on GPU are research-level problems
5. **Integration**: Connecting GPU solver to existing architecture

#### Hardware Assessment (My Machine)
| Component | Spec | Notes |
|-----------|------|-------|
| GPU | GTX 1660 Ti (6 GB) | CUDA 7.5, Turing architecture |
| CPU | Ryzen 5 3600 | 6-core @ 3.59 GHz |
| VRAM | 6 GB | Limiting factor for large meshes |

**Verdict**: Can handle small-to-medium meshes (up to ~50×50 elements) for proof-of-concept, but not production-scale real-time analysis. The bigger barrier is software complexity, not hardware.

---

## Decision: Which to Tackle First?

### Recommendation: Start with Force Diagrams ✅

| Factor | Force Diagrams | GPU Plate FE |
|--------|----------------|--------------|
| Difficulty | Moderate | Very High |
| Time to ship | 1-2 weeks | 2-4 months |
| Uses existing code | ✅ Yes | ❌ New solver |
| Immediate value | ✅ High | ⚠️ Requires new UI |
| Learning curve | Manageable math | GPU programming + advanced FE |
| Risk | Low | High |

**Reasoning**: Force diagrams build directly on our existing solver, make the tool look professional, and are required for any real engineering use. Can be done incrementally.

---

## Implementation Plan: Force Diagrams

### Step 1: Curved Deflected Shapes
- Add Hermite interpolation for frame element deflection
- Use nodal displacements (ux, uy) AND rotations (rz) 
- Sample multiple points along each element for smooth curves

### Step 2: Internal Force Diagrams (N, V, M)
- Extend `post.py` with `element_force_diagram()` function
- For constant axial: N is constant along length
- For shear with UDL: V(x) = V_i - w*x
- For moment with UDL: M(x) = M_i + V_i*x - w*x²/2

### Step 3: Visualization
- Add `plot_force_diagrams()` to `viz.py`
- Draw diagrams perpendicular to each element
- Color by tension/compression or magnitude
- Scale diagrams for visibility

### Step 4: Integration (Optional)
- Add to web frontend if desired
- Export diagram data via API

---

## Status

- [x] Discussed concepts and implementation approach
- [x] Implement curved deflected shapes (Hermite interpolation in `mini_branch/diagrams.py`)
- [x] Implement N, V, M computations along elements (`compute_element_internal_forces()`)
- [x] Add force diagram visualization (`Canvas2D.tsx`, `DiagramPanel.tsx`)
- [x] Integrate with web frontend (FastAPI endpoints, React components)

---

## Implementation Summary (Completed Jan 25, 2026)

### Backend (Python)

| File | Changes |
|------|---------|
| `mini_branch/diagrams.py` | **NEW** - Core force diagram computations with Hermite interpolation |
| `mini_branch/post.py` | Extended with drift, reactions, and re-exports from diagrams |
| `api/main.py` | Added `/api/frame2d/generate` and `/api/frame2d/diagrams` endpoints |

### Frontend (React/Next.js)

| File | Changes |
|------|---------|
| `web/src/lib/types.ts` | Added Frame2D types, DiagramPoint, etc. |
| `web/src/lib/api.ts` | Added `generateFrame2D()`, `getFrame2DDiagrams()` |
| `web/src/store/designStore.ts` | Added 2D mode state, diagram controls |
| `web/src/components/Canvas2D.tsx` | **NEW** - 2D frame SVG visualization with diagrams |
| `web/src/components/DiagramPanel.tsx` | **NEW** - N/V/M toggle and force summary |
| `web/src/components/ControlPanel.tsx` | Added 2D/3D mode toggle |
| `web/src/components/Canvas3D.tsx` | Added deflected shape visualization |
| `web/src/app/page.tsx` | Integrated 2D mode view |

### How to Test

1. Start the backend: `cd api && uvicorn main:app --reload`
2. Start the frontend: `cd web && npm run dev`
3. Toggle between "3D Spaceframe" and "2D Frame" modes in the control panel
4. In 2D mode:
   - Adjust span, height, loads
   - Toggle diagram types (N, V, M)
   - View curved deflected shape
   - See max forces and drift results

---

*Last updated: January 25, 2026*
