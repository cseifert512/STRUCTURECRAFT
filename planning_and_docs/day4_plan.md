Day 4 is the "AI + engineering discipline" day: take the clean dataset from Day 3, train a surrogate model, use it to guide exploration, and package the whole repo so it reads like a miniature Branch R&D product.

No hype. The pitch is: "Model proposes; physics verifies."

Day 4 objectives

By end of Day 4 the following should run:

py .\demos\train_surrogate.py
py .\demos\guided_search.py


…and the outputs are:

artifacts/model.joblib (trained surrogate)

artifacts/pred_vs_actual.png (trust check)

artifacts/feature_importance.png (interpretability)

artifacts/guided_vs_random.png (the "AI actually helps" plot)

artifacts/guided_results.csv (re-evaluated best candidates)

Updated README.md + ENGINEERING_NOTES.md (the "detail-oriented" proof)

pytest still passes.

System requirements (Day 4)

Inside the venv:

py -m pip install pandas scikit-learn joblib


numpy/matplotlib/pytest are already available.

Scope (Day 4)
IN

Supervised surrogate model (fast, robust): RandomForest or GradientBoosting

Proper train/validation split + metrics (MAE/R²)

Guided search loop: surrogate screens thousands → solver confirms top K

Clean plots + saved artifacts

Packaging: README + engineering notes + "how to reproduce" commands

OUT

LLM "prompt-to-parameters" (not needed; distracts)

Deep learning

Any claim of code-compliant design

Architecture rule (how it interacts with Days 1–3)

The solver is not touched. Day 4 consumes Day 3's outputs.

Day 4 reads:

artifacts/results.csv from Day 3

The same explore.make_portal() and explore.evaluate_variant() to re-evaluate candidates with true physics

Day 4 adds:

scripts that call existing evaluation functions

no breaking changes to core modules

This is important: it proves one can build on a stable kernel like a real codebase.

Day 4 plan (timeboxed, ruthless)
Block A (1–2 hours): Make the dataset "ML-ready"
Goal

Turn results.csv into clean X, y with no footguns.

Deliverable

Create demos/_ml_utils.py (private helper module) with:

load_clean_df(path):

filter ok == True

drop rows with NaNs

enforce numeric dtypes

make_features(df):

features: span, height, brace, sec_col, sec_beam, sec_brace, udl_w, wind_P

encode categoricals: simplest is keep ints; better is one-hot for brace (but RF can handle ints fine)

targets: start with one target for clarity:

drift (or max_drift)

Expectation
The training script never dies because of strings/NaNs.

How it relates
Guided search uses the same feature function to score new candidates.

Block B (2–3 hours): Train a surrogate model + prove it's not nonsense
Goal

A model that predicts drift reasonably and is easy to explain.

Deliverable

Create demos/train_surrogate.py:

What it does

Load artifacts/results.csv

Build X/y

Split train/test (e.g., 80/20 with fixed seed)

Train model (pick one):

RandomForestRegressor(n_estimators=300, random_state=0)

or HistGradientBoostingRegressor(random_state=0)

Report metrics:

MAE (mean absolute error)

R²

Save artifacts:

artifacts/model.joblib

pred_vs_actual.png

feature_importance.png (RF has this)

Expectation

The claim becomes: "The surrogate predicts drift with MAE ≈ X and R² ≈ Y on held-out data."

The scatter plot should roughly follow the diagonal.

How it relates
This is the model that will be used to cheaply score thousands of candidates.

Block C (3–5 hours): Guided search (the money demo)
Goal

Demonstrate that ML accelerates design exploration without bypassing physics.

Deliverable

Create demos/guided_search.py:

Workflow

Bootstrap: evaluate N0=200 random variants with the true solver

reuse explore.sample_params(seed) + explore.evaluate_variant()

save to artifacts/bootstrap.csv

Train surrogate on bootstrap (or load the model trained from full Day 3 dataset if available)

Candidate flood: generate Ncand=5000 parameter sets (no solving)

Score candidates with surrogate (predict drift)

Apply hard constraints before re-eval:

shipping limit

unique sizes

etc. (reuse the same constraint logic from Day 3)

Select top K=100 by predicted drift (or drift + volume score)

Truth check: re-evaluate those K with the real solver

Compare against random baseline:

Evaluate another K random designs with the solver

Plot distribution / best-of-K:

Outputs

artifacts/guided_results.csv

artifacts/random_results.csv

artifacts/guided_vs_random.png:

scatter: volume vs drift for guided vs random (two marker styles)

plus highlight best 10

Expectation
Guided set should be visibly shifted toward better drift for similar volume, or show better Pareto points.

How it relates
This becomes the "AI twist" that still respects structural engineering.

Block D (1–2 hours): Add one "explainability" hook (fast, impressive)
Goal

Show the ability to introspect the model and the design space.

Deliverable

In guided_search.py print a tiny "explanation card" for the best design:

params

predicted drift

actual drift

top 3 feature importances (from the model)

plus solver metrics: max |M|, volume proxy

Expectation
This reads like a product feature: "why this option".

Block E (1–2 hours): Hardening + tests (detail-oriented proof)
Deliverables

Add tests/test_ml_smoke.py:

Load a small subset of results.csv (or generate n=30 on the fly to avoid test-data dependency)

Fit a tiny model quickly

Assert:

model predicts finite numbers

prediction shape matches inputs

Add tests/test_guided_search_smoke.py (optional):

Run a tiny guided loop with Ncand=200, K=10 and assert it completes without error.

Expectation
The ML layer is reproducible and doesn't rot.

Block F (1–2 hours): Packaging (the last mile that wins interviews)
Deliverables

Update README.md to a "3-command" story:

single portal demo

random search + Pareto

train surrogate + guided search

Add ENGINEERING_NOTES.md (short but surgical):

assumptions: 2D, linear elastic, small deflection

validation table (from Day 1–2)

UDL implementation notes

mechanism detection / failure handling

how the surrogate is used (proposal vs verification)

limitations + next steps

Expectation
Lucas can skim it and instantly see: correctness + systems thinking.

End-of-Day-4 deliverables checklist

Expected deliverables:

Scripts

demos/train_surrogate.py

demos/guided_search.py

demos/_ml_utils.py

Artifacts

artifacts/model.joblib

artifacts/pred_vs_actual.png

artifacts/feature_importance.png

artifacts/guided_vs_random.png

artifacts/guided_results.csv

artifacts/random_results.csv

Docs

README.md

ENGINEERING_NOTES.md

Tests

tests/test_ml_smoke.py (+ optional guided smoke)

Narrative that can be claimed after Day 4

"I built and validated a 2D frame analysis kernel, wrapped it in a parametric design explorer with fabrication-aware constraints and Pareto outputs, then trained a surrogate model to guide search—using ML for proposal and the solver for verification. The workflow is reproducible, tested, and produces decision-grade tradeoff visuals."

That's the exact combination of technical + design + AI + paranoia that structural grads rarely package in four days.

Ruthless priority order (if time gets tight)

train_surrogate.py + pred_vs_actual.png

guided_search.py producing guided_vs_random.png

README + notes

tests (at least ML smoke)

Everything else is garnish.
