<!-- Tier 2C.5 -->

# Experiment Tracking — MLflow, Declined

← [Back to README](../README.md)

Evidence: [`outputs/reports/tracking_traceability_audit.json`](../outputs/reports/tracking_traceability_audit.json)
· regenerate with `python src/monitoring/tracking_audit.py`

**Plan item:** Tier 2C.3 — *"MLflow (local file backend is fine): log params,
metrics, artifacts, model versions. Migrate the hand-rolled JSON registry to
MLflow Model Registry **or document why not**."*

**Decision: DECLINED.** This document is the "why not", and it is written to be
falsifiable — §6 states exactly what would overturn it.

---

## 1. The claim being tested

> The artifact + `superseded/` pattern already provides per-run traceability with
> preserved predecessors, so MLflow would duplicate existing function.

That is a claim **about this repository**, which makes it checkable. A checkable
claim asserted in prose is a claim nobody has checked, so it is measured by
`src/monitoring/tracking_audit.py` and the audit is reported **with its
failures**. An all-green traceability audit written by the person who built the
traceability is not evidence.

## 2. What per-run traceability requires

Six requirements. Each is checked by an observable that is **false when the
property is false** (CLAUDE.md R6) — the point of listing them first is that the
answer could have come back negative, and on two of them it did.

| | Requirement | Result |
|---|---|---|
| **T1** | every generated report names itself and lives at a stable path | ✅ |
| **T2** | every report records seed and package versions | ❌ **1 gap** |
| **T3** | superseding an artifact preserves its predecessor | ✅ |
| **T4** | model lineage records training data, trigger and promotion evidence | ✅ |
| **T5** | model binaries carry provenance (versions + content hash) | ✅ *(closed Tier 2C.7)* |
| **T6** | a re-run is byte-comparable to the original | ✅ |

**4 of 5 machine-checked requirements pass.** T5 was a failure when this
document was written and is now closed — §4 keeps both the gap and its closure,
because deleting the finding would delete the reason the audit was worth running.
The one remaining failure stays open and named.

## 3. What already works, measured

### T2 — reports record seed and package versions

**15 of 16 checkable reports** carry a `reproducibility` block recording seed,
Python version and package versions. Two further reports
(`language_audit_plan.json`, `temporal_language_inventory.json`) are exempted by
name, because they are text scans of the source tree with no seed and no numeric
result — the exemption is declared in the audit rather than hidden in a silent
skip.

### T3 — predecessors are preserved

| | |
|---|---|
| Preserved files under `outputs/reports/superseded/` | **84** |
| Artifacts with a versioned history in `superseded/auto/` | **10** |
| Deepest histories | `defense_report_lgbm_v1.json` (5 versions), then `alert_report_val_test.json`, `concept_drift_val_test.json`, `pipeline_summary.json`, `sliding_windows_val_test.csv` (4 each) |
| Curated supersession sets | `superseded/tier0_8signal/` (8 files), `superseded/tier2a_merged_target/` (34 files) |

Enforcement is **structural, not conventional**: `src/monitoring/artifact_io.py`
raises `ArtifactOverwriteError` on an unflagged overwrite, and copies the prior
version into `superseded/auto/` when the overwrite is explicit. Both behaviours
are unit-tested in `tests/test_artifact_integrity.py`. This is the property the
plan calls the preservation rule, and it is the one the project's scientific
argument depends on: the falsification arc is legible only because the superseded
results still exist.

### T4 — model lineage

`model_registry.json` records `train_splits`, `trigger`, `status`, `params`,
per-split `metrics`, `registered_at`, `n_features` and `train_rows` for every
model; `registry_history.csv` is the append-only view; `model_comparison.json`
carries the promotion evidence (DeLong test plus a patient-clustered paired
bootstrap). The `lgbm_v2` entry records its own confound in the lineage note —
that v1 used early stopping and v2 a fixed tree count, so the test difference is
not attributable to the extra data alone.

### T6 — byte-reproducibility

Enforced in CI on every commit by `.github/workflows/ci.yml:determinism`. Current
verdict **PASS over 24 artifacts**, evidenced in
`outputs/reports/determinism.json`.

Worth recording how that number got there. The check first ran over **12**
artifacts and passed. It reached 24 only after the sandbox was found to be
**leaking into the real repository** — seven modules held hardcoded absolute
paths, so a "sandboxed" run was writing its artifacts into the working tree it
was supposed to be isolated from. The sandbox is now verifiably hermetic: a run
touches **zero** repository files, checked by hashing the repository before and
after. A coverage number is only meaningful once you know what it was measuring.

## 4. The gaps, named — one closed, one open

An audit that reports only what passes is a brochure. Both failures below were
real, and **neither was closed by adopting MLflow** — one was closed by making
the two-line call the audit proved was missing, which is the argument in §5.4
made concrete.

### Gap 1 — `threshold_policy_lgbm_v1.json` has no `reproducibility` block

One report of sixteen. It is a substantive one — R-12's evidence — so this is not
cosmetic. **Fix:** add the block to `src/uncertainty/threshold_policy.py` and
regenerate. Roughly four lines.

**Would MLflow close it?** No. MLflow records the environment of a *run*, but the
number quoted in the README comes from this *file*, and the file would still lack
its provenance block. The fix is the same four lines either way.

### Gap 2 — no shipped model carried a provenance sidecar — **CLOSED (Tier 2C.7)**

**As measured: 0 of 4** model binaries (`lgbm_v1.pkl`, `lgbm_v2.pkl`,
`logreg_v1.pkl`, `logreg_scaler.pkl`) had one. **Now 4 of 4.**

The diagnosis is the sharper part. `src/monitoring/model_io.py` **implements**
sidecars — package versions, a SHA-256 content hash, `ArtifactVersionError`
raised loudly on version skew or tampering — and it is **unit-tested**, and it has
**zero callers anywhere in `src/`**. Models are still written by a bare
`pickle.dump` in `trainer.py`.

This is the same criticism this project levels at conformal prediction that
nothing consumes: **a capability nothing calls protects nothing.**

The audit therefore checks for the sidecar **file**, not for the existence of
the sidecar-writing function. **That is a design decision, stated as one and not
as an apology:** a check on the function would have passed while the property was
false, which is CLAUDE.md R6 in a single line — *verify the property, not a
side effect of it*. The audit was built to be able to return this answer, and it
did.

**Fix, applied.** `trainer.py` and `registry.py` now route through
`model_io.save_model`. Every shipped model carries package versions, the git
commit, and a SHA-256 of its own bytes.

**The regeneration moved nothing but provenance**, which was the prediction:
all four model binaries came back **byte-identical**, and `evaluation_report.json`,
`model_comparison.json` and every registry metric block are unchanged.

One deliberate limitation, stated rather than glossed: `save_model` was given a
`serializer="pickle"` option and the two call sites use it, so the **on-disk
format is unchanged**. Eighteen modules read these artifacts with `pickle.load`,
and a joblib file is not readable that way — joblib wraps numpy arrays in objects
only its own unpickler resolves, so a silent format swap would have handed every
consumer a wrapper instead of an array. **Provenance and format are two different
gaps and only one is closed here.** Migrating off pickle is
`docs/TIER_1_7_SCOPE.md` P1 and needs all eighteen call sites moved to
`load_model()` in one change.

A test now asserts the property directly: every shipped `.pkl` has a sidecar, and
the sidecar's recorded hash matches the artifact's actual bytes. Asserting that
`save_model` exists would have passed throughout the period it had no callers.

**Would MLflow have closed it?** Partly, and not more cheaply — and this is now
settled empirically rather than argued. `mlflow.lightgbm.log_model` would capture
the model with its environment, but only if called, at exactly the call site
where `save_model` needed to be called. **The gap was a missing call, not a
missing tool**, and closing it took two call sites and no new dependency.
Adopting a tracking framework to fix an uncalled function would have replaced a
two-line change with a permanent one.

## 5. What MLflow would actually add, and what it would cost

Stated fairly, because the decision is worth something only if the alternative was
considered properly.

**It would genuinely add:**

- A run-comparison UI and a query API over runs.
- A standard schema, so a newcomer knows where to look without learning this
  repository's conventions.
- Automatic environment capture at log time.
- Model Registry stage transitions (`Staging` → `Production`) with an audit trail.

**It would cost:**

1. **A second store that duplicates the first.** `mlruns/` alongside `outputs/`.
   Two places to look, and a standing question of which is authoritative.
2. **README traceability gets *worse*, not better.** R4 requires every number in
   the README to map to a **named file**. Today that is
   `outputs/reports/fairness_audit.json` — citable, greppable, diffable in git,
   reviewable by someone who has only the repository. Under MLflow it becomes
   `mlruns/0/<32-hex-run-id>/artifacts/fairness_audit.json`: opaque, unstable
   across re-runs, and not meaningfully diffable. For a repository whose central
   claim is *"every number traces to a file a hostile reviewer can open"*, that is
   a regression in the exact property being optimised.
3. **It does not encode supersession.** MLflow preserves *runs*. It does not
   record that artifact B **replaced** artifact A **and why** — which is precisely
   what `superseded/` and its README carry, and precisely what the scientific
   story (initial framing → falsification → re-diagnosis) is made of. That
   relation would have to be maintained by hand *on top of* MLflow.
4. **The decisive one: MLflow would not have caught a single traceability defect
   this remediation actually found.** There are now six:

| Defect | Would MLflow have caught it? |
|---|---|
| `registry.py` hardcoded `n_estimators: 173` while the fitted model stopped at 126 | **No** — `mlflow.log_param("n_estimators", 173)` logs the wrong number just as faithfully |
| `defense.py` findings prose carried stale hand-typed AUCs after a regeneration — and `README_MUST_INCLUDE` then quoted them verbatim | **No** — prose inside a logged artifact is opaque to any tracker |
| `list(to_drop)` over a `set` made `pipeline_objects.pkl` byte-different across runs | **No** — MLflow logs bytes; it does not compare two runs' bytes |
| 0/4 models carry provenance | **No** — it needs the same call `save_model` needs |
| **Seven hardcoded absolute paths** to one developer's machine, including the raw-data directory | **No** — it would have logged artifacts from whatever path the code used, wherever that was |
| **The canonical metrics file computed at a threshold a prior phase had already withdrawn** | **No** — both values are legitimate parameters to log; nothing in a tracker knows one of them is superseded |

Every one was caught by an assertion, a hash comparison, a sandbox, or a check
designed to be able to fail. **The binding constraint on traceability in this
project has been checks that can fail, not a place to put the data.**

## 6. Decision, and what would overturn it

**DECLINED for this repository, at this scale.** Four models, one pipeline, one
operator, and a traceability requirement — *"every number maps to a named file"* —
that the current pattern serves better than MLflow would.

Adding a tool that duplicates existing function is the anti-pattern this
remediation has spent its whole length removing: the seven-stage selector with
five decorative stages, the five-layer defense with one layer carrying signal, the
conformal predictor nothing consumed. **Declining is that same judgement applied
to our own tooling choice.**

**This decision is wrong, and should be revisited, if any of the following becomes
true:**

- More than one person runs experiments concurrently and their results must be
  reconciled.
- Hyperparameter sweeps begin — beyond roughly 50 runs, hand-named artifacts stop
  scaling and a query API earns its place.
- A model must be **served** from a registry with enforced stage transitions,
  rather than versioned in a file.
- A reviewer can name a number in the README that cannot be traced to a named
  artifact, or an artifact whose predecessor was lost. *(They currently can name
  the two gaps in §4. Neither is a lost predecessor, and neither is fixed by
  MLflow — but both are open.)*

Until then: the tracking store is `outputs/`, the supersession log is
`outputs/reports/superseded/`, the lineage record is
`outputs/registry/model_registry.json`, and the guarantee that a re-run matches
the original is a CI job that compares hashes.
