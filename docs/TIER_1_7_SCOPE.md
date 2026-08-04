# Tier 1.7 — artifact and serialization integrity

Scope for the infrastructure tier. Every item here was discovered *while doing
other work*, which is the point: these defects damage evidence silently.

---

## P0 — Log truncation on import (PRIORITY DEFECT, three incidents)

`src/monitoring/logger.py` creates `FileHandler(LOG_DIR / f"{name}.log", mode="w")`
and `get_logger()` is called at **module import time** in every module. So merely
`import`ing a module **truncates its log file**. Reading the codebase destroys
the codebase's audit trail.

The audit called logging discipline a genuine strength — *"Every module writes
structured, readable logs. This is what made the audit possible."* That artifact
is destroyed by any import.

**Three incidents in this remediation:**

| # | When | What was lost |
|---|---|---|
| 1 | Phase 0.5 verification import loop | 9 original pipeline logs, 867 lines — restored from git |
| 2 | Tier 1.4 test run (`pytest` imports `concept_drift`) | `concept_drift.log` truncated — restored from git |
| 3 | Tier 1.4 restore attempt | `git checkout -- outputs/log/` reverted the **regenerated report** along with the logs, so a before/after comparison silently compared the original file against itself. Caught only because the "after" still showed 8 evidence keys |

Incident 3 is the serious one: the defect did not just destroy evidence, it
produced a **wrong analytical result that looked right**. Restores are currently
scoped to `*.log` by hand, which is a workaround, not a fix.

**Fix:**
- `mode="a"` with rotation, or a run-scoped log filename, or defer handler
  creation until first write
- never truncate on import
- CI check: importing every module must leave `outputs/log/` byte-identical

---

## P1 — Fixed-path artifact overwrite

Detectors write to fixed paths and silently clobber prior results:

| Module | Path | Incident |
|---|---|---|
| `alerting.py` | `outputs/alerts/alert_report_{ref}_{prod}.json` | would have destroyed the original entry-cohort alert artifacts during the Tier 0 sweep; worked around by monkey-patching `ALERTS_DIR` to a scratch directory |
| `concept_drift.py` | `outputs/log/concept_drift_{ref}_{prod}.json` | overwritten during Tier 1.5 verification; original recovered from git into `outputs/reports/superseded/` |
| `data_drift.py` | `outputs/log/data_drift_{prod}.csv` | same pattern |

**Fix:** output path becomes a parameter with a run-scoped default; refuse to
overwrite an existing artifact unless explicitly told to; remove the Tier 0
scratch-directory monkey-patch in `src/investigation/split_regimes.py` once done.

---

## P1 — Pickle fragility (three distinct symptoms)

1. **Version skew.** `lgbm_v1.pkl` raises `InconsistentVersionWarning` on every
   load (pickled under scikit-learn 1.7.2, loaded under 1.9.0).
2. **`__main__`-bound class resolution.** The isotonic calibrator was pickled
   from a `__main__` script, so unpickling resolves `IsotonicCalibrator` against
   whichever module is currently `__main__`. A function-local import fails;
   only a module-level import in the entry-point module works. This cost a
   debugging cycle in Tier 1.3.
3. **No schema or version metadata** on any artifact — nothing records which
   code version produced it.

**Fix:** joblib or ONNX for models; explicit version + schema metadata written
alongside every artifact; a loader that fails loudly on version mismatch rather
than warning.

---

## P2 — `.gitignore`

`__pycache__/` and generated outputs are untracked but not ignored, so
`git status` is noisy and it is easy to commit stale artifacts.
