# DriftSentinel — Remediation Rules

This repository is **complete and runs end to end**. This session is not a
new build. It is a **remediation and hardening pass** driven by an
adversarial audit (`docs/AUDIT.md`), ahead of review by world-class ML
researchers at KAIST / GIST / DGIST / UNIST for an InnoCORE Global
Postdoctoral Fellowship application.

**Phased plan: `docs/REMEDIATION_PLAN.md`. Read it before starting any phase.**

Assume the reviewer is hostile, knows the UCI Diabetes 130-US dataset, and
will open the JSON artifacts to check the README's claims.

---

## THE SCIENTIFIC QUESTION — RESOLVED IN TIER 0 (was: open)

**Status: resolved. Do not re-litigate; extend.**
Evidence: `outputs/reports/temporal_validity.json`, `regime_random.json`,
`regime_synthetic.json`, `regime_matrix.csv`.

| Question | Verdict |
|---|---|
| Is `encounter_id` ordering chronological? | **SUPPORTED.** Verified against three external anchors — troglitazone withdrawal (2000-03-21, all 3 uses in the lowest 0.12% of ranks, p=1.7e-09), ICD-9 V85 introduction (2005-10-01, zero occurrences before rank 0.535, p=2.9e-85), rosiglitazone safety changepoint (2007-05-21, within-class share 0.511→0.243, p=1e-04). The anchor-derived calendar map reproduces the dataset's own 30-day boundary: median implied gap 17.9 days for `<30` vs 173.5 for `>30` |
| Is the **split** temporal? | **NO — entry-cohort split.** True even under verified chronology: sorting *patients* by *first* encounter puts an early entrant's later encounters in train |
| Is the label shift concept drift? | **NO.** It tracks observation-window truncation. `readmitted` is essentially an in-extract successor indicator (0.89% of `NO` rows have any later encounter); the final-observed-encounter share rises 0.683→0.826 val→test, and the `<30` gradient **reverses sign** (−0.0098 → +0.0311) once final encounters are excluded |
| Do the detectors fire without drift? | **YES.** 2.15/8 signals on a random split where drift is impossible; `cusum_alarm` and `ph_alarm` are structurally broken, not merely mis-calibrated |

**The original "temporal split" assumption was UNTESTED, not correct.** It
happened to align with a fact nobody had checked. That is a process failure,
and it is recorded as one — being accidentally right is not a defence, and this
framing must not be softened into "it turned out to be correct all along."

The original claims, for the record, were:

- `encounter_id` **is not a timestamp**. This dataset has no date column.
  The validator itself reports `encounter_id_monotonic = False`.
- The split ranges overlap almost entirely (train 12,522–443,847,176;
  test 241,367,706–443,867,222). This is a split by **patient entry
  cohort**, not by time.
- The label is **right-censored**: `readmitted` is observable only if the
  patient returns inside the collection window. Late-entering patients are
  mechanically more likely to be labelled `NO`.
- Corroborating: `number_inpatient` falls 0.723 → 0.598 → 0.359 across
  train/val/test. That is a *history* variable. Truncation compresses both
  past and future for late entrants.

**Mechanism claims require mechanism evidence.** Do not write explanatory prose
such as "hospital billing codes changed" or "patient demographics shifted over
time" anywhere in code, logs, docs or README. Tier 0 evidenced *chronology*, not
*causes*, and the causal claims it did test came back as observability artifacts.
These strings were removed in Phase 0.5 and must not reappear.

**Naming rule (Phase 0.5, surgical).** "Temporal" is earned ONLY where it refers
to `encounter_id` chronology — which Tier 0 verified — or to the `temporal`
regime in `src/investigation/split_regimes.py`. Everywhere it describes the
*split*, the correct term is **entry-cohort split**. A blanket rename would
delete a true statement; a blanket keep would preserve a false one.

**Two stories, both required.** Story A (the spine): initial temporal framing →
falsification → discovery of entry-cohort + observability confound → re-diagnosis.
Story B (additive): with chronology now evidenced, a genuine temporal-drift study
is possible on this dataset. Story B must never be allowed to overwrite Story A —
the self-correction arc is the credibility anchor.

---

## Non-negotiable reporting rules

**R1 — No single-number claims.** Every headline metric is reported as
mean ± std over ≥20 repeated splits/seeds, with a bootstrap 95% CI. If a
number cannot be repeated, state why and mark it single-run explicitly.

**R2 — Train metrics are never labelled as validation.** Every metrics dict
uses explicit keys (`train_*`, `val_*`, `test_*`) validated against a schema.
`registry.py` must raise if two split metric dicts are identical, or if a
model's evaluation split intersects its training split. This was a real bug:
`val_metrics = train_v2_metrics`.

**R3 — No headline metric that conceals degenerate behaviour.** Specifically:
- Any detection-rate number is reported **with the full confusion matrix**
  (predicted class × actual class), never as a single cell. The defense
  system's real behaviour was 941/1000 clean samples flagged SUSPICIOUS
  while the README reported "false positive rate 0.014".
- Any "optimal" threshold reports **predicted-positive-rate** alongside
  precision / recall / F1. A threshold that flags 97% of patients is
  degenerate, not optimal. Auto-flag if PPR > 0.60.
- Any coverage number states whether it was measured **in-sample**
  (on the calibration set) or held out. Calibration-set coverage is
  guaranteed by construction and is not evidence.

**R4 — Traceability.** Every number in the README maps to a named file in
`outputs/reports/` and to a named split window. No hand-typed results. If a
figure or table cannot be regenerated by a script, it does not ship.

**R5 — Independence claims require independence.** Do not describe
correlated signals as independent. `auc_drop`, `f1_drop`, `brier_increase`
and `auc_slope_negative` are four views of one degradation.

**R6 — Verify the property, not a side effect of it.** Every acceptance
criterion must be checked by an observable that is FALSE when the property is
false. If the check would pass either way, it is not a check.

This failure has now occurred three times in this remediation, each time
producing a confident wrong answer rather than an error:

| Claimed property | What was actually checked | Why it could not fail |
|---|---|---|
| "the defense system's detection rate is 1.000" | one cell of the confusion matrix | 94% of clean traffic was flagged SUSPICIOUS — the cell was true and the claim was false |
| "the adversarial health check passes" | `def_report.get("attacked", {}).get("detection_rate", 0)` | the key had been removed; `.get` returned the default and the check reported `0.000` as if measured |
| "all figures regenerate headlessly" | files appear in `outputs/figure/` | files appear under an interactive backend too — the observable does not distinguish the two |

Applying R6 in practice:
- assert the mechanism (`matplotlib.get_backend() == "agg"`), not the artifact
- a missing input must raise or report UNKNOWN, never fall back to a default
  that is indistinguishable from a real measurement
- before writing a check, ask: *if this property were false, what would this
  assertion do?* If the answer is "still pass", rewrite it
- prefer static/structural assertions for structural claims — they cannot be
  satisfied accidentally by a side effect

Related precedent: `git checkout -- outputs/log/` silently reverted a
regenerated report, so a before/after comparison compared a file against
itself and appeared to show "no change". The comparison was real; the inputs
were not what they were assumed to be. Verify inputs, not just outputs.

---

## Anti-pattern rules from the audit

- **No quality gate may be hardcoded.** `pipeline_ready = True` was written
  over a check reporting 12 FAIL. Gates evaluate expressions and log their
  reasoning; expected failures are separated from unexpected ones by name.
- **No no-op stages presented as real ones.** Selection stages 5 and 6
  operated on a single feature and removed nothing, while the README
  advertised a seven-stage pipeline. Either repair or re-describe.
- **Multiple testing gets corrected.** ~265 hypothesis tests are run across
  features and windows with no correction. Apply Benjamini–Hochberg FDR and
  report raw and adjusted p-values.
- **Uncertainty must gate a decision.** Conformal prediction that no
  downstream component consumes is decorative. If nothing acts on it,
  either wire it into a decision or remove it.
- **Attacks must be valid for the model class.** Finite-difference FGSM/PGD
  on a piecewise-constant tree ensemble produces a zero gradient almost
  everywhere. ASR ≈ 0 means the attack did not execute, not that the model
  is robust.
- **Every experiment needs a negative control.** A detector that has never
  been shown to stay silent under no-drift is unfalsifiable.

---

## Narrative honesty

The censoring finding is **not** a limitations footnote. It is the
scientific spine of the corrected repository:

> initial temporal-drift framing → falsification testing → discovery of the
> observation-window confound → rigorous re-diagnosis under multiple split
> regimes → corrected, defensible findings

Write code, logs, docs and README so this arc is visible. A reviewer should
conclude "this person audits their own work," not "this person got lucky."

Never delete evidence of the original framing. Preserve superseded results
under `outputs/reports/superseded/` and reference them.

---

## Engineering conventions (unchanged from the original build)

- Every module importable, with a `run_*()` entry point and `__main__` block
- Type hints and docstrings stating WHAT and WHY
- No magic numbers — everything in `configs/*.yaml`
- Structured logging to `outputs/log/<module>.log`
- Seed everything; record package versions with every artifact
- If uncertain about a scientific claim, say so in the comment

Rules in markdown are context, not enforcement. Anything that must hold is
encoded as an assertion, a test, or a CI check — not as prose here.

---

## Workflow

Work phase by phase in the order given in `docs/REMEDIATION_PLAN.md`.
**After each phase, print a summary block and stop for review.** Do not
begin the next phase until told to proceed.

Phase definition of done = the acceptance criteria for that phase. Print
each criterion with your evidence before declaring a phase complete.

Tier 0 gates everything. Do not touch Tier 2 or Tier 3 until the split
question is resolved with evidence, because the answer determines what
every downstream number means.
