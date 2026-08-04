<!-- superseded evidence -->

# Superseded: headline metrics computed at the contaminated threshold

**Preserved 2026-08-04, Tier 2C.6.**

## What this file is

`headline_metrics_ci.json` as produced by Tier 2A.2, before the operating
threshold was reconciled to the decontaminated value.

## What was wrong with it

The file declares itself canonical:

> "R4 — this file is the canonical source for every number in the README. A
> number not present here does not ship."

and computed every threshold-dependent metric at **0.15334** — the F1-max
threshold fitted **on val**.

Tier 2A.4 had already established why that value cannot be used: **val is both
the threshold-selection set and the drift reference window.** The reference
window therefore carried a threshold tuned to itself while the production window
did not, and the reported val→test F1 drop was inflated by **0.0641** of pure
threshold optimism. Tier 2A.4 produced the replacement — **0.18**, selected on a
held-out, patient-level slice of *train* — and `fairness_audit.json` adopted it.
`headline_metrics_ci.json` did not, because it was generated in 2A.2, before the
decontamination existed.

So the repository held two operating thresholds at once, and the file a reviewer
is told to trust held the one a whole phase was spent proving wrong.

## Scope of the change

| | |
|---|---|
| **Unaffected** | AUC, Brier, prevalence — threshold-free |
| **Affected** | precision, recall, F1, predicted-positive rate |

The before/after comparison for every moved number is in
`outputs/reports/threshold_reconciliation.json` and in
`docs/10_threshold_reconciliation.md`.

## What replaced it

`headline_metrics_ci.json`, regenerated after `src/models/repeated_eval.py` was
changed to read the threshold from
`decontamination.json → threshold/decontaminated_selected_on_train_holdout`.

The loader **raises** if that artifact is absent rather than falling back to
`evaluation_report.json`. A fallback would be indistinguishable from a real
measurement and would silently restore the exact contamination the change
removes (CLAUDE.md R6).
