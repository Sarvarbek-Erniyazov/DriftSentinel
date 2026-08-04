# Superseded: Tier 0 regime study, 8-signal evidence count

Produced before Tier 1.4/1.5. Replaced because:

* **Tier 1.5** — `cusum_alarm` and `ph_alarm` were retired from the evidence
  count (8 -> 6 voting signals) as structurally broken. The evidence for that
  decision came from THIS study, so it is preserved rather than deleted.
* **Tier 1.4** — `label_drift` changed from a fixed absolute threshold
  (|delta| > 0.05) to a two-proportion test plus a 10% relative-effect floor.
  In these files `label_drift` reads 0.00 everywhere under the `<30` target
  because the old rule required a ~45% relative shift to fire.

Replaced by `outputs/reports/regime_*.json` / `regime_matrix.csv`.
