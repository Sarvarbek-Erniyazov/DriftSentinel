# README restructure — required content

Running list of things the final README **must** state. Added as they are
discovered, so nothing found mid-remediation is lost by the time the README is
rewritten.

---

## R-1 — Adversarial perturbation vs schema violation (Tier 1.2)

State explicitly and prominently, in the robustness section:

> The defense system's apparent detection power came from a **schema-validity
> check, not adversarial detection.** Before repair, Layer 1's violation count
> separated attacked from clean inputs at AUC 0.94 — but only because the attack
> adds continuous noise to **binary** columns. What was being "detected" was a
> binary feature holding a non-integer value: a data-type violation. It implies
> nothing about robustness to an adversary who respects the schema. After
> repairing the degenerate bounds, the same layer separates at **AUC 0.500**, and
> the full five-layer combined score reaches **AUC 0.651**, with detection of
> **0.129 at a 5% false-positive rate**. The score actually used for the verdict —
> the one kept layer — reaches **AUC 0.5654** and detection **0.064** at the same
> 5% FPR. Quote whichever is relevant, but never mix them.

> **Correction applied 2026-08-04 (Tier 2C.3).** This requirement previously read
> "the full five-layer system reaches 0.617, with detection of 0.064 at a 5%
> false-positive rate." Both halves were wrong: 0.617 was a stale value from
> before the Tier 2A.1 target switch, and 0.064 belongs to the *kept-layer* ROC,
> not the five-layer one. The numbers came from hand-typed prose embedded in
> `defense.py`'s generated report, which survived a regeneration that changed
> every computed value around it. The prose is now interpolated from the computed
> values, so it cannot go stale again. This is the same defect class as
> `TARGET_COLS` (R-6): output that looked right for reasons unrelated to being
> right. Superseded artifact:
> `outputs/reports/superseded/auto/defense_report_lgbm_v1.20260804-181116.json`.

Why it must be prominent: this is the distinction between a metric that happened
to look good and a claim that survives inspection. A reviewer who spots it
unaided concludes the number was never understood.

Supporting evidence: `outputs/log/defense_report_lgbm_v1.json` →
`findings.F2_the_original_detection_was_a_type_violation_artifact`.

Related, same section:
- **L4 is not a dead layer, it is an unreachable threshold.** Its flag fires on
  0.001 of clean and 0.003 of attacked input, but its continuous statistic still
  separates at **AUC 0.582**. Audit F16 called it dead; that was wrong in a way
  that matters, because "dead" implies delete and "unreachable threshold" implies
  repair. *(Corrected 2026-08-04: this previously read "the most informative of
  the five (AUC 0.602)". Under the 30-day target the most informative single
  layer is L5 EnsembleAgreement at AUC 0.638; L4 is 0.582. The point about
  threshold vs. layer is unchanged — the superlative was stale.)*
- **L1's 93.6% clean-trigger rate was structural, not a tuning error.** 20 of 53
  features are binary or zero-inflated, so Q1 == Q3 and the IQR rule collapses to
  a single point, flagging everything at *every* multiplier. The audit's
  prescribed fix (recalibrate the multiplier) cannot work without repairing the
  degenerate bounds first.

---

## R-2 — The original split assumption was untested, not correct (Tier 0)

Already applied to CLAUDE.md; must also appear in the README's scientific-story
section. "Right by accident" is a process finding, not a retroactive
justification, and must not be softened.

---

## R-3 — Every detection rate carries its false-positive rate (Tier 1.2)

No detection number appears in the README without the FPR it was measured at.
A detection rate alone is unfalsifiable: 100% detection is available to any
system willing to flag everything, which is what the shipped configuration did.

---

## R-4 — The seven-stage selection claim is WITHDRAWN (Tier 2A.5)

State explicitly, not by quiet omission:

> The "7-stage feature selection pipeline" is **a two-stage variance/correlation
> filter with five decorative stages.** Under repeated patient-grouped CV with
> selection refitted inside every fold, stages 1–2 alone produced **byte-identical
> output to the full pipeline in all 10 folds** (Jaccard 1.0, AUC difference
> exactly 0.00000, CI [0, 0]). Stages 3–7 change nothing.
>
> Corroborating: a 4× change in target prevalence (46.1% → 11.2%) left the 53
> selected features **identical**, because the consensus vote is carried by
> target-independent stages. Per-stage counts under both targets:
> Boruta 42→1, SHAP 1→1, Stability 1→1, Consensus 78→53.

## R-5 — HEADLINE: no selection at all beats the shipped selector

> Doing **no feature selection whatsoever** outperforms the shipped 7-stage
> pipeline by **+0.0054 AUC** (CI [+0.0041, +0.0068]), and a simple in-fold
> target-aware selector beats it by **+0.0048** (CI [+0.0040, +0.0057]).
> The selection pipeline does not merely add nothing — **it costs accuracy while
> claiming to add sophistication.**

Evidence: `outputs/reports/selection_ablation.json`, `selection_ablation_folds.csv`.
Also record the leakage path it corrected: `pipeline.py` fits the selector once on
the full training set and `trainer.py` then cross-validates over those features,
so every fold's held-out data helped choose the features it was scored on.

## R-6 — Engineering findings: six defects that "worked by luck, not by design"

All belong in the engineering section, together, in this order:

1. **`TARGET_COLS` hash randomisation.** `TARGET_COLS` is a `set`, and Python
   randomises string hashing per process, so the output **column order changed on
   every run** — identical values, different bytes. No seed could have caught it;
   it was in serialisation, not modelling. Dangerous because the adversarial
   modules index positionally (`X[:, i]`), so an artifact fitted in one run and
   applied to another run's data would have read the wrong column, silently.

2. **The Boruta empty-candidate crash.** Selector stages 5–6 consume Boruta's
   confirmed set. Boruta confirms **exactly one** feature on the full training
   data and **zero** in some CV folds, where `GradientBoostingClassifier.fit`
   raises `ValueError: at least one array or dtype is required`. The shipped
   pipeline was **one feature away from a hard failure** — it survived only
   because Boruta happened to confirm one rather than none.

3. **A third instance, found by the CI check built to look for the first one
   (Tier 2C.4).** `_stage2_correlation` returned `list(to_drop)` where `to_drop`
   is a `set`, so the dropped-feature list was ordered by string hash and
   `pipeline_objects.pkl` came out **byte-different on every run** while every
   value inside it stayed identical. It was benign *only* because the sibling
   line builds `selected` by filtering the ordered candidate list rather than by
   iterating the set — luck again, one line away from the same hazard.

   State the sequence, because it is the point: the determinism check **failed on
   its first run**, named the artifact, the defect was fixed, and it now passes.
   A check that had never failed would have been indistinguishable from a check
   that could not fail.

4. **A fourth, in the registry (Tier 2C.5).** `registry.py` hardcoded
   `n_estimators: 173` — the tree count from *before* the Tier 2A.1 target
   switch — so the registry's record of the **active production model** disagreed
   with the model it described (actual: 126). A registry exists to make
   provenance checkable; one whose provenance is typed rather than read can
   contradict its own artifact. Now read from `training_summary.json`.
   Predecessor preserved in
   `outputs/reports/superseded/tier2c5_registry_hardcoded_params/`.

5. **A fifth, in `defense.py` — and this one reached the countermeasure
   (Tier 2C.3).** The generated defense report's `findings` block was **prose
   containing hand-typed numbers**. The Tier 2A.1 target switch changed the
   model, the threshold and every computed value around them; the prose did not
   move. F1 said "23 of 53" where the calibration computed **20**. F3 said L4's
   statistic reached AUC **0.602** where it computed **0.582** — and called it
   the most informative of the five layers, which it is not. F4 said the
   five-layer system reached AUC **0.617** at **0.064** detection: stale (0.651)
   *and* a conflation, because 0.064 belongs to the kept-layer ROC, not the
   five-layer one (0.129).

   **This document then quoted those numbers verbatim as must-ship text.** R-1
   above shipped `0.617` and `0.064` as a requirement, and the L4 bullet shipped
   the backwards superlative. So the error propagated into the requirements file
   that exists to prevent errors — **the pattern contaminated its own
   countermeasure.** A reviewer opening the artifact R-1 points at would have
   found it contradicting R-1.

   Fixed by interpolating every number in that prose from the values computed in
   the same function, so a regeneration cannot leave it behind. R-1 and the L4
   bullet are corrected, dated, and carry what they previously said.

6. **A sixth, and the largest: seven hardcoded absolute paths (Tier 2C.6).**
   `loader.py`, `preprocessor.py`, `splitter.py`, `consistency.py`,
   `engineer.py`, `selector.py` and `logger.py` each pinned a path to one
   developer's machine — including the **raw-data directory** and the **artifact
   directory**. The global acceptance criterion *"`pipeline.py` reproduces
   everything from raw data"* was therefore **false on every machine but one**.
   On Linux CI the same literal resolves to a *relative* directory whose name
   contains backslashes, so a run would appear to succeed while writing nowhere
   useful. It was invisible precisely because the machine it was written on is
   the machine it ran on.

   Caught by the Tier 2C.4 determinism sandbox: artifacts were landing in the
   real repository instead of the sandbox. Fixing it took the determinism
   check's coverage from **12 artifacts to 24**, and the sandbox is now
   verifiably hermetic — zero repository files touched by a run.

Framing: none was a modelling error. All six are cases where the pipeline
produced correct-looking output for reasons unrelated to being correct — and in
four of the six, a value that was *asserted* sat next to a value that was
*measured*, with nothing forcing them to agree.

The fifth is the one to state plainly, because it is the sharpest instance of
the pattern in the repository: **a defect of this class propagated into the
document written to stop defects of this class.** A requirements file is an
artifact too, and nothing was checking this one against the evidence it cited.

---

## R-7 — HEADLINE RESULT: the age disparity (Tier 2C.1, revised Tier 2C.6)

**Goes in the README's headline results, NOT in a fairness section a reviewer
has to go looking for.** This framing survives verbatim:

> **The model is best at the group that needs it least.** Highest-risk patients
> (70+, 9.6% prevalence) get the worst discrimination — AUC **0.6257**
> [0.6064, 0.6462] against **0.7461** [0.681, 0.8074] for the under-40s, an
> interval-separated gap of **+0.1204**.

Measured at the decontaminated operating threshold **0.175**:

| Age band | n | prevalence | AUC [95% CI] | recall [95% CI] | PPR [95% CI] |
|---|---|---|---|---|---|
| <40 | 931 | 0.0763 | **0.7461** [0.6810, 0.8074] | 0.3521 [0.2037, 0.4935] | 0.1214 [0.0935, 0.1513] |
| 40–69 | 8,557 | 0.0820 | 0.6369 [0.6144, 0.6598] | 0.1966 [0.1664, 0.2264] | 0.0802 [0.0740, 0.0867] |
| **70+** | 7,837 | **0.0962** | **0.6257** [0.6064, 0.6462] | 0.1804 [0.1530, 0.2078] | 0.0916 [0.0849, 0.0987] |

**Supported** (non-overlapping intervals): AUC **+0.1204** (<40 vs 70+);
predicted-positive rate **+0.0412** (<40 vs 40–69).

**WITHDRAWN at the corrected threshold — state this, do not quietly drop it.**
The recall disparity was previously reported as supported at **+0.1796**. At the
decontaminated threshold the gap is **+0.1717** — barely smaller — but the
under-40 interval widens to [0.2037, 0.4935] and now overlaps 70+'s
[0.1530, 0.2078]. **The gap did not vanish; the evidence for it did.** With 931
patients and 71 positives, that cohort cannot support a recall claim of this
size, and reporting it as supported was an artifact of the operating point.

Two earlier phrasings are retired and recorded here rather than deleted:

| Was | Now |
|---|---|
| "flagged at less than half the rate of the under-40s" (0.0426 vs 0.0956) | **Not true at 0.175.** 70+ PPR is 0.0916 against 0.1214 — about three quarters, not under half. The interval-separated PPR gap is <40 vs **40–69**, and 70+ is now flagged *more* than 40–69 |
| recall gap **+0.1796**, supported | **+0.1717, NOT supported** — intervals overlap |

Evidence: `outputs/reports/fairness_audit.json`;
before/after in `outputs/reports/threshold_reconciliation.json` →
`fairness_claims`, and `docs/10_threshold_reconciliation.md`.
Predecessor preserved at
`outputs/reports/superseded/tier2c6_contaminated_threshold/fairness_audit.at_thr_0.18.json`.

**The point of this entry is not the surviving number, it is the arithmetic of
withdrawal.** A claim that was shipped as supported is now shipped as not
supported, because the threshold it was measured at was itself corrected. That
is the process working. The AUC disparity — the one that matters clinically,
and the one the headline rests on — is unchanged at **+0.1204** and remains
interval-separated.

This is a concrete, quantified equity problem in a clinical model, **found by
our own audit** — a stronger position than not having looked.

## R-8 — The race null and its caveat (verbatim, model card)

This sentence ships attached to the result, never separated from it:

> **"Not supported" is not "no disparity exists."** AfricanAmerican AUC is 0.035
> lower than Caucasian (0.6133 vs 0.6478), but the intervals overlap, so no
> racial disparity in discrimination is claimed. This cohort is **underpowered
> to rule out** a gap of that size. An underpowered null must ship with its
> limitation attached or it reads as reassurance.

The three levels marked `INSUFFICIENT_EVIDENCE` — Asian (n=209), Hispanic
(n=496), Other (n=445) — are reported as such. **Never dropped, never averaged
away.**

## R-9 — Calibration by recorded race (its own line)

> The model is **worst-calibrated for patients whose race was never recorded**
> (ECE 0.0343 vs 0.0071 for Caucasian; gap 0.0272, intervals non-overlapping).
> This is a data-quality problem and an equity problem simultaneously.

## R-10 — Out-of-scope use (model card, explicit)

> **Not for use in age-stratified triage without recalibration**, given the
> measured AUC/recall gap across age bands.

## R-11 — Contribution framing (literature positioning)

Drift detection on UCI Diabetes 130-US is **not novel**, and readmission
prediction on it is well-trodden (Strack et al. 2014 and successors). The
contribution is **the negative-control methodology and the systematic
falsification design** — every phase carrying a condition under which the
method MUST fire, which is what makes three nulls interpretable rather than
ambiguous. Cite Obermeyer et al. (Science, 2019) in the fairness discussion:
the audit raised it, and there is now a measured answer (`payer_code` rank
17/53 under the 30-day target).

---

## R-12 — The alert-budget constraint never binds (Tier 1.3 → Tier 3)

Discovered while adding a caption to the console's budget selector, and it is a
stronger result than the fix it came from:

> **On the corrected model the alert-budget cap is redundant.** At every budget
> offered — PPR ≤ 10%, ≤ 20%, ≤ 30% — the cost-optimal threshold already sits
> *below* the cap (PPR 0.056, 0.135, 0.135 respectively), so the constraint
> never actually binds. Raising the cap changes nothing.

**Why this strengthens the Tier 1.3 story, and the connection must be stated
explicitly.** The audit (F5) prescribed an alert-budget constraint because the
shipped "cost-optimal" threshold flagged **97% of patients** — the constraint
existed to rescue a degenerate optimiser. Tier 1.3 found the degeneracy had a
different cause: the cost function normalised each error type by its own class
size, inflating the requested 5:1 FN:FP ratio into an effective **42.3:1** at
the 30-day target's 11.2% prevalence.

Correcting that to a population-weighted expected cost did not merely move the
threshold from degenerate to feasible (PPR **99.4% → 13.5%**). **It moved it far
enough that the safety rail the audit asked for is no longer load-bearing.** The
budget cap is retained as a guard-rail and as a reporting discipline, not because
it is doing work.

Sequence to state in the README:

| | threshold | PPR | binds? |
|---|---|---|---|
| Shipped (rate-form cost) | 0.0100 | **99.4%** | degenerate — cap essential |
| Corrected (population-weighted) | 0.1676 | **13.5%** | cap redundant at every budget |

The general lesson: a constraint added to contain a symptom can become
unnecessary once the cause is fixed — and noticing that the rail no longer
carries load is part of the fix, not a footnote to it.

Evidence: `outputs/reports/threshold_policy_lgbm_v1.json` → `budget_constrained`,
`corrected_recommendation_population_form`, `shipped_recommendation_rate_form`.

---

## R-13 — LEAD the literature section with the split-regime result, not the table

The comparison table is context. **The split-regime contrast is the argument**,
and it goes first:

> Hold the model, the features, the code and the seed fixed. Move only the split
> regime:
>
> | Regime | AUC | Δ vs deployed |
> |---|---|---|
> | Patient-grouped CV | 0.6806 [0.6759, 0.6851] | **+0.0396** |
> | Random patient split (20 seeds) | 0.6785 ± 0.0067 | **+0.0376** |
> | Entry-cohort held-out window | 0.6410 [0.6255, 0.6556] | — |
>
> **Nothing about the model changed.** A ~0.04 AUC swing is produced entirely by
> how the data was divided — larger than the ~0.024 spread between the best and
> worst *tree-based* model across the published comparison studies on this
> dataset. A single AUC reported without naming its split regime is not
> comparable across papers, and this is the strongest quantitative claim the
> project makes.

Only *then* the baseline table. State the strength of each claim explicitly:
the within-repository contrast is controlled (one variable moves); the
between-paper observation is not, and is offered as consistent with it, never
as evidence for it.

Evidence: `outputs/reports/literature_baselines.json` → `contrast`.

## R-14 — Strack et al. reports no discrimination metric at all

One line, and worth the space:

> **Strack et al. (2014) reports no discrimination metric.** It is an
> *etiologic* study — multivariable logistic regression estimating the
> association between HbA1c measurement and early readmission, reporting odds
> ratios. It is routinely cited as a predictive baseline on this dataset. It is
> not one, and this project does not cite it as one.

It remains the correct citation for the dataset's provenance and for the
`<30` / `>30` / `NO` encoding. The precision is the point: declining to borrow
authority from a paper that never made the claim is the kind of detail a
specialist notices.

## R-15 — The traceability audit checks for the file, not the function

In the engineering section, alongside the MLflow decision:

> `src/monitoring/model_io.py` writes provenance sidecars — package versions, a
> SHA-256 content hash, `ArtifactVersionError` raised loudly on version skew. It
> is unit-tested. It has **zero callers anywhere in `src/`**. Models still ship
> via a bare `pickle.dump`, so **0 of 4** carry provenance.
>
> The audit therefore checks for the sidecar **file**, not for the existence of
> the sidecar-writing function. A check on the function would have passed while
> the property was false — which is the whole of R6 in one line. **A capability
> nothing calls protects nothing**, and this is the same criticism the project
> levels at conformal prediction that no component consumes.

State it as a design decision, not as an apology: the audit was built to be able
to return this answer, and it did.
