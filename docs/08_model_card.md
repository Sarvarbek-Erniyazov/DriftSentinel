# Model Card — `lgbm_v1`

← [Back to README](../README.md)

Format follows **Mitchell et al. (2019), *Model Cards for Model Reporting*, FAT\*.**

> **Status: research artifact. NOT approved for clinical use.** This card exists
> so that the model's failure modes are legible, not to certify it as deployable.
> See §7 (Out-of-scope use) — one restriction there is load-bearing.

Every number below is read from a named artifact. Traceability index: §12.

---

## 1. Model details

| | |
|---|---|
| **Name / version** | `lgbm_v1` |
| **Type** | LightGBM gradient-boosted tree ensemble, binary classification |
| **Trained by** | DriftSentinel, remediation pass 2026 |
| **Features** | 53, selected from 78 available (29 engineered `FE_*` + 24 raw) |
| **Trees** | 126 (early stopping on val) |
| **Key hyperparameters** | `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=50`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`, `reg_alpha=0.1`, `reg_lambda=0.1`, `random_state=42`, `deterministic=True`, `force_row_wise=True` |
| **Calibration** | Isotonic regression, fitted on the val window |
| **Class imbalance** | No resampling and no class weighting. The base learner is trained on the natural 11.2% prevalence and the operating point is set by threshold selection instead. This is a deliberate choice: resampling distorts the probability scale that the calibration, conformal and cost-threshold layers all depend on. |
| **Environment** | Python 3.12.10 · lightgbm 4.7.0 · scikit-learn 1.9.0 · numpy 2.4.6 · pandas 2.3.3 · scipy 1.18.0 |
| **License / dataset** | UCI Diabetes 130-US Hospitals (1999–2008), Health Facts (Cerner) |

**Companion models.** `logreg_v1` (baseline, CV AUC 0.6395 ± 0.0097) and
`lgbm_v2` (retrained on train+val). `lgbm_v2` is **not** promoted: the only
valid comparison is on test, where the difference is **+0.0052 AUC**, DeLong
p = 0.171, patient-clustered bootstrap CI **[−0.0019, +0.0124]** — an interval
containing zero. Verdict recorded as `NO_SIGNIFICANT_DIFFERENCE`.

## 2. Intended use

**Intended.** A research substrate for studying **post-deployment reliability**:
drift detection under known ground truth, conformal coverage under shift,
threshold policy under an alert budget, subgroup performance auditing, and the
negative-control methodology that makes null results interpretable.

**Intended users.** ML reliability researchers and reviewers evaluating the
methodology. The subject of study is the *monitoring stack*; the readmission
model is the instrument, not the product.

**Not intended.** Any clinical decision, triage, resource allocation, or
patient-facing use. See §7.

## 3. Data, target, and the target discrepancy history

### 3.1 Data

| | Rows (encounters) | Patients | Prevalence |
|---|---|---|---|
| Full dataset | 101,766 | 71,518 | 0.1116 |
| Train | 63,492 | 42,910 | 0.1200 |
| Val | 20,949 | 14,303 | 0.1056 |
| Test | 17,325 | 14,305 | 0.0881 |

130 US hospitals and integrated delivery networks, 1999–2008. Each row is one
inpatient encounter for a patient with a diabetes diagnosis and a 1–14 day stay.
**46.2% of patients contribute more than one encounter**, so encounters are not
independent; every interval in this card is a **patient-clustered** bootstrap.

**Split: entry-cohort, not temporal.** Patients are sorted by their *first*
`encounter_id` and assigned to train/val/test in that order, with no patient
crossing a boundary. Tier 0 verified that `encounter_id` ordering *is*
chronological (three external anchors, all significant after FDR), but that does
not make the split temporal: an early-entering patient's later encounters sit
wholly in train. The prevalence gradient across splits above (0.120 → 0.106 →
0.088) is **observation-window truncation**, not falling clinical risk — late
entrants have less follow-up inside the extract in which to be observed
returning.

### 3.2 Target

```
readmitted_binary = 1  if readmitted == "<30"     (readmission within 30 days)
                  = 0  if readmitted in {">30", "NO"}
```

Prevalence **11.16%**. This is the CMS-penalised outcome, the clinical question,
and the definition used by the comparison literature (Strack et al. 2014;
Liu et al. 2024; Salim & Ibrahim 2026).

### 3.3 The target discrepancy history — stated, not buried

This is recorded because the discrepancy is more informative than the target:

| Stage | What the README said | What the code did | Prevalence |
|---|---|---|---|
| Original build | "readmission **within 30 days**" | `{"NO":0, "<30":1, ">30":1}` — merged, i.e. readmission *ever* | **46.1%** |
| Audit | flagged the mismatch | — | — |
| Tier 2A.1 | primary target switched | `{"NO":0, "<30":1, ">30":0}` | **11.16%** |

**The documented target and the implemented target disagreed for the whole
original build.** Every headline number in the original README — AUC, F1, the
"cost-optimal" threshold, the conformal set sizes, the drift magnitudes — was
computed against a 4× more prevalent target than the one advertised. The
discrepancy was never a modelling error; it was a definition that nobody had
checked against the code.

Consequences that only became visible after the switch:

- **AUC falls** from ~0.69 to ~0.64. Expected, and consistent with the published
  band for the 30-day target — the easier merged target was inflating it.
- **Feature selection did not change at all.** A 4× change in prevalence left the
  53 selected features **identical**, which is what exposed the selection
  pipeline as target-blind (R-4).
- **The cost-threshold defect became severe.** At 46.1% prevalence the rate-form
  cost bug inflated a requested 5:1 FN:FP ratio to 5.5:1; at 11.2% it inflates it
  to **42.3:1** (R-12).

The merged target is retained as a documented secondary analysis. Results
computed under it are preserved in `outputs/reports/superseded/tier2a_merged_target/`
and are never quoted as 30-day results.

## 4. Metrics and evaluation protocol

**Operating threshold: 0.175, everywhere.** Selected by F1-max on a **held-out,
patient-level slice of train**, so the drift reference window (val) carries no
fitted quantity. Every module now reads it from one place —
`decontamination.json → threshold/decontaminated_selected_on_train_holdout` —
and the loader **raises** if that artifact is missing rather than falling back to
the val-fitted value, which would silently restore the contamination (R6).

Two corrections got it here, both worth stating:

| | |
|---|---|
| **The canonical file held the contaminated value.** | `headline_metrics_ci.json` calls itself "the canonical source for every number in the README" and computed every threshold-dependent metric at **0.1533** — the F1-max threshold fitted *on val*, which is also the drift reference window. It was generated in Tier 2A.2, before the Tier 2A.4 decontamination existed, and never adopted it. Reconciled in Tier 2C.6; **12 values moved**. |
| **Tier 2A.4's own threshold block mixed three probability scales.** | It selected the threshold under one calibrator, then scored val under a second and test under a third. A threshold is a cut point on a probability scale, so that is not a like-for-like comparison. The reported threshold optimism of **0.0641** was an artifact; recomputed on one scale it is **0.0024**. **The val→test F1 degradation was largely real, not threshold optimism.** Found because the reconciliation produced a different test F1 than `decontamination.json` reported for the same threshold — the disagreement was the symptom. |

Full before/after: [Threshold Reconciliation](10_threshold_reconciliation.md).

**Intervals.** Patient-clustered bootstrap, 2,000 resamples, resampling *patients*
rather than rows. Standard row-level intervals would be anti-conservative here.

## 5. Quantitative performance

### 5.1 Headline (test window, entry-cohort split, threshold **0.175**)

| Metric | Train | Val | **Test** | Test 95% CI |
|---|---|---|---|---|
| AUC | 0.8125 | 0.6646 | **0.6410** | [0.6255, 0.6556] |
| Precision | 0.4373 | 0.2162 | **0.1971** | [0.1758, 0.2172] |
| Recall | 0.4679 | 0.2616 | **0.1958** | [0.1741, 0.2170] |
| F1 | 0.4521 | 0.2368 | **0.1965** | [0.1756, 0.2161] |
| Brier | 0.0922 | 0.0906 | **0.0785** | [0.0749, 0.0821] |
| ECE | 0.0441 | 0.0000 † | **0.0080** | [0.0053, 0.0128] |
| Predicted-positive rate | 0.1284 | 0.1278 | **0.0876** | [0.0827, 0.0923] |
| Prevalence | 0.1200 | 0.1056 | **0.0881** | [0.0834, 0.0928] |

AUC, Brier and prevalence are threshold-free and are **unchanged** by the Tier
2C.6 correction — they are used as the control that proves the two runs differ
in the threshold and nothing else. Precision, recall, F1 and PPR all moved.

† **val ECE is in-sample and is not evidence.** The isotonic calibrator was
fitted on val, so measuring calibration error on val measures the fit. The
held-out numbers are **0.0057** (val audit half) and **0.0067–0.0080** (test).

**Learner stochasticity**, 20 seeds, same split: test AUC **0.6345 ± 0.0026**.
Note this refits with `n_estimators=500` and no early stopping, whereas the
deployed model stopped at 126 trees — the seed spread characterises the learner,
not the deployed model.

**Regime dependence.** Same model, same features, same code — split regime varied:
patient-grouped CV **0.6806** [0.6759, 0.6851]; random patient split (20 seeds)
**0.6785** ± 0.0067; entry-cohort test **0.6410**. A ~0.04 AUC swing from the
split alone. See [Literature Positioning §3.1](07_literature_positioning.md).

### 5.2 What the model actually learns

The top five features by gain are `discharge_disposition_id`,
`FE_has_prior_inpatient`, `FE_lab_to_procedure_ratio`, `FE_labs_per_day`,
`FE_meds_per_day`.

**The task is partly tautological.** Prior-utilisation features dominate, so the
model substantially identifies **frequent utilisers** rather than predicting a
clinical event. This is stated as a property of the task, not a defect of the
model — but it bounds what any performance number here can mean, and it is the
mechanism by which access-to-care patterning enters the model even when
`payer_code` is demoted (§6.5).

### 5.3 Coverage, thresholds, robustness

| Claim | Value | Measured |
|---|---|---|
| Conformal coverage @ 90% target | 0.8943 | **held out** (val audit half, patient-disjoint) |
| Conformal coverage @ 90% target | 0.9134 | **held out** (test) |
| Conformal coverage @ 90% target | 0.9005 | **in-sample** (calibration half) — guaranteed by construction, not evidence |
| Cost-optimal threshold @ 5:1 | 0.1676, **PPR 0.1354** | val; not degenerate |
| Alert-budget constraint | **never binds** at any offered budget | R-12 |
| Worst realistic EHR data-quality failure | **−0.0370 AUC** (feature outage, top features) | test |
| Destructive control | **−0.0456 AUC** | confirms the harness can detect damage |

**No adversarial robustness score is reported.** The shipped score of 0.8954 is
**withdrawn**, not supplemented: 71.7% of finite-difference gradients on this
tree ensemble are exactly zero, so FGSM/PGD had no direction to step in and the
near-zero attack success rate was the null behaviour of a method that did not
execute. There is no credible adversary for a hospital readmission model. The
robustness framing is now **EHR data-quality failure**, which has a real threat
model.

**The defense system's detection power was a type-violation artifact.** Before
repair, Layer 1 separated attacked from clean input at AUC 0.94 — but only
because the attack adds continuous noise to **binary** columns, so what was being
detected was a binary feature holding a non-integer value. After repairing the
degenerate bounds that same layer separates at **AUC 0.500**; the five-layer
combined score reaches **AUC 0.651** with detection **0.129 at a 5% FPR**, and
the kept-layer score reaches **AUC 0.5654** with detection **0.064 at 5% FPR**.
There is no operating point at which the system usefully detects this attack.
**Every detection rate in this repository ships with the false-positive rate it
was measured at** — a detection rate alone is unfalsifiable, since 100% detection
is available to any system willing to flag everything, which is what the shipped
configuration did (941/1000 clean samples flagged SUSPICIOUS while the README
reported "false positive rate 0.014").

## 6. Subgroup analysis and fairness

Cohort: test split, 14,305 patients / 17,325 rows, threshold **0.175**.
Intervals are patient-clustered bootstrap, 1,000 resamples.

**Evidence rule, fixed before results were seen.** A subgroup supports a claim
only with n ≥ 500 **and** ≥ 30 positives. Below that it is marked
`INSUFFICIENT_EVIDENCE` — reported, never dropped, never averaged into a
reassuring aggregate. A disparity is claimed only when the two intervals **do not
overlap**.

**The audit can detect a disparity.** An injected disparity — one subgroup's
predictions shuffled — produced a **0.1313** AUC gap and was flagged. The real
gender gap of 0.0091 was not. The nulls below are therefore measurements, not
blind spots.

### 6.1 Age — HEADLINE RESULT

> **The model is best at the group that needs it least.** Highest-risk patients
> (70+, 9.6% prevalence) get the worst discrimination — AUC **0.6257**
> [0.6064, 0.6462] against **0.7461** [0.6810, 0.8074] for the under-40s, an
> interval-separated gap of **+0.1204**.

| Age band | n | prevalence | AUC [95% CI] | recall [95% CI] | PPR [95% CI] |
|---|---|---|---|---|---|
| <40 | 931 | 0.0763 | **0.7461** [0.6810, 0.8074] | 0.3521 [0.2037, 0.4935] | 0.1214 [0.0935, 0.1513] |
| 40–69 | 8,557 | 0.0820 | 0.6369 [0.6144, 0.6598] | 0.1966 [0.1664, 0.2264] | 0.0802 [0.0740, 0.0867] |
| **70+** | 7,837 | **0.0962** | **0.6257** [0.6064, 0.6462] | 0.1804 [0.1530, 0.2078] | 0.0916 [0.0849, 0.0987] |

Two disparities with **non-overlapping** intervals:

| Metric | Gap | Between |
|---|---|---|
| AUC | **+0.1204** | <40 vs 70+ |
| Predicted-positive rate | **+0.0412** | <40 vs 40–69 |

**One disparity was WITHDRAWN at the corrected threshold, and it is reported
rather than dropped.** Recall previously showed a supported gap of **+0.1796**.
At 0.175 the gap is **+0.1717** — barely smaller — but the under-40 interval
widens to [0.2037, 0.4935] and overlaps 70+'s [0.1530, 0.2078]. **The gap did
not vanish; the evidence for it did.** With 931 patients and 71 positives that
cohort cannot support a recall claim of this size, and reporting it as supported
was an artifact of the operating point.

Retired with it: the phrase "flagged at less than half the rate of the
under-40s". At 0.175 the 70+ PPR is 0.0916 against 0.1214 — about three
quarters, not under half — and 70+ is now flagged *more* than 40–69. The
interval-separated PPR gap runs <40 vs **40–69**.

The clinical core is unaffected: **the AUC disparity is unchanged at +0.1204 and
remains interval-separated.** Before/after:
[Threshold Reconciliation](10_threshold_reconciliation.md).

This is a concrete, quantified equity problem in a clinical model, **found by our
own audit**. That is a stronger position than not having looked. It is a headline
result, not a fairness-appendix result, because it changes what the model may be
used for (§7).

### 6.2 Race — an underpowered null, with its caveat attached

> **"Not supported" is not "no disparity exists."** AfricanAmerican AUC is 0.035
> lower than Caucasian (0.6133 vs 0.6478), but the intervals overlap, so no
> racial disparity in discrimination is claimed. This cohort is **underpowered to
> rule out** a gap of that size. An underpowered null must ship with its
> limitation attached or it reads as reassurance.

| Race | n | positives | prevalence | AUC [95% CI] | ECE |
|---|---|---|---|---|---|
| Caucasian | 13,394 | 1,205 | 0.0900 | 0.6478 [0.631, 0.665] | 0.0071 |
| AfricanAmerican | 2,181 | 194 | 0.0890 | 0.6133 [0.573, 0.654] | 0.0125 |
| **Missing** | 600 | 39 | 0.0650 | 0.5849 [0.498, 0.670] | **0.0343** |
| Asian | 209 | 19 | 0.0909 | `INSUFFICIENT_EVIDENCE` | — |
| Hispanic | 496 | 32 | 0.0645 | `INSUFFICIENT_EVIDENCE` | — |
| Other | 445 | 38 | 0.0854 | `INSUFFICIENT_EVIDENCE` | — |

**The three `INSUFFICIENT_EVIDENCE` levels are named here and everywhere they
appear: Asian (n=209, 19 positives), Hispanic (n=496, 32 positives), Other
(n=445, 38 positives).** They are never dropped from a table and never folded
into an "Other" aggregate to make the cohort look better powered than it is.

Also note the `Missing` row: its AUC interval **[0.498, 0.670] contains 0.5**, so
discrimination for patients with no recorded race is not distinguishable from
chance on this cohort.

### 6.3 Calibration by recorded race — its own line

> The model is **worst-calibrated for patients whose race was never recorded**
> (ECE **0.0343** vs **0.0071** for Caucasian; gap **0.0272**, intervals
> non-overlapping). This is a data-quality problem and an equity problem
> simultaneously.

It is the **only** supported disparity on the race attribute, and it is a
calibration disparity rather than a discrimination one — meaning the risk scores
for this group are miscalibrated in a way that would propagate directly into any
threshold-based decision.

### 6.4 Gender — a null with adequate power

| Gender | n | AUC [95% CI] | recall | PPR | ECE |
|---|---|---|---|---|---|
| Female | 9,044 | 0.6454 [0.623, 0.667] | 0.1153 | 0.0471 | 0.0096 |
| Male | 8,280 | 0.6363 [0.615, 0.657] | 0.1152 | 0.0418 | 0.0071 |

No supported disparity on any metric. `Unknown/Invalid` (n=1) is
`INSUFFICIENT_EVIDENCE`. Unlike the race null, this cohort is large enough that
the null carries some weight — the AUC gap is 0.0091 against a detection
threshold demonstrated at 0.1313.

### 6.5 `payer_code`, and the Obermeyer frame

**Obermeyer et al. (*Science*, 2019)** showed an algorithm affecting millions of
patients using *healthcare cost* as a proxy for health need, which systematically
under-referred Black patients — with race nowhere in the feature set. The lesson
is that a variable encoding **access to care** gets learned as though it encoded
**clinical severity**.

`payer_code` — insurance status — is structurally that kind of variable, and it
is both among the most drifted features (PSI 0.84–0.93 between windows) and
present in the model. **The measured answer: rank 17 of 53 by gain importance.**
Not in the top five.

This is weaker than the audit assumed — the audit was reasoning from the merged
target, where `payer_code` was a top feature. It is not a dismissal:

- Rank 17 of 53 is still inside the model and still contributing.
- The top features are prior-utilisation variables (§5.2), and utilisation is
  itself patterned by access. **The Obermeyer mechanism is not avoided by
  demoting `payer_code`; it is relocated** to features that look clinical.
- The disparity actually found is by **age**, and it is large.

**`race` is also a model input** (rank outside the top five). This is a
deliberate, contestable choice: including it makes subgroup calibration
measurable and the disparity in §6.3 visible, but it also means the model can
condition directly on race. For any use beyond research this decision would need
to be revisited explicitly rather than inherited.

## 7. Out-of-scope use

The following are excluded, and the first is derived from a measurement in this
card rather than asserted as boilerplate:

> **Not for use in age-stratified triage without recalibration**, given the
> measured AUC and recall gap across age bands (§6.1). Applying one threshold
> across age bands means the 70+ group — the highest-prevalence group — is
> flagged at less than half the rate of the under-40s while being harder for the
> model to rank at all.

Also out of scope:

- **Any clinical or patient-facing decision.** No prospective validation, no
  external validation, no deployment outcomes.
- **Resource allocation or benefit determination.** §6.5 applies directly.
- **Decisions about patients with no recorded race**, whose risk scores are the
  worst-calibrated in the cohort (§6.3) and whose discrimination is not
  distinguishable from chance.
- **Any subgroup marked `INSUFFICIENT_EVIDENCE`** — Asian, Hispanic, Other. The
  absence of a measured disparity there is an absence of measurement.
- **Transfer to another hospital system or era** without revalidation. Health
  Facts onboarded client hospitals across 1999–2008 and the UCI release drops the
  hospital identifier, so cohort composition and calendar time are confounded and
  cannot be stratified apart.
- **Use of the uncertainty layer to catch model mistakes.** The falsification arm
  showed the conformal gate cannot detect label corruption (lift −0.004) — it
  sees only *x* and *p(x)*. "Route the uncertain cases" does not mean "route the
  cases the model gets wrong."

## 8. Ethical considerations

- **Retrospective, de-identified data.** No prospective consent question arises,
  and no deployment occurred.
- **The measured disparity is age, and it runs the wrong way** — worst
  performance on the highest-risk group. In a deployed triage system this would
  systematically under-serve the elderly.
- **Access-to-care proxies are unavoidable in this feature space** (§6.5). They
  are reported, not eliminated.
- **Underpowered subgroups are a fairness finding in themselves.** Three race
  levels cannot support a claim at this cohort size, which is a statement about
  who this dataset can and cannot speak for.

## 9. Caveats and limitations

- **Single dataset.** No external validation (MIMIC-IV, eICU) — future work.
- **No verified timestamps.** `encounter_id` ordering is evidenced as
  chronological, but there is no date column and no calendar map beyond three
  anchor points. The project therefore studies **cohort shift under
  observation-window truncation**, and says so.
- **The split is an entry-cohort split, not a temporal one.** This is stated
  everywhere the word "temporal" would otherwise be unearned.
- **AUC ≈ 0.64, without apology.** This sits inside the published band for this
  task and target (0.61–0.66; see [Literature Positioning](07_literature_positioning.md)).
  The contribution is reliability infrastructure, not discrimination.
- **The label is essentially an in-extract successor indicator.** Only 0.89% of
  `NO` rows have any later encounter in the extract, and the final-observed-
  encounter share rises 0.683 → 0.826 from val to test.
- **The feature-selection pipeline is withdrawn as described.** The advertised
  seven stages are a two-stage variance/correlation filter plus five stages that
  changed nothing (Jaccard 1.0 across all 10 folds, AUC difference exactly
  0.00000). **No selection at all beats it by +0.0054 AUC** [+0.0041, +0.0068].
- **Clustered observations.** All intervals are patient-clustered; any number
  imported from elsewhere without that treatment is anti-conservative.
- **Two engineering defects that worked by luck, not design** (R-6):
  `TARGET_COLS` is a `set`, and Python randomises string hashing per process, so
  output **column order changed on every run** — identical values, different
  bytes, uncatchable by any seed because the defect was in serialisation.
  Dangerous specifically because the adversarial modules index positionally.
  And Boruta confirms **exactly one** feature on the full training data and
  **zero** in some CV folds, where the downstream fit raises — the shipped
  pipeline was **one feature away from a hard failure**.

## 10. Quantitative traceability

| Claim area | Artifact |
|---|---|
| Headline metrics, intervals, variance components | `outputs/reports/headline_metrics_ci.json` |
| Subgroup performance, disparities, falsification arm | `outputs/reports/fairness_audit.json` |
| Split validity, anchors, censoring | `outputs/reports/temporal_validity.json` |
| Regime × signal matrix | `outputs/reports/regime_matrix.json`, `regime_random.json`, `regime_synthetic.json` |
| Threshold policy, PPR, budgets, DCA | `outputs/reports/threshold_policy_lgbm_v1.json` |
| Conformal / calibration decontamination | `outputs/reports/decontamination.json` |
| Adaptive conformal | `outputs/reports/adaptive_conformal.json` |
| Multivariate drift detection | `outputs/reports/multivariate_drift.json` |
| Multiple-testing correction | `outputs/reports/fdr_correction.json`, `fdr_corrected_tests.csv` |
| Selection ablation | `outputs/reports/selection_ablation.json`, `selection_ablation_folds.csv` |
| Triage policy | `outputs/reports/triage_policy.json` |
| Data-quality robustness | `outputs/reports/data_quality_robustness.json` |
| Defense evaluation | `outputs/log/defense_report_lgbm_v1.json` |
| Model comparison (DeLong + bootstrap) | `outputs/registry/model_comparison.json` |
| Literature baselines | `outputs/reports/literature_baselines.json` |
| Superseded predecessors | `outputs/reports/superseded/` |

## 11. Datasheet pointer

Dataset provenance, collection, and preprocessing follow
**Gebru et al. (2021), *Datasheets for Datasets*** and are documented in
[EDA & Data Pipeline](01_eda_and_data_pipeline.md) and
[Feature Engineering](02_feature_engineering.md).

## 12. `README_MUST_INCLUDE` coverage

Every requirement R-1 … R-12, and where it is discharged in this card:

| # | Requirement | Section |
|---|---|---|
| **R-1** | Adversarial detection was a schema-violation artifact; AUC 0.500 after repair; five-layer 0.651 @ 0.129 detection at 5% FPR; kept-layer 0.5654 @ 0.064; L1's clean-trigger rate was structural (20 of 53 features have Q1 == Q3); L4 is an unreachable threshold (AUC 0.582), not a dead layer | §5.3 |
| **R-2** | The original split assumption was **untested, not correct** | §3.1, §9 |
| **R-3** | Every detection rate carries its false-positive rate | §5.3 |
| **R-4** | The seven-stage selection claim is **withdrawn** | §9 |
| **R-5** | **No selection at all beats the shipped selector** by +0.0054 AUC [+0.0041, +0.0068] | §9 |
| **R-6** | `TARGET_COLS` hash randomisation; the Boruta empty-candidate near-crash | §9 |
| **R-7** | **The age disparity as a headline result**, verbatim framing; the withdrawn recall disparity and the retired "less than half" phrasing stated, not dropped | §6.1 |
| **R-8** | The race null with its caveat attached verbatim; three `INSUFFICIENT_EVIDENCE` levels named | §6.2 |
| **R-9** | Missing-race calibration (ECE 0.0343 vs 0.0071) on its own line | §6.3 |
| **R-10** | Out-of-scope: **not for age-stratified triage without recalibration**, verbatim | §7 |
| **R-11** | Contribution framing; Obermeyer cited in the fairness discussion; `payer_code` rank 17/53 | §6.5, and [Literature Positioning §4, §6](07_literature_positioning.md) |
| **R-12** | The alert-budget constraint never binds | §5.3, and [Literature Positioning](07_literature_positioning.md) context |
| **R-13** | Lead the literature section with the split-regime result, not the table | [Literature Positioning §3.1](07_literature_positioning.md) |
| **R-14** | Strack et al. reports no discrimination metric and is not a predictive baseline | [Literature Positioning §2](07_literature_positioning.md) |
| **R-15** | The traceability audit checks for the sidecar file, not the function | [Experiment Tracking §4](09_experiment_tracking_decision.md) |

## 13. References

- Gebru, T., et al. (2021). Datasheets for datasets. *CACM*, 64(12), 86–92.
- Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**, 220–229.
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453.
- Strack, B., et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*, 2014, 781670.
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis. *Medical Decision Making*, 26(6), 565–574.

Full bibliography: [Literature Positioning §8](07_literature_positioning.md#8-references).
