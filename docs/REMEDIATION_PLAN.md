> ## ⚠️ RESOLUTION BANNER — read before this document
>
> **This file is a historical record and its body is deliberately NOT edited.**
> It is the remediation plan as approved. Tier 0 changed two of its instructions,
> recorded here rather than by silently rewriting the plan:
>
> - **Phase 0.5 is SURGICAL, not blanket.** The plan assumed Phase 0.1 would
>   return NOT SUPPORTED and that every use of "temporal" would be unearned. It
>   returned **SUPPORTED**, so "temporal" is kept where it refers to
>   `encounter_id` chronology and replaced only where it describes the *split*.
> - **A fourth regime was added.** Phase 0.4 column (c) ("time-proxy split if
>   Phase 0.1 justified one") was conditional; Phase 0.1 justified it, so a
>   genuine chronological split of encounters now runs as regime `temporal`.
>
> Everything below is preserved verbatim as the plan that was approved.

# DriftSentinel — Remediation Plan

Source: `docs/AUDIT.md` (30 findings, 6 blockers).
Scope: **all four tiers**, ~7–8 weeks.
Goal: move from "solid portfolio piece" to "defensible InnoCORE-grade
research artifact."

**Working rule:** after each phase, print a summary and stop for review.
Tier 0 gates everything — the split question determines what every
downstream number means.

**Preservation rule:** never delete superseded results. Move them to
`outputs/reports/superseded/` with a README noting what replaced them and
why. The audit trail is part of the contribution.

---

# TIER 0 — GROUND TRUTH AND FALSIFICATION

Everything downstream depends on whether the observed drift is real. Do not
fix bugs, modernise, or deploy before this tier is complete.

## Phase 0.1 — Is `encounter_id` correlated with calendar time?

Create `src/investigation/temporal_validity.py`.

The dataset (1999–2008, Health Facts) has no date column. Test whether
`encounter_id` ordering carries any calendar-time signal using **external
temporal anchors** — clinical practice patterns whose timing is known
independently:

1. **HbA1c measurement rate.** A1C testing rates rose substantially through
   the 2000s. If ordering is chronological, `A1Cresult` non-missing rate
   should trend upward with encounter_id.
2. **Thiazolidinedione prescribing.** Rosiglitazone prescribing collapsed
   after the May 2007 safety meta-analysis; pioglitazone was less affected.
   If ordering is chronological, the rosiglitazone rate should fall in the
   upper encounter_id range while pioglitazone is comparatively stable.
   Verify these dates from the literature before relying on them.
3. **Any other defensible anchor you can justify** — document the reasoning.

Method: bin encounters into deciles of `encounter_id`, compute each rate per
decile, fit a trend, report slope with CI and a permutation test against
random ordering.

**Also test the mechanical alternative directly.** For each patient compute
`first_encounter_id` (entry rank), `n_encounters`, `observed_readmission_rate`,
`number_inpatient`. Regress each on entry rank. Monotonic decline in all
three is the censoring signature.

**Acceptance:** `outputs/reports/temporal_validity.json` with a verdict of
SUPPORTED / NOT SUPPORTED / INCONCLUSIVE for chronological ordering, each
anchor's trend with CI, and the censoring regression. A figure per anchor.

**If NOT SUPPORTED:** rename the split to **entry-cohort split** across the
entire codebase — module docstrings, log strings, config keys, variable
names, docs, README. Do not leave the word "temporal" anywhere it is not
earned. Log every file changed.

## Phase 0.2 — Random-split falsification control

Create `src/investigation/split_regimes.py`.

Re-run the *complete* detection pipeline (data_drift, feature_drift,
concept_drift, alerting) with patients assigned to train/val/test **at
random**, preserving the same split proportions and the same patient-level
grouping.

Repeat over **≥20 random seeds** — a single random split proves nothing.

Record, per seed, which of the 8 concept-drift evidence signals fires, the
PSI-critical feature count, and the alert system status.

**Acceptance:** `outputs/reports/regime_random.json`. Report the firing rate
of each signal across seeds. Any signal firing in >20% of no-drift seeds is
mis-calibrated and must be reported as such.

**Interpretation to state explicitly:** if signals fire under random
splitting, the original "8/8 CRITICAL" result is partly detector
mis-calibration, not evidence of drift.

## Phase 0.3 — Synthetic positive controls

A detector that never fires is as useless as one that always fires. Prove
the signals *can* fire, and characterise *what they respond to*.

Starting from a random split, construct three synthetic shifts on the test
half, each isolating one mechanism:

| Control | Construction | Should fire |
|---|---|---|
| **Pure covariate shift** | Perturb feature distributions (e.g. shift `number_diagnoses` by +2, resample `payer_code`) while preserving P(Y\|X) | data-drift signals; performance signals should be mild |
| **Pure label shift** | Resample to change P(Y) by 15pp, preserving P(X\|Y) | label_drift, prediction_drift, threshold-dependent metrics |
| **Pure concept shift** | Flip the label mechanism for a defined subgroup, preserving P(X) | performance signals, CUSUM/PH; data-drift signals should stay quiet |

Sweep shift magnitude to obtain a **detection-power curve** per signal.

**Acceptance:** `outputs/reports/regime_synthetic.json`; power curves per
signal per shift type; a statement of which signals are diagnostic for which
mechanism. Any signal that fires under all three is not diagnostic and must
be described as a general alarm, not evidence of a specific mechanism.

## Phase 0.4 — The regime × signal matrix

Consolidate into one table — this becomes the centrepiece of the corrected
README.

Rows: the 8 evidence signals (grouped into independent families per CLAUDE.md
R5). Columns: (a) original entry-cohort split, (b) random control (firing
rate over seeds), (c) time-proxy split if Phase 0.1 justified one,
(d) covariate-shift control, (e) label-shift control, (f) concept-shift
control.

**Acceptance:** `outputs/reports/regime_matrix.csv` and a publication-quality
figure. Written interpretation answering: *what was the original drift
actually?* State the answer plainly even if it weakens the original claim.

## Phase 0.5 — Codebase-wide language correction

Apply the Phase 0.1 verdict everywhere. Grep for `temporal`, `over time`,
`drift began`, `newer patients`, `billing codes`, `demographics shift`.
Every occurrence is either backed by Phase 0.1–0.4 evidence or rewritten.

**Acceptance:** a diff report listing every changed string; no unearned
mechanism claim survives anywhere in the repository.

---

# TIER 1 — BLOCKING CORRECTNESS BUGS

## Phase 1.0 — Integrity sweep (added: audit finding F2)

Remove `summary["pipeline_ready"] = True`. Replace with an evaluated
expression that separates expected drift-related failures from unexpected
ones, logs the reasoning, and exposes `expected_failures` and
`unexpected_failures` as distinct fields.

Audit the whole repository for any other hardcoded pass, silenced check, or
severity mapping tuned to produce a desired output — including the
`health_check.py` CRITICAL→WARN remap, which must carry an inline
justification or be reverted.

**Acceptance:** no hardcoded gate remains; every gate logs its inputs and
decision; `grep -rn "= True" src/` reviewed and each instance justified.

## Phase 1.1 — `registry.py`: in-sample metrics labelled as validation

The bug: `val_metrics = train_v2_metrics`. lgbm_v2 was trained on train+val,
so its reported "val AUC 0.7987" is training performance, and the comparison
table declares it the winner over lgbm_v1's honest held-out 0.6865.

Fix:
- Add `assert_split_disjoint(model_train_splits, eval_split)` raising if a
  model is evaluated on data it was fitted on
- Add an assertion raising if two metric dicts for different splits are
  identical beyond floating-point tolerance
- Enforce a metrics schema with explicit `train_` / `val_` / `test_` keys
- Report the only valid comparison — test AUC 0.6560 → 0.6648 — with a
  **DeLong test** and a paired bootstrap CI. If the difference is not
  significant, say so.
- Either hold out a fourth split for honest v2 evaluation, or mark v2 val
  metrics as unavailable

Also control the confound: lgbm_v1 used early stopping (173 trees), lgbm_v2
a fixed 300 with none. Match the protocol or document the difference.

**Acceptance:** assertions in place and unit-tested; regenerated
`model_registry.json`, `registry_history.csv` and README table; the promote
verdict re-derived from valid evidence only.

## Phase 1.2 — Defense system: confusion matrix, not a single cell

The bug: README reports "detection rate 1.000, false positive rate 0.014"
while 941/1000 clean samples are flagged SUSPICIOUS.

Fix:
- Report the full 3×2 confusion matrix (CLEAN / SUSPICIOUS / ADVERSARIAL ×
  actual clean / attacked) as the primary artifact
- Report detection rate **at matched false-positive rate**, plus an ROC over
  the decision threshold. A raw detection rate without an FPR is meaningless.
- Recalibrate L1 InputValidation (IQR multiplier) against a target
  clean-trigger rate, e.g. 5%
- L3 Consistency has **negative lift** (0.280 clean vs 0.121 attacked) —
  repair or remove
- L4 Smoothing never fires — repair or remove
- Re-describe the system by the number of layers that carry signal, not the
  number implemented

**Acceptance:** regenerated `defense_report_lgbm_v1.json` with the confusion
matrix; ROC figure; README claim rewritten; removed layers documented with
the evidence that justified removal.

## Phase 1.3 — Threshold reporting: predicted-positive-rate

The bug: the "cost-optimal" threshold 0.1282 yields recall 0.995 at
precision 0.3452, implying a **96.96% predicted-positive rate**. Presented
as a headline win.

Fix:
- Every threshold recommendation reports PPR alongside precision / recall /
  F1 / cost
- Auto-flag any threshold with PPR > 0.60 as `DEGENERATE` in the report
- Add an **alert-budget constraint** to the search: maximise utility subject
  to PPR ≤ B, for B ∈ {0.10, 0.20, 0.30}
- Sweep the FN:FP cost ratio over 1:1 → 100:1 and report the feasible
  operating region; no conclusion may rest on the single asserted 5:1
- Add **decision curve analysis** (Vickers & Elkin 2006) — the standard
  clinical framing for threshold utility

**Acceptance:** regenerated `threshold_report_lgbm_v1.json` with PPR and
degeneracy flags; cost-ratio sensitivity figure; decision curve figure;
README table replaced with the constrained result.

---

# TIER 2A — STATISTICAL RIGOR

*Added beyond the original skeleton: these change every number, so they must
precede modernisation or the modernisation work is done twice.*

## Phase 2A.1 — Target definition

`readmitted_binary` currently merges `<30` and `>30` (49% prevalence). The
clinical task, the CMS-penalised outcome, and the literature standard is
**30-day readmission** (11.16% prevalence).

Make `<30` the primary target. Keep the merged target as a documented
secondary analysis. Expect AUC to fall; published 30-day models on this
dataset sit roughly in the 0.63–0.70 band.

**Acceptance:** both targets trained and reported; the choice justified in
`docs/03_model_training.md` with citations; class-imbalance handling stated.

## Phase 2A.2 — Repeated evaluation and confidence intervals

Every split-dependent number is currently from one seed. Repeat the full
pipeline over ≥20 seeds. Report mean ± std and bootstrap 95% CIs for: AUC,
F1, precision, recall, Brier, degradation deltas, PSI values, MI/SHAP rank
correlation, robustness score.

**Acceptance:** every README number carries variance; a table of
"claims that survived repetition" vs "claims that did not".

## Phase 2A.3 — Multiple testing correction

~265 hypothesis tests run across features, windows and test types with no
correction. Apply Benjamini–Hochberg FDR at q=0.05. Report raw and adjusted
p-values. Restate the "31/53 drifted" claim post-correction.

**Acceptance:** regenerated drift CSVs with `p_adj` columns; corrected
counts everywhere they appear.

## Phase 2A.4 — Decontaminate threshold and conformal calibration

Two circularities:
- The operating threshold is fitted by F1-max **on val**, and val is then
  used as the drift **reference window**. Part of the reported F1 collapse
  is threshold optimism.
- Reported conformal "val coverage" (0.9041) is measured on the
  **calibration set** — guaranteed by construction, then compared against
  held-out test coverage to conclude "no drift".

Fix: select thresholds on a held-out slice of train or by nested selection;
split val into calibration and coverage-audit halves, or use cross-conformal.

**Acceptance:** re-reported degradation with the decontaminated protocol,
and an explicit statement of how much of the original gap was optimism.

## Phase 2A.5 — Clustering and selection integrity

- Patients contribute multiple encounters (46.2% multi-visit). Within-split
  correlation makes all standard errors anti-conservative. Add
  cluster-robust CIs or a patient-level bootstrap.
- Selection stages 5–6 operated on one feature. Repair Boruta (tune depth /
  iterations / `perc`) or re-describe the pipeline honestly.
- Add the missing ablations: selected-53 vs all-78 vs random-53 vs top-53
  by MI; and **raw-only vs raw+FE**, which is the only real test of the
  "+56% SHAP" feature-engineering claim.

**Acceptance:** cluster-robust intervals; a documented selection verdict;
ablation table with CIs.

---

# TIER 2B — METHODOLOGICAL MODERNISATION

## Phase 2B.1 — Adaptive conformal inference

Static split conformal assumes exchangeability — precisely what shift
violates. Implement **Adaptive Conformal Inference** (Gibbs & Candès,
NeurIPS 2021), which updates the quantile online to maintain coverage under
arbitrary shift. Optionally add covariate-shift-weighted conformal
(Tibshirani et al. 2019) and discuss Barber et al. (2023) on conformal
beyond exchangeability.

Compare ACI vs static split conformal across the Tier 0 regimes: coverage
over time, interval/set size, and recovery speed after an induced shift.

**Acceptance:** coverage-over-time figure per regime; a table showing where
static conformal fails and ACI holds; `docs/04_uncertainty.md` rewritten.

## Phase 2B.2 — Multivariate drift detection

Every current test is univariate and marginal, so no change in the
*dependence structure* between features is detectable. Add:

- **Classifier two-sample test** (Lopez-Paz & Oquab, ICLR 2017): train a
  discriminator to separate reference from production; report AUC with a
  permutation test
- **MMD two-sample test** (Gretton et al., JMLR 2012) with a suitable kernel
- **Black-box shift detection** (Lipton et al. 2018; benchmarked in
  Rabanser et al., NeurIPS 2019, "Failing Loudly"): univariate tests on
  model outputs with multiple-testing correction — the standard strong
  baseline this project currently approximates without the correction

Benchmark all of them inside the Tier 0 regime framework. That converts the
drift module from "a suite of tests" into "a characterised detector
comparison with known ground truth."

**Acceptance:** detection-power comparison across classical vs modern
detectors for each synthetic shift type; a recommendation with justification.

## Phase 2B.3 — Uncertainty must gate a decision

Conformal prediction is currently decorative — no component consumes it, and
mean set size 1.679 at 90% coverage means the system abstains on ~68% of
patients without that being interpreted.

Wire it into a **three-way clinical triage**: confident-low → routine,
confident-high → intervention, uncertain set → clinician review, under an
explicit review budget. Demonstrate that the abstention region carries
higher error than the confident regions — otherwise abstention is not
buying anything.

**Acceptance:** a triage policy module; error-rate-by-confidence-region
table; review-budget sensitivity; set-size efficiency interpreted, not just
reported.

## Phase 2B.4 — Threat-modelled robustness

Finite-difference FGSM/PGD on a piecewise-constant tree ensemble yields a
zero gradient almost everywhere; ASR ≈ 0.2–0.3% means the attack did not
execute. Two valid paths:

**(a) Keep the adversarial framing** — use tree-appropriate attacks:
Kantchelian et al. (ICML 2016) MILP evasion; Chen et al. (ICML 2019) robust
trees. State the threat model explicitly: who is the adversary and what do
they gain?

**(b) Reframe as data-quality robustness (recommended).** For a hospital
readmission model there is no credible adversary, but there are very real
EHR data-quality failures: missingness injection, coding drift, unit errors,
delayed labs, feature outages. This has a genuine threat model and direct
clinical relevance.

Whichever you choose, remove the invalid FGSM/PGD results or explicitly
document why they are near-zero.

**Acceptance:** stated threat model; valid attacks for the model class;
robustness score recomputed and re-interpreted.

---

# TIER 2C — ENGINEERING AND MLOPS

## Phase 2C.1 — Tests and CI

`tests/`: leakage guards (no future information, no cross-split cell
overlap), metric-schema assertions, split-disjointness assertions, a
pipeline smoke test. GitHub Actions running lint (ruff) + tests + smoke run
on push.

*A reliability toolkit with no automated verification is a contradiction —
this is the criticism a reviewer will most enjoy making.*

**Acceptance:** CI green on a clean clone; badge in README.

## Phase 2C.2 — Reproducibility

Pinned `requirements.txt` with lockfile; Dockerfile; deterministic seeding
verified by running twice and diffing artifacts; `make reproduce` target.

**Acceptance:** two clean runs produce byte-identical reports.

## Phase 2C.3 — Experiment tracking and registry

MLflow (local file backend is fine): log params, metrics, artifacts, model
versions. Migrate the hand-rolled JSON registry to MLflow Model Registry or
document why not.

**Acceptance:** every model in the tracking store with full lineage.

## Phase 2C.4 — Model card, datasheet, and fairness

- **Model card** (Mitchell et al. 2019) and **datasheet** (Gebru et al. 2021)
- **Subgroup analysis** across `race`, `gender`, `age`: performance,
  calibration, and alert rate per subgroup
- Explicit discussion of `payer_code` as a top predictor. Insurance status
  driving clinical risk prediction is exactly the pattern the algorithmic-
  bias literature warns about (cf. Obermeyer et al., *Science* 2019). A
  clinical ML reviewer will raise this; you need an answer.
- Note that prior-utilisation features make the task partly tautological —
  the model substantially identifies frequent utilisers

**Acceptance:** model card, datasheet, subgroup tables and an equity
discussion in `docs/`.

## Phase 2C.5 — Literature positioning

- Baseline comparison table: your results vs published results on this
  dataset and target (start from Strack et al. 2014)
- Rewrite related work as a *framing*: what prior work established, what it
  left open, where this project sits. The current six-item bibliography is
  never engaged in the text.

**Acceptance:** baseline table; a related-work section a specialist would
recognise as informed.

---

# TIER 3 — DEPLOYMENT

## Phase 3.1 — Gradio monitoring console

Match the FinRiskGuard quality bar. Tabs:

1. **Patient risk** — conformal prediction set, triage decision, SHAP
   waterfall, PPR-constrained threshold in use
2. **Drift monitor** — live PSI/KS/classifier-2ST/MMD panel with FDR-
   corrected significance, and the regime matrix as reference
3. **Model registry** — version history, promotion decisions, the valid
   test-only comparison with CIs
4. **The scientific story** — the falsification arc, interactive: choose a
   split regime and watch which signals fire
5. **Method and limitations** — including the censoring finding, stated
   plainly, and the model card

CPU-only, cold start < 30 s, demo subset only.

**Acceptance:** runs locally; deployment instructions verified end to end;
Space live and linked from the README.

---

# FINAL — README RESTRUCTURE

The README must tell the honest scientific story. Required section order —
the falsification arc is **section 3**, before results, not a footnote:

1. **Problem** — production models degrade silently; monitoring is the gap
2. **What we built** — summary table
3. **The scientific story** ← *the spine*
   - Initial framing: patient-level "temporal" split, 8/8 signals, CRITICAL
   - Falsification testing: random-split control over 20 seeds
   - The discovery: `encounter_id` is not a verified timestamp; the label is
     right-censored; the split is an entry-cohort split
   - Rigorous re-diagnosis: the regime × signal matrix
   - What the drift actually was, stated plainly
   - What survived and what did not
4. **Corrected results** — every number with mean ± std and CI
5. **Detector characterisation** — power curves; which signals diagnose
   which shift mechanism; classical vs modern detectors
6. **Uncertainty gating a decision** — triage policy, abstention budget
7. **Robustness** — stated threat model, valid attacks
8. **Engineering** — CI, reproducibility, tracking, model card
9. **Live demo**
10. **Limitations** — single dataset; no verified timestamps; no external
    validation; task-intrinsic ceiling on AUC
11. **Related work and baselines**
12. **References**

Every number traces to a file in `outputs/reports/`. Every superseded claim
is preserved in `outputs/reports/superseded/` and referenced from section 3.

**Acceptance:** all links resolve; a traceability script verifies every
numeric claim in the README appears in a generated artifact; section 3 reads
as a scientific contribution, not an apology.

---

# GLOBAL ACCEPTANCE CRITERIA

- [ ] `python src/pipelines/pipeline.py` reproduces everything from raw data
- [ ] CI green; tests genuinely test what they claim
- [ ] No headline number without variance or CI
- [ ] No metrics dict where train and val are identical
- [ ] Every detection rate accompanied by a confusion matrix
- [ ] Every threshold accompanied by predicted-positive-rate
- [ ] Every coverage number labelled in-sample or held-out
- [ ] Multiple testing corrected everywhere
- [ ] No hardcoded quality gate
- [ ] No unearned mechanism claim anywhere in the repository
- [ ] Regime × signal matrix generated and interpreted
- [ ] Superseded results preserved, not deleted
- [ ] Space live and linked

---

# HONEST CAVEATS — state, do not fix

Some things are cheaper to state plainly than to solve. Put these in
Limitations without apology:

- **Single dataset.** External validation on MIMIC-IV or eICU is future work.
- **No verified timestamps.** State that calendar-time ordering is
  unverifiable in this dataset and that the project therefore studies
  cohort shift under observation-window truncation.
- **AUC ≈ 0.65.** Do not apologise. Cite the published range for this task
  and note the contribution is reliability infrastructure, not discrimination.
- **No real adversary.** If the adversarial framing is retained, say plainly
  that the value is as a perturbation-sensitivity study.
- **Retrospective data.** No prospective validation; no deployment outcomes.

---

# REFERENCES TO IMPLEMENT AGAINST

- Gibbs & Candès (2021), *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS
- Tibshirani, Barber, Candès & Ramdas (2019), *Conformal Prediction Under Covariate Shift*, NeurIPS
- Barber, Candès, Ramdas & Tibshirani (2023), *Conformal Prediction Beyond Exchangeability*, Ann. Statist.
- Rabanser, Günnemann & Lipton (2019), *Failing Loudly*, NeurIPS
- Lipton, Wang & Smola (2018), *Detecting and Correcting for Label Shift*, ICML
- Lopez-Paz & Oquab (2017), *Revisiting Classifier Two-Sample Tests*, ICLR
- Gretton et al. (2012), *A Kernel Two-Sample Test*, JMLR
- Kantchelian, Tygar & Joseph (2016), *Evasion and Hardening of Tree Ensemble Classifiers*, ICML
- Chen et al. (2019), *Robust Decision Trees Against Adversarial Examples*, ICML
- Vickers & Elkin (2006), *Decision Curve Analysis*, Med. Decis. Making
- Obermeyer et al. (2019), *Dissecting racial bias in an algorithm...*, Science
- Mitchell et al. (2019), *Model Cards for Model Reporting*, FAT*
- Gebru et al. (2021), *Datasheets for Datasets*, CACM
- Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*, JRSS-B
- DeLong, DeLong & Clarke-Pearson (1988), *Comparing areas under ROC curves*, Biometrics
- Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates*, BioMed Res. Int.

---

# START HERE

Begin with **Phase 0.1**. Print your plan for the temporal validity
investigation, including which external anchors you will use and why, then
stop for review before writing code.
