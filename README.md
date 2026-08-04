<!-- Restructured Tier 3 / FINAL — written against final numbers -->

<div align="center">

# DriftSentinel

### Post-deployment reliability for a clinical prediction model — and a worked example of falsifying your own results

[![CI](https://github.com/Sarvarbek13/DriftSentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/Sarvarbek13/DriftSentinel/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Determinism](https://img.shields.io/badge/determinism-24_artifacts_byte--identical-3FB950)](outputs/reports/determinism.json)
[![Tests](https://img.shields.io/badge/tests-93_passing-3FB950)](tests/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Diabetes_130--US-yellow)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

</div>

> **The short version.** This project set out to detect distribution drift in a
> hospital readmission model, reported `8/8 signals CRITICAL`, and then tested
> whether that was true. It was not. Two of the eight detectors were structurally
> broken, the "drift" was largely an observation-window artifact, and the split
> the whole framing rested on had never been checked. **The corrected repository
> is the more useful artifact, and this README leads with how it was corrected
> rather than with what survived.**

---

## 1. The problem

A deployed model's inputs and operating conditions move, and the model is not
told. Precision degrades with no alert. By the time a clinician stops trusting
it, the damage is done.

That is the standard motivation, and it is why monitoring stacks get built. The
less standard problem — the one this repository is actually about — is that
**monitoring stacks are almost never falsified.** A drift detector that has never
been shown to stay silent when there is no drift cannot distinguish "drift" from
"my detector fires on anything." A null result from an instrument that has never
produced a positive one is not evidence of anything.

So the question here is not *did we detect drift*. It is **would we have known if
there were none?**

## 2. What we built

| | |
|---|---|
| **Dataset** | UCI Diabetes 130-US Hospitals (1999–2008) · 101,766 encounters · 71,518 patients |
| **Target** | 30-day readmission (`readmitted == "<30"`), prevalence **11.16%** |
| **Split** | Patient-level **entry-cohort** split — leakage-free, and *not* temporal ([§3](#3-the-scientific-story)) |
| **Model** | LightGBM, 53 features, 126 trees, isotonic-calibrated · test AUC **0.6410** [0.6255, 0.6556] |
| **Drift detection** | PSI/KS/χ² per feature with **BH–FDR** over a counted 318-test family · CUSUM · Page-Hinkley · **classifier two-sample test**, **MMD**, **BBSD** |
| **Negative control** | Every detector re-run on a random patient split over 20 seeds, where drift is impossible by construction |
| **Positive controls** | Synthetic covariate / label / concept shifts at swept magnitudes, to characterise *what each signal responds to* |
| **Uncertainty** | Split conformal + **Adaptive Conformal Inference** (Gibbs & Candès 2021), decontaminated calibration |
| **Robustness** | Threat-modelled **EHR data-quality** failures — the adversarial framing was withdrawn, with reasons |
| **Fairness** | Subgroup performance with patient-clustered intervals, and a falsification arm that injects a disparity the audit must catch |
| **Engineering** | CI on every commit · byte-level determinism over 24 artifacts · 93 tests · model card · live console |

**Every number in this README maps to a named file in
[`outputs/reports/`](outputs/reports/).** Nothing is hand-typed. Where a claim was
withdrawn, the superseded artifact is preserved in
[`outputs/reports/superseded/`](outputs/reports/superseded/) and referenced.

---

## 3. The scientific story

*This is the spine of the repository, and it is section 3 rather than a
limitations footnote because the correction is the contribution.*

### 3.1 The initial framing

Patients were sorted by their first `encounter_id` and split 60/20/20. The later
windows looked drifted: `payer_code` PSI 0.93, readmission rate falling across
splits, F1 collapsing. All eight concept-drift evidence signals fired.
**`SYSTEM STATUS: CRITICAL`.** The story wrote itself: a model degrading over
time, caught by monitoring.

### 3.2 Falsification

Two questions had never been asked.

**Is `encounter_id` ordering chronological?** The dataset has no date column, so
"drift over time" was an assumption, not a measurement. Tested against three
**external anchors** whose dates are known independently of this data:

| Anchor | Known date | Result |
|---|---|---|
| Troglitazone withdrawal | 2000-03-21 | all 3 uses fall in the lowest **0.12%** of ranks, p = 1.7e-09 |
| ICD-9 V85 (BMI) introduction | 2005-10-01 | **zero** occurrences before rank 0.535, p = 2.9e-85 |
| Rosiglitazone safety changepoint | 2007-05-21 | within-class share 0.511 → 0.243, p = 1e-04 |

**Verdict: SUPPORTED.** The three anchors also fall in the right *order*, and the
implied calendar map reproduces the dataset's own 30-day boundary — median implied
gap **17.9 days** for `<30` against **173.5** for `>30`.

**Do the detectors fire when there is no drift?** The complete pipeline was re-run
on a random patient split over 20 seeds. Drift is impossible there by
construction.

### 3.3 What that produced — three findings, none of them comfortable

**(a) The split is not temporal, even though the ordering is.** Sorting *patients*
by *first* encounter puts an early entrant's 2008 encounters in train. It is an
**entry-cohort split**. The word "temporal" is now used only where it refers to
`encounter_id` chronology, which was earned, and never for the split, which was
not.

**(b) The label shift is an observability artifact, not concept drift.**
`readmitted` is essentially an *in-extract successor indicator*: only **0.89%** of
`NO` rows have any later encounter. Late-entering patients have less follow-up
inside the extract in which to be observed returning. The final-observed-encounter
share rises **0.683 → 0.826** from val to test, and once final encounters are
excluded the `<30` gradient **reverses sign** (−0.0098 → +0.0311).

**(c) Two of the eight detectors were structurally broken.** `cusum_alarm` and
`ph_alarm` were not mis-tuned; they could not work. The original 8/8 counted
them.

**And the process finding, which matters more than any of the three:** the
original split assumption was **untested, not correct**. It happened to align
with a fact nobody had checked. Being accidentally right is not a defence.

### 3.4 The regime × signal matrix

The centrepiece. Same code, same features, same seed — only the split regime and
the injected shift move.
[`regime_matrix.json`](outputs/reports/regime_matrix.json)

| Family | Signal | Entry-cohort | **Random (no drift)** | Temporal | Covariate | Label | Concept |
|---|---|---|---|---|---|---|---|
| distribution | `label_drift` | 1.00 | **0.00** | 1.00 | 0.80 | 1.00 | 0.00 |
| distribution | `prediction_drift` | 1.00 | **0.05** | 1.00 | 1.00 | 1.00 | 0.00 |
| performance | `auc_drop` | 0.45 | **0.05** | 0.85 | 0.90 | 0.10 | 0.80 |
| performance | `auc_slope_negative` | 0.80 | **0.15** | 1.00 | 0.90 | 0.20 | 1.00 |
| performance | `brier_increase` | 0.00 | **0.00** | 0.00 | 0.60 | 1.00 | 0.00 |
| performance | `f1_drop` | 0.05 | **0.00** | 0.00 | 0.90 | 0.00 | 0.10 |

Mean signals fired: **3.30/6** entry-cohort · **0.25/6** random control ·
3.85/6 temporal.

The four performance signals are **not independent** — they are four views of one
degradation (R5), which is why they are grouped and why "8/8" was never eight
pieces of evidence.

### 3.5 What the drift actually was

> A **cohort-composition shift compounded by observation-window truncation**,
> with a modest genuine AUC degradation on top (**0.0199** entry-cohort against
> **0.0002** on the random control).
>
> It is **not** concept drift. `P(Y|X)` was never shown to change; what changed
> was who is in the window and how much of their future the extract can see.

No mechanism prose appears anywhere in this repository — no "billing codes
changed", no "demographics shifted". Tier 0 evidenced *chronology*, not *causes*,
and the causal claims it did test came back as observability artifacts.

### 3.6 What survived and what did not

| Original claim | Status |
|---|---|
| "8/8 drift signals, CRITICAL" | **WITHDRAWN.** Two detectors structurally broken; four of the rest are one signal in four views |
| "temporal split" | **WITHDRAWN.** Entry-cohort split. Ordering *is* chronological; the split is not |
| "concept drift detected" | **WITHDRAWN.** Observation-window truncation |
| "7-stage feature selection" | **WITHDRAWN.** Two real stages, five decorative ([§4.4](#44-the-selection-pipeline-costs-accuracy)) |
| "robustness score 0.8954, ROBUST" | **WITHDRAWN.** The attacks did not execute ([§7](#7-robustness)) |
| "detection rate 1.000, FPR 0.014" | **WITHDRAWN.** 941/1000 clean samples were flagged ([§7](#7-robustness)) |
| "cost-optimal threshold 0.128" | **WITHDRAWN.** 97% predicted-positive rate ([§6](#6-uncertainty-that-gates-a-decision)) |
| Val AUC 0.6865 | **WITHDRAWN.** Computed against the wrong target ([§4.1](#41-the-target-was-not-what-the-readme-said)) |
| `encounter_id` ordering is chronological | **SUPPORTED** — and it was never checked before |
| Patient-level split is leakage-free | **SUPPORTED** |
| A real, small AUC degradation exists | **SUPPORTED** — 0.0199 against a 0.0002 no-drift floor |

---

## 4. Corrected results

Every number carries an interval. Intervals are **patient-clustered** bootstraps
(2,000 draws, resampling *patients*), because 46.2% of patients contribute more
than one encounter and row-level intervals would be anti-conservative.
[`headline_metrics_ci.json`](outputs/reports/headline_metrics_ci.json)

### 4.1 The target was not what the README said

| Stage | Documented | Implemented | Prevalence |
|---|---|---|---|
| Original build | "within 30 days" | `<30` **or** `>30` — readmission *ever* | 46.1% |
| Tier 2A.1 | `<30` | `<30` | **11.16%** |

**The documented target and the implemented target disagreed for the entire
original build.** Every headline number was computed against a 4× more prevalent
target than advertised. AUC falls ~0.69 → ~0.64 on the switch, which is expected:
the merged target was easier.

### 4.2 Headline performance

Operating threshold **0.175**, selected by F1-max on a held-out, patient-level
slice of *train* — so the drift reference window carries no fitted quantity.

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

† **val ECE is in-sample and is not evidence** — the isotonic calibrator was fitted
on val. The held-out numbers are 0.0057 (val audit half) and 0.0080 (test).

Learner stochasticity over 20 seeds: test AUC **0.6345 ± 0.0026**.

**Retraining did not help.** `lgbm_v2` (trained on train+val) beats `lgbm_v1` by
**+0.0052 AUC** on the only comparable split — DeLong p = **0.171**, paired
patient-clustered bootstrap CI **[−0.0019, +0.0124]**. The interval contains zero.
Verdict: `NO_SIGNIFICANT_DIFFERENCE`, and v1 stays active.
[`model_comparison.json`](outputs/registry/model_comparison.json)

### 4.3 Multiple testing corrected — and it changed almost nothing

318 hypothesis tests **counted by re-running the shipped test functions**, not
read off the saved CSVs (`data_drift` overwrites its own output mid-run, so
reading the files would have undercounted in the flattering direction).

At matched α, Benjamini–Hochberg changes **one verdict in 318** (269 → 268 at
0.05; 257 → 256 at 0.01).

**Multiplicity was never the binding constraint. Effect size is.** Of 141
feature-windows surviving FDR, **129 have PSI < 0.10** — statistically certain and
practically negligible. The lesson that ships is *significance **and** a minimum
effect, never significance alone.*
[`fdr_correction.json`](outputs/reports/fdr_correction.json)

### 4.4 The selection pipeline costs accuracy

Under repeated patient-grouped CV with selection refitted **inside** every fold:

| Arm | AUC | vs shipped |
|---|---|---|
| **No selection at all (78 features)** | **0.6806** [0.6759, 0.6851] | **+0.0054** [+0.0041, +0.0068] |
| Simple in-fold target-aware selector | 0.6801 [0.6759, 0.6843] | **+0.0048** [+0.0040, +0.0057] |
| Stages 1–2 only | 0.6753 [0.6713, 0.6791] | 0.0000 [0, 0] |
| **Shipped 7-stage pipeline** | 0.6753 [0.6713, 0.6791] | — |

> The "7-stage feature selection pipeline" is **a two-stage variance/correlation
> filter with five decorative stages.** Stages 1–2 alone produced byte-identical
> output to the full pipeline in **all 10 folds** (Jaccard 1.0, AUC difference
> exactly 0.00000). **Doing no feature selection whatsoever beats it.**

Corroborating: a 4× change in target prevalence left the 53 selected features
**identical**, because the consensus vote is carried by target-independent stages.
It also corrected a leakage path — `pipeline.py` fitted the selector once on the
full training set and `trainer.py` then cross-validated over those features, so
every fold's held-out data helped choose the features it was scored on.
[`selection_ablation.json`](outputs/reports/selection_ablation.json)

### 4.5 Decontamination, and a correction to the correction

The operating threshold was fitted by F1-max **on val**, and val is also the drift
reference window — so the reference window carried a threshold tuned to itself
while the production window did not.

Tier 2A.4 measured that optimism at **0.0641** and reported it as a headline
finding: most of the observed F1 collapse was supposedly threshold optimism
rather than real degradation.

**That number was wrong.** The block had selected the threshold under one
calibrator, scored **val** under a second and **test** under a third — three
probability scales in one comparison. A threshold is a cut point on a probability
scale, so scoring two windows on different calibrators is not a like-for-like
comparison and the difference between them is partly an artifact of which
calibrator each side got.

Recomputed on a single scale, the threshold optimism is **0.0024**.
**The val→test F1 degradation was largely real, not an artifact of a self-fitted
threshold.** The conclusion Tier 2A.4 published was substantially wrong in the
direction that flattered the analysis.

> **A phase that existed to correct a contaminated number produced a
> contaminated correction.**
>
> And the error was visible only because reconciling the canonical metrics file
> to that threshold **forced a consistency check nobody had run**: two artifacts
> reporting a test F1 for the same threshold, and disagreeing. Neither was
> obviously wrong on its own. The disagreement was the entire signal.
>
> This is the most important finding in the remediation, and not because of the
> threshold. It is the clearest evidence that **a correction is an artifact like
> any other, and inherits no immunity from having been produced by an audit.**
> Nothing was checking Tier 2A.4 against anything else, because it *was* the
> check.
[`decontamination.json`](outputs/reports/decontamination.json) ·
[`10_threshold_reconciliation.md`](docs/10_threshold_reconciliation.md)

### 4.6 Fairness — the headline is age, and one claim was withdrawn

> **The model is best at the group that needs it least.** Highest-risk patients
> (70+, 9.6% prevalence) get the worst discrimination — AUC **0.6257**
> [0.6064, 0.6462] against **0.7461** [0.6810, 0.8074] for the under-40s, an
> interval-separated gap of **+0.1204**.

| Age band | n | prevalence | AUC [95% CI] | recall [95% CI] | PPR [95% CI] |
|---|---|---|---|---|---|
| <40 | 931 | 0.0763 | **0.7461** [0.6810, 0.8074] | 0.3521 [0.2037, 0.4935] | 0.1214 [0.0935, 0.1513] |
| 40–69 | 8,557 | 0.0820 | 0.6369 [0.6144, 0.6598] | 0.1966 [0.1664, 0.2264] | 0.0802 [0.0740, 0.0867] |
| **70+** | 7,837 | **0.0962** | **0.6257** [0.6064, 0.6462] | 0.1804 [0.1530, 0.2078] | 0.0916 [0.0849, 0.0987] |

**Supported** (non-overlapping intervals): AUC **+0.1204**; predicted-positive
rate **+0.0412** (<40 vs 40–69).

**Withdrawn, and stated rather than dropped.** The recall disparity previously
shipped as supported at **+0.1796**. At the corrected threshold the gap is
**+0.1717** — barely smaller — but the under-40 interval widens to
[0.2037, 0.4935] and now overlaps. **The gap did not vanish; the evidence for it
did.** With 931 patients and 71 positives that cohort cannot support a recall
claim of this size. The phrase "flagged at less than half the rate of the
under-40s" is retired with it — at 0.175 it is about three quarters.

**Race — an underpowered null, with its caveat attached:**

> **"Not supported" is not "no disparity exists."** AfricanAmerican AUC is 0.035
> lower than Caucasian (0.6133 vs 0.6478), but the intervals overlap, so no
> racial disparity in discrimination is claimed. This cohort is **underpowered to
> rule out** a gap of that size.

Three levels are marked `INSUFFICIENT_EVIDENCE` and are **named, never averaged
away**: Asian (n=209, 19 positives), Hispanic (n=496, 32), Other (n=445, 38).

**The one supported race disparity is calibration:**

> The model is **worst-calibrated for patients whose race was never recorded**
> (ECE **0.0343** vs **0.0071** for Caucasian; gap 0.0272, intervals
> non-overlapping). A data-quality problem and an equity problem simultaneously.
> That group's AUC interval [0.498, 0.670] also **contains 0.5**.

**The audit can detect a disparity**: an injected one (gap 0.1313) was flagged;
the real gender gap (0.0091) was not. The nulls above are measurements, not blind
spots. [`fairness_audit.json`](outputs/reports/fairness_audit.json)

**`payer_code`, and the Obermeyer frame.** Obermeyer et al. (*Science* 2019)
showed an algorithm using cost as a proxy for need systematically under-referring
Black patients — with race nowhere in the features. Insurance status is
structurally that kind of variable. **The measured answer: rank 17 of 53** by gain
importance, not top-five. Weaker than the audit assumed — but not a dismissal, because
the top features are **prior-utilisation** variables, and utilisation is itself
patterned by access. **The mechanism is relocated, not avoided.** The task is also
partly tautological: the model substantially identifies frequent utilisers.

---

## 5. Detector characterisation

Classical and modern detectors benchmarked **inside** the regime framework, so
each has a known-ground-truth answer rather than a verdict.
[`multivariate_drift.json`](outputs/reports/multivariate_drift.json)

| Regime | C2ST held-out AUC | C2ST perm. p | MMD² | MMD p | BBSD | Fires? |
|---|---|---|---|---|---|---|
| **Random (negative control)** | 0.5067 | 0.095 | 0.000183 | 0.184 | no | **silent** |
| Entry-cohort | **0.6932** | 0.048 | 0.002422 | 0.005 | yes | ✓ |
| Temporal | **0.6792** | 0.048 | 0.002112 | 0.005 | yes | ✓ |

All three are **silent on the control and fire on both real regimes** — the
calibration property the original 8-signal suite lacked. The C2ST's held-out AUC
doubles as a bounded **effect size**, which is exactly what §4.3 showed was
missing.

Honest qualification: on *this* data the multivariate tests revealed no drift the
marginal tests had missed — the marginals had already flagged 43–50 of 53
features. What they add here is calibration and one interpretable number; the
capability to catch a pure dependence-structure change is real but was not
exercised.

**Adopting a modern method and finding it unnecessary is a finding.** Of four
modern methods implemented, **two returned nulls and one a positive** —
[Literature Positioning §6](docs/07_literature_positioning.md).

---

## 6. Uncertainty that gates a decision

**Adaptive Conformal Inference** (Gibbs & Candès, NeurIPS 2021) vs static split
conformal. [`adaptive_conformal.json`](outputs/reports/adaptive_conformal.json)

| Stream | Static coverage | ACI coverage |
|---|---|---|
| val audit half (held out) | 0.8943 | 0.8996 |
| test (entry-cohort) | 0.9134 | 0.9004 |
| **synthetic hard label shift** | **0.4120** | **0.8980** |
| changepoint at 50% | 0.6654 | 0.8987 |

On the real streams ACI buys **+0.013 coverage** — essentially nothing, because
static conformal already holds. Exchangeability here is *bent, not broken*. **The
prediction was registered before the run**, and the synthetic hard shift was
included as the falsification condition: static collapses to 0.412 while ACI
recovers to 0.898. **So the null is a property of the data, not of the method or
our implementation of it** — a distinction unavailable without the arm. ACI is
retained as the correct default for a deployment whose shift is not known in
advance to be this benign.

**Every coverage number states whether it was measured in-sample or held out.**
Calibration-set coverage is guaranteed by construction and is not evidence.

**Threshold policy — and the rail that stopped carrying load.**
[`threshold_policy_lgbm_v1.json`](outputs/reports/threshold_policy_lgbm_v1.json)

| | threshold | PPR | binds? |
|---|---|---|---|
| Shipped (rate-form cost) | 0.0100 | **99.4%** | degenerate — cap essential |
| Corrected (population-weighted) | 0.1676 | **13.5%** | **cap redundant at every budget** |

The audit prescribed an alert-budget constraint because the "cost-optimal"
threshold flagged 97% of patients. The cause turned out to be different: the cost
function normalised each error type by its own class size, inflating a requested
5:1 FN:FP ratio into an effective **42.3:1** at 11.2% prevalence. Correcting that
moved PPR from 99.4% to 13.5% — **far enough that the safety rail the audit asked
for is no longer load-bearing**. At every budget offered (≤10%, ≤20%, ≤30%) the
optimum already sits below the cap. It is retained as a guard-rail and a
reporting discipline, not because it does work. *A constraint added to contain a
symptom can become unnecessary once the cause is fixed, and noticing that is part
of the fix.*

**Every threshold reports predicted-positive-rate**, and anything above PPR 0.60
is auto-flagged `DEGENERATE`. Decision curve analysis (Vickers & Elkin 2006) is
reported alongside.

**Triage — a null, reported as one.** The conformal gate does **not** beat a
matched-budget trivial baseline at any review budget, and the falsification arm
explains why: with 10% of labels corrupted, the gate routes them at lift
**−0.004**. Label corruption is invisible to any gate that sees only *x* and
*p(x)*. **"Route the uncertain cases" does not mean "route the cases the model
gets wrong."** [`triage_policy.json`](outputs/reports/triage_policy.json)

---

## 7. Robustness

**Threat model, stated first.** There is **no credible adversary** against a
hospital readmission triage model: no payout, no authentication to bypass, no
filter to evade. There *are* very real EHR data-quality failures. The adversarial
framing is withdrawn and replaced.

**Why it was withdrawn, not merely supplemented:** **71.7%** of finite-difference
gradients on this tree ensemble are **exactly zero**. FGSM and PGD had no
direction to step in. The shipped ASR of 0.2–0.3% was the null behaviour of a
method that did not execute — not evidence of robustness. The score of 0.8954 is
gone.

| Realistic scenario | Worst ΔAUC |
|---|---|
| Feature outage (top features) | **−0.0370** |
| Prior-utilisation fields arrive empty (50%) | −0.0070 |
| **Destructive control** | **−0.0456** |

The destructive control is the falsification arm: **the harness can detect damage
when damage is real**, so the graceful degradation above is interpretable.
[`data_quality_robustness.json`](outputs/reports/data_quality_robustness.json)

### The defense system's detection power was a schema artifact

> Before repair, Layer 1's violation count separated attacked from clean input at
> **AUC 0.94** — but only because the attack adds continuous noise to **binary**
> columns. What was being "detected" was a binary feature holding a non-integer
> value: **a data-type violation, not adversarial detection.** It implies nothing
> about an adversary who respects the schema. After repairing the degenerate
> bounds the same layer separates at **AUC 0.500**.

Corrected numbers, each with the false-positive rate it was measured at:

| Score | AUC | Detection @ 5% FPR |
|---|---|---|
| Five layers combined | 0.651 | 0.129 |
| Kept layer (the verdict score) | 0.5654 | 0.064 |

**There is no operating point at which this system usefully detects this attack.**
The shipped "detection rate 1.000, false positive rate 0.014" was an artifact of
flagging **941 of 1,000 clean samples** SUSPICIOUS while counting only the
`ADVERSARIAL` cell. **No detection rate appears anywhere in this repository
without its FPR** — a detection rate alone is unfalsifiable, since 100% is
available to any system willing to flag everything.

Two related corrections:

- **L1's 93.6% clean-trigger rate was structural, not a tuning error.** 20 of 53
  features are binary or zero-inflated, so Q1 == Q3 and the IQR rule collapsed to
  a single point that flagged everything at *every* multiplier. The audit's
  prescribed fix could not have worked without repairing the bounds first.
- **L4 is not a dead layer, it is an unreachable threshold.** Its flag fires on
  0.001 of clean input, but its continuous statistic still separates at
  **AUC 0.582**. "Dead" implies delete; "unreachable threshold" implies repair.
  *(The Tier 1.2 write-up also called L4 the most informative of the five. That
  was backwards — the most informative single layer is **L5 Ensemble Agreement at
  AUC 0.638**.)*

One layer of five carries signal. The system is described by that number, not by
the number implemented. [`defense_report_lgbm_v1.json`](outputs/log/defense_report_lgbm_v1.json)

---

## 8. Engineering

### Reproducibility, and the defect that made the claim false

The plan's acceptance criterion read *"`pipeline.py` reproduces everything from
raw data."* **It was false on every machine but one.**

Seven modules — including the **raw-data directory** and the **artifact
directory** — held hardcoded absolute paths to a single developer's machine. On
Linux CI the same literal resolves to a *relative* folder whose name contains
backslashes, so a run appears to succeed while writing nowhere useful. It was
invisible precisely because the machine it was written on is the machine it ran
on. Caught by the determinism sandbox, which was leaking into the real repository
instead of isolating from it.

```bash
python src/pipelines/reproduce.py          # rebuild the core chain from raw data
python src/pipelines/reproduce.py --list   # what is in the chain, and what is not
```

The claim is now **scoped**: `reproduce.py` rebuilds data preparation, both
models, calibration, conformal prediction and the registry. The analysis and
investigation modules **consume** that chain and run separately — they are listed
by name in `ANALYSIS_MODULES`, not silently implied.
[`reproduce_run.json`](outputs/reports/reproduce_run.json)

### Determinism is a CI check, not a habit

Determinism had been verified by hand three times, and by hand it kept passing.
The defect it needed to catch would have passed too, because that defect was
**value-identical and byte-different**: `TARGET_COLS` is a `set`, Python
randomises string hashing per process, so the output **column order changed on
every run** while every value stayed the same. No seed could catch it — it was in
serialisation. Dangerous specifically because the adversarial modules index
positionally.

So the check compares **artifact hashes**, and runs the chain twice under **two
different explicit `PYTHONHASHSEED` values** — verified by probing the
interpreter's actual hash in each environment, because `PYTHONHASHSEED=0`
*disables* randomisation and reading the variable back would prove nothing.

It **failed on its first run** and named a live instance: `list(to_drop)` over a
`set` in the correlation filter. Fixed; now **PASS over 24 artifacts**, including
all three model binaries. The verdict is `PASS` / `FAIL` / **`UNKNOWN`**, and
`UNKNOWN` exits non-zero — an unfalsifiable green is worse than a red.
[`determinism.json`](outputs/reports/determinism.json)

### Eight defects that worked by luck, not by design

| # | Defect | Why it survived |
|---|---|---|
| 1 | `TARGET_COLS` hash randomisation | values identical; only bytes moved |
| 2 | Boruta empty-candidate crash | Boruta confirmed **one** feature; **zero** raises. One feature from a hard failure |
| 3 | `list(to_drop)` over a `set` | the sibling line happened to preserve order |
| 4 | `registry.py` hardcoded `n_estimators: 173` | the model actually stopped at 126; provenance was typed, not read |
| 5 | `defense.py` stale hand-typed prose | numbers embedded in a *generated* artifact |
| 6 | Seven hardcoded absolute paths | it ran on the machine it was written on |
| 7 | Tier 2A.4's mixed-scale threshold block | a correction that nothing was checking, because it *was* the check ([§4.5](#45-decontamination-and-a-correction-to-the-correction)) |
| 8 | `age_band.recall` shipped as a supported disparity | the gap barely moved; only the **evidence** collapsed |

**#8 is worth its own note, because of what caught it.** When the operating
threshold was corrected, that disparity's gap moved 0.1796 → 0.1717 — a
rounding-error change. **A numbers-only diff would have shown a barely-moved gap
and passed it.** What withdrew the claim was diffing the fairness audit at
**claim level**: the audit does not report numbers so much as statements of the
form *this disparity is supported by the data*, and at 0.175 the under-40
interval widens enough to overlap. 931 patients and 71 positives cannot support
a recall claim of that size.

> **The gap did not vanish. The evidence for it did.**

So the reconciliation compares two things, deliberately: point estimates *and*
which claims survive. Diffing only the first is how a withdrawn result stays in
a README.

**#5 is the sharpest.** `README_MUST_INCLUDE.md` then quoted those stale numbers
*verbatim as must-ship requirements*. **The defect propagated into the document
written to prevent defects of that class.** A requirements file is an artifact
too, and nothing was checking it against the evidence it cited.

None was a modelling error. In four of the eight, a value that was **asserted**
sat next to a value that was **measured**, with nothing forcing them to agree. In
two more, a **check** was the thing nobody was checking.

### CI, tests, tracking

**CI runs on every commit** — `lint` (blocking correctness rules; the full rule
set is reported non-blocking with its count, because this repository is *not*
lint-clean and does not claim to be), `test` (93 passing), `determinism`, and
`console` (installs only the deploy runtime).

**MLflow was declined**, with a measured audit rather than an opinion: **4 of 5**
machine-checked traceability requirements pass, and **the one remaining failure is
named and left open** — closing everything would make the audit look curated.

The decisive argument is evidential — **MLflow would not have caught any of the
eight defects above.** `mlflow.log_param("n_estimators", 173)` logs the wrong
number just as faithfully.

The sharpest line in that audit was about our own tooling: `model_io.py` writes
provenance sidecars, is unit-tested, and had **zero callers** — so the audit
checked for the sidecar **file**, not the function, because a check on the
function would have passed while the property was false. **That gap is now
closed**: two call sites, no new dependency, and the regeneration moved nothing
but provenance — all four model binaries came back byte-identical. It is the
argument for declining MLflow, demonstrated rather than asserted.
[Experiment Tracking Decision](docs/09_experiment_tracking_decision.md)

**[Model card](docs/08_model_card.md)** (Mitchell et al. 2019) with intended use,
subgroup performance, and out-of-scope use — including, derived from §4.6 rather
than asserted as boilerplate: **not for age-stratified triage without
recalibration.**

---

## 9. Live demo

A Streamlit console whose central design rule is that **no drift verdict can be
rendered without the no-drift baseline beside it** — the contrast is the
component, not a caption.

```bash
pip install -r requirements.txt      # console runtime only
streamlit run app.py
```

The console loads **no model**: a 30 KB precomputed evidence bundle, verified
against its source artifacts by `src/pipelines/demo_bundle.py` (currently
**IN_SYNC**). Deployment prep found three defects that would have failed on the deploy target
while working perfectly locally — four invalid alert icons that raised at script
execution, an incompatible CORS/XSRF config pair, and a non-ASCII `print()` that
crashes only when a Windows parent captures the pipe. All are fixed and guarded
by tests.

*The icon defect is the same failure class as the script-mode crash on a sibling
project's Spaces deployment: two platforms, two deployments, one root cause —
**the app was broken and looked fine, because nothing had ever executed the
script the way the platform executes it.*** The fix is not a better icon; it is a
test that runs the script the way the platform will.
[Deployment](docs/11_deployment.md)

---

## 10. Limitations

- **Single dataset.** No external validation (MIMIC-IV, eICU). Future work.
- **No verified timestamps.** `encounter_id` ordering is *evidenced* as
  chronological but there is no date column; the calendar map is piecewise-linear
  through three anchors. This studies **cohort shift under observation-window
  truncation**, and says so.
- **AUC ≈ 0.64, without apology.** Inside the published band for this task and
  target (0.61–0.66). The contribution is reliability infrastructure, not
  discrimination.
- **The task is partly tautological** — prior-utilisation features dominate, so
  the model substantially identifies frequent utilisers.
- **Health Facts confounds cohort with calendar time.** The UCI release drops the
  hospital identifier, so panel composition cannot be stratified apart from time.
- **Retrospective.** No prospective validation, no deployment outcomes.
- **One traceability gap remains open and named**: `threshold_policy_lgbm_v1.json`
  lacks a `reproducibility` block. It is left open deliberately — one documented
  gap is honest; closing everything makes an audit look curated. It is in
  [the audit](outputs/reports/tracking_traceability_audit.json), which reports
  **4 of 5** passing.

---

## 11. Related work and baselines

**Nothing here is methodologically novel** — not the dataset, not the task, not
the detectors. Stated first because a specialist knows it within a paragraph.

The strongest quantitative claim is the **split-regime result**: hold the model,
features, code and seed fixed, move only the split regime.

| Regime | AUC | Δ vs deployed |
|---|---|---|
| Patient-grouped CV | **0.6806** [0.6759, 0.6851] | **+0.0396** |
| Random patient split (20 seeds) | **0.6785** ± 0.0067 | **+0.0376** |
| Entry-cohort held-out (deployed) | **0.6410** [0.6255, 0.6556] | — |

**Nothing about the model changed.** A ~0.04 swing from the split alone — larger
than the ~0.024 spread between the best and worst tree-based model across the
published comparison studies.

| Source | AUC | Protocol | Patient-grouped? |
|---|---|---|---|
| Strack et al. (2014) | *not reported* | logistic regression, ~70k encounters | — |
| Liu, Sue & Wu (2024) | 0.64 [0.64, 0.65] | group 5-fold CV | ✅ |
| Salim & Ibrahim (2026) | 0.664 | nested stratified 5-fold CV | ❌ *(stated limitation)* |
| **DriftSentinel** (deployed) | **0.6410** | entry-cohort held-out | ✅ |

**Strack et al. (2014) reports no discrimination metric at all.** It is an
*etiologic* study — logistic regression estimating the association between HbA1c
measurement and early readmission, reporting odds ratios. It is routinely cited
as a predictive baseline on this dataset. **It is not one, and this project does
not cite it as one.**

**Claimed:** a single AUC without its split regime is not comparable across papers
on this dataset. **Not claimed:** that the published difference *is* leakage — two
studies is not a sample, and they differ in more than the grouping dimension.

**The contribution, precisely:** not the model, not the detectors, but **the
negative-control methodology and the systematic falsification design** — every
phase carrying a stated condition under which the method *must* fire, run before
the real result is interpreted. That is what makes this project's nulls
interpretable rather than ambiguous.
[Literature Positioning](docs/07_literature_positioning.md)

---

## Documentation

| | |
|---|---|
| [EDA & Data Pipeline](docs/01_eda_and_data_pipeline.md) · [Feature Engineering](docs/02_feature_engineering.md) · [Model Training](docs/03_model_training.md) | build |
| [Drift Detection](docs/04_drift_detection.md) · [Uncertainty](docs/05_uncertainty.md) · [Adversarial](docs/06_adversarial.md) ⚠ *superseded, banner-marked* | original framing |
| [Literature Positioning](docs/07_literature_positioning.md) · [Model Card](docs/08_model_card.md) · [Experiment Tracking Decision](docs/09_experiment_tracking_decision.md) | corrected |
| [Threshold Reconciliation](docs/10_threshold_reconciliation.md) · [Deployment](docs/11_deployment.md) | generated / ops |
| [AUDIT](docs/AUDIT.md) · [REMEDIATION_PLAN](docs/REMEDIATION_PLAN.md) · [README_MUST_INCLUDE](docs/README_MUST_INCLUDE.md) | the correction trail |

## Quick start

```bash
git clone https://github.com/Sarvarbek13/DriftSentinel.git && cd DriftSentinel

pip install -r requirements.txt          # console only
streamlit run app.py

pip install -r requirements-dev.txt      # full research environment
python src/pipelines/reproduce.py        # rebuild the core chain from raw data
pytest tests -q                          # 93 tests
python src/monitoring/determinism.py --stage trainer   # byte-level determinism
```

## 12. References

- Barber, Candès, Ramdas & Tibshirani (2023). Conformal prediction beyond exchangeability. *Ann. Statist.* 51(2), 816–845.
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *JRSS-B* 57(1), 289–300.
- DeLong, DeLong & Clarke-Pearson (1988). Comparing areas under correlated ROC curves. *Biometrics* 44(3), 837–845.
- Gebru et al. (2021). Datasheets for datasets. *CACM* 64(12), 86–92.
- Gibbs & Candès (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
- Gretton et al. (2012). A kernel two-sample test. *JMLR* 13, 723–773.
- Kantchelian, Tygar & Joseph (2016). Evasion and hardening of tree ensemble classifiers. *ICML*.
- Lipton, Wang & Smola (2018). Detecting and correcting for label shift with black box predictors. *ICML*.
- Liu, Sue & Wu (2024). Comparison of ML models for predicting 30-day readmission in diabetes. *J. Med. AI* 7:23. doi:10.21037/jmai-24-70
- Lopez-Paz & Oquab (2017). Revisiting classifier two-sample tests. *ICLR*.
- Mitchell et al. (2019). Model cards for model reporting. *FAT\** 220–229.
- Obermeyer, Powers, Vogeli & Mullainathan (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science* 366(6464), 447–453.
- Rabanser, Günnemann & Lipton (2019). Failing loudly. *NeurIPS*.
- Salim & Ibrahim (2026). A machine learning approach for predicting 30-day hospital readmission in diabetes. *Healthcare* 14(9):1185. doi:10.3390/healthcare14091185
- Strack et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Res. Int.* 2014:781670.
- Tibshirani, Barber, Candès & Ramdas (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Vickers & Elkin (2006). Decision curve analysis. *Med. Decis. Making* 26(6), 565–574.
