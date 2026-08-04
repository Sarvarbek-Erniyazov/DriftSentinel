> ## ⚠️ RESOLUTION BANNER — read before this document
>
> **This file is a historical record and its body is deliberately NOT edited.**
> It is the adversarial audit as written, before any of it was tested. Editing it
> would destroy the evidence of the original framing (the preservation rule: "Narrative
> honesty"). Tier 0 tested its central finding and returned a mixed verdict:
>
> | Audit claim | Tier 0 outcome |
> |---|---|
> | F1(a) `encounter_id` is not a verified timestamp; "there is currently no answer" | **REFUTED as framed.** It was never unverifiable — it was unverified. Three external anchors verify chronological ordering (`outputs/reports/temporal_validity.json`, verdict SUPPORTED) |
> | F1(b) the split windows overlap; this is an entry-cohort split | **CONFIRMED**, and independent of F1(a) |
> | F1(c) the label is right-censored, correlated with entry order | **CONFIRMED**, with a sharper mechanism than proposed: `readmitted` is essentially an in-extract successor indicator |
> | F6 no negative control | **CONFIRMED and quantified.** 2.15/8 signals fire on a random split where drift is impossible |
> | F25 CUSUM/PH parameters arbitrary | **UNDERSTATED.** Both are structurally broken, not merely mis-calibrated: CUSUM is saturated (101–207 alarms/run incl. no-drift); Page-Hinkley's verdict is decided by whether the first `MIN_WINDOW_SIZE=200` rows happen to have above- or below-average error |
>
> Everything below is preserved verbatim as the input to that work.

# DriftSentinel — Adversarial Audit

**Reviewer stance:** skeptical KAIST/GIST/DGIST/UNIST professor with an ML
background, reading the repository line by line, looking for reasons to
doubt the result.

**Disclosure:** I co-wrote this codebase with you. Several of the findings
below are defects I introduced. That does not make them less serious — it
means they were never independently checked, which is exactly the condition
this audit is meant to correct.

**How to read severity:**
- **BLOCKER** — if a reviewer finds this, the conversation is over
- **SERIOUS** — materially damages credibility, recoverable if pre-empted
- **MODERATE** — a competent reviewer notices and marks you down
- **POLISH** — visible sloppiness, cheap to fix

---

# VERDICT SUMMARY

| # | Finding | Severity | Effort |
|---|---|---|---|
| F1 | Label shift is likely a right-censoring artifact, not concept drift | **BLOCKER** | High |
| F2 | `pipeline_ready = True` is hardcoded over a failing check | **BLOCKER** | Trivial |
| F3 | lgbm_v2 "val AUC" is in-sample; the model comparison is invalid | **BLOCKER** | Low |
| F4 | Defense system triggers on 93.6% of clean data; reported FP rate hides it | **BLOCKER** | Medium |
| F5 | Cost-sensitive threshold flags 97% of all patients | **BLOCKER** | Low |
| F6 | No negative control: drift detector never tested on a no-drift split | **BLOCKER** | Medium |
| F7 | Target merges `<30` and `>30`; not the clinical readmission task | SERIOUS | Medium |
| F8 | Every number is from one split, seed 42. No CIs, no repeats | SERIOUS | Medium |
| F9 | "7-stage" selection contains three no-op stages | SERIOUS | Low |
| F10 | 265 hypothesis tests, no multiple-testing correction | SERIOUS | Low |
| F11 | FGSM/PGD are ill-defined on tree ensembles; ASR≈0 means "attack did nothing" | SERIOUS | Medium |
| F12 | Conformal prediction is decorative — no decision consumes it | SERIOUS | Medium |
| F13 | Threshold fitted on val, then val used as the drift reference window | SERIOUS | Low |
| F14 | Reported conformal "val coverage" is in-sample and trivially satisfied | SERIOUS | Low |
| F15 | "8 independent evidence signals" — they are not independent | SERIOUS | Trivial |
| F16 | Defense layers L3 anti-informative, L4 dead, L5 near-dead | SERIOUS | Low |
| F17 | Drift methods are 1954–2014 vintage; no modern baseline | MODERATE | Medium |
| F18 | Static split conformal where adaptive conformal is the correct tool | MODERATE | Medium |
| F19 | No comparison to published results on this dataset | MODERATE | Low |
| F20 | Related work is a citation list, not a framing | MODERATE | Low |
| F21 | No CI, no experiment tracking, no container, no tests | MODERATE | Medium |
| F22 | Not deployed — for a "production" toolkit | MODERATE | Medium |
| F23 | No FE ablation; "+56% SHAP" is not evidence of value | MODERATE | Low |
| F24 | Health-check severity mapping was tuned to produce a desired output | MODERATE | Trivial |
| F25 | CUSUM/PH parameters arbitrary; no ARL calibration; 109 alarms = saturated | MODERATE | Medium |
| F26 | AUC slope reported with no significance test | MODERATE | Trivial |
| F27 | README/artifact inconsistency: PSI 0.93 vs 0.84 vs 0 depending on window | MODERATE | Low |
| F28 | Repeated-measures structure ignored (patients contribute many rows) | MODERATE | Medium |
| F29 | Prior-utilization features make the task near-tautological | MODERATE | Low |
| F30 | Matplotlib defaults, unresolved warnings in shipped logs | POLISH | Low |

---

# PART 1 — METHODOLOGICAL WEAKNESSES

## F1 — BLOCKER: your drift is probably a censoring artifact

**The claim.** The README asserts a patient-level *temporal* split, and
attributes a 14pp readmission-rate drop (47.6% → 33.6%) to concept drift:
"the world keeps changing," "newer patients have different insurance
patterns, fewer diagnoses, and a lower readmission rate."

**The problem.** Three separate issues compound:

**(a) `encounter_id` is not a timestamp.** The UCI Diabetes 130-US dataset
contains no date column. `encounter_id` is an identifier from the Health
Facts database. Your own validator flagged
`encounter_id_monotonic = False`. You are *assuming* it is chronologically
ordered. The README states it as fact. A reviewer who knows this dataset —
and clinical ML reviewers do, it is one of the most-used clinical
benchmarks — will ask "how do you know?" and there is currently no answer.

**(b) The split windows overlap almost completely.**
```
Train enc_id : 12,522      — 443,847,176
Val   enc_id : 162,291,804 — 443,857,166
Test  enc_id : 241,367,706 — 443,867,222
```
Train's range subsumes nearly all of val's and test's. This is not a
temporal split. You split on *first* encounter_id per patient, which sorts
patients by **entry cohort**, not encounters by time. A patient who entered
early but kept visiting through 2008 is entirely in train, including their
late encounters.

**(c) The label is right-censored, and censoring correlates with entry
time.** `readmitted` is only observable if the patient returns *within the
data collection window*. A patient whose first encounter falls near the end
of the window has less observable future, and is therefore mechanically
more likely to be labelled `NO`.

Your test set is, by construction, the latest-entering patients. So it is
the most censored cohort. A 14pp increase in `NO` is exactly what
right-censoring predicts.

**Corroborating evidence already in your logs.** `number_inpatient` falls
monotonically across splits: train 0.723 → val 0.598 → test 0.359. Prior
utilisation is a *history* variable. Late entrants have less recorded
history for the same reason they have less recorded future — the
observation window truncates both ends. That is a data-collection artifact,
not an epidemiological trend.

**Why this is a blocker.** Every headline result in the project rests on
the interpretation "the world changed." If the shift is a censoring
artifact, then:
- the drift is real (distributions genuinely differ) but the *causal
  narrative* is unsupported
- "retraining recovers performance" becomes "training on more of the
  censored distribution partly compensates for censoring"
- the clinical framing ("hospital billing codes changed," "readmission
  rates dropped") is fabricated explanation

**The diagnostic experiment you must run.** For each patient compute
`first_encounter_id`, `n_encounters`, `observed_readmission_rate`, and
`number_inpatient`. Regress each on entry rank. If all three decline
monotonically with entry order, censoring is confirmed.

**What world-class looks like.** You do not have to abandon the project.
You have to *diagnose it and report it*. The strongest possible version of
this section reads:

> We initially framed the observed shift as temporal concept drift. On
> investigation we found `encounter_id` is not a verified timestamp and
> that the label is right-censored in a manner correlated with patient
> entry order. We therefore re-characterise the shift as **cohort shift
> under observation-window truncation**, verify it with [experiment], and
> re-run the full detection pipeline under three split regimes: random
> patient split (negative control), entry-cohort split, and a
> label-preserving synthetic covariate shift.

That is a *better* paper than the one you have. Reviewers reward
self-correction; they punish unexamined assumptions.

---

## F2 — BLOCKER: `pipeline_ready = True` is hardcoded

`pipelines/pipeline.py` contains:
```python
summary["pipeline_ready"] = True
```
This overrides a consistency check that reported **12 FAIL**. The intent
was "drift is expected, do not halt." The appearance to a reviewer reading
the file cold is: *a quality gate was disabled to make the pipeline
report success.*

Nothing else in the audit will matter if a reviewer sees this line before
seeing the justification.

**Fix:** replace with an explicit, auditable expression —
`ready = consistency.ready or (drift_expected and all_fails_are_drift_related)`
— log the reasoning, and surface `expected_drift_failures` and
`unexpected_failures` as separate fields.

---

## F3 — BLOCKER: the lgbm_v2 comparison is invalid

`registry.py` contains:
```python
val_metrics = train_v2_metrics
```
lgbm_v2 was trained on `train + val`. Its reported "val AUC = 0.7987" is
**training performance on data it was fitted on**. The comparison table then
prints:

```
auc  val  0.6865 → 0.7987  +0.1122↑  lgbm_v2 WINS
f1   val  0.6735 → 0.7345  +0.0610↑  lgbm_v2 WINS
```

This compares lgbm_v1's held-out performance against lgbm_v2's in-sample
performance and declares a winner. It is in `registry_history.csv`, in
`model_registry.json`, and it drives the `PROMOTE` verdict.

The only defensible number in that table is test AUC 0.6560 → 0.6648
(**+0.0088**), and even that has no confidence interval. On 17,325 samples
that difference is plausibly within noise — you need a DeLong test or a
paired bootstrap to claim it.

Additional confound: lgbm_v1 used early stopping (173 trees); lgbm_v2 used
a fixed 300 trees with no early stopping, because val had been consumed.
The two models differ in training data *and* capacity *and* stopping rule.
Not a controlled comparison.

**Fix:** either (a) hold out a fourth split for honest v2 evaluation, or
(b) report only test metrics with a paired bootstrap CI and state plainly
that val metrics for v2 are in-sample and excluded.

---

## F4 — BLOCKER: the defense system fires on 93.6% of clean data

From `defense_report_lgbm_v1.json`:

| Layer | Clean trigger | Attacked trigger | Lift |
|---|---|---|---|
| L1 InputValidation | **0.936** | 1.000 | +0.064 |
| L2 AnomalyDetection | 0.077 | 0.098 | +0.021 |
| L3 Consistency | 0.280 | 0.121 | **−0.159** |
| L4 Smoothing | 0.000 | 0.000 | 0.000 |
| L5 EnsembleAgreement | 0.002 | 0.018 | +0.016 |

Clean data: 45 CLEAN / 941 SUSPICIOUS / 14 ADVERSARIAL.

The README reports **"Detection rate 1.000, false positive rate 0.014."**

That FP rate counts only the ADVERSARIAL verdict. **94.1% of clean
production traffic is flagged SUSPICIOUS.** A detector that flags 94% of
normal traffic has no operating value, and the reported metric was chosen —
whether deliberately or not — in a way that conceals this.

This is the single most damaging finding after F1, because it is not a
methodological subtlety. It is a headline claim that inverts when you open
the JSON.

Compounding (F16): L3 fires *more* on clean than attacked data (negative
lift — anti-informative). L4 never fires at all. L5 fires on 1.8%. Of five
advertised layers, **one carries usable signal**.

**Fix:** recalibrate L1's IQR multiplier against a target clean-trigger
rate (e.g. 5%), delete or repair L3 and L4, and report a proper ROC over
the decision threshold — detection rate *at matched false-positive rate*,
which is the only comparison that means anything.

---

## F5 — BLOCKER: the cost-optimal threshold flags 97% of patients

Reported as a headline win:

> Cost-sensitive 0.128 → F1 0.5126, Recall 99.5%, missed readmissions 29

Compute the implied alert volume:
```
predicted_positive_rate = recall × prevalence / precision
                        = 0.995 × 0.3364 / 0.3452
                        = 0.9696
```
**The system flags 96.96% of all patients as high-risk.**

This is a degenerate solution. A clinical decision support tool that alerts
on 97% of admissions provides no triage information and would be switched
off within a week for alert fatigue — a well-documented failure mode of
clinical prediction deployment. Presenting it as "+54.2pp recall gain" is
technically true and practically meaningless.

Compounding: the FN×5 / FP×1 cost ratio is asserted with no citation and no
sensitivity analysis. The entire conclusion depends on a number you made up.

**Fix:** constrain the threshold search by a realistic alert budget (e.g.
"flag at most 20% of admissions"), sweep the cost ratio over two orders of
magnitude, and report the *feasible* operating region. Report net benefit /
decision curve analysis (Vickers & Elkin 2006), which is the standard
clinical framing and which reviewers in health informatics expect.

---

## F6 — BLOCKER: no negative control

The central claim is "DriftSentinel detects drift." There is no experiment
showing it **does not fire when there is no drift**.

Without that, the claim is unfalsifiable. A skeptical reviewer's first
question: *"Would your 8/8 evidence signals also fire on a random split?"*
Given uncorrected multiple testing (F10), saturated CUSUM parameters (F25),
and thresholds like `auc_drop > 0.02`, the honest answer may well be yes.

**The required experiment.** Re-run the entire detection pipeline under:
1. random patient-level split (expect: 0/8 or near-0 signals)
2. entry-cohort split (your current setup)
3. synthetic covariate shift with the label mechanism held fixed
4. synthetic label shift with covariates held fixed

Report a 4×8 matrix of which signals fire in which regime. That single
table converts the project from "we detected something" to "we
characterised detector behaviour under known ground truth" — which is what
a detection paper is actually supposed to contain.

This is the highest value-per-hour item in the entire audit.

---

## F7 — SERIOUS: the target is not the clinical task

`readmitted_binary` merges `<30` and `>30` into one positive class,
producing 49% prevalence.

The clinically meaningful and literature-standard target is **30-day
readmission** (11.16% prevalence) — the CMS-penalised outcome, the endpoint
in Strack et al. 2014, and the endpoint in essentially every published
model on this dataset. `>30` includes a patient readmitted two years later,
which is not a readmission-risk signal in any operational sense.

By merging, you made the task easier, less imbalanced, and less relevant.
A health-informatics reviewer will flag this in the first minute.

**Fix:** make `<30` the primary target and report the merged target as a
secondary analysis with the rationale stated. Expect AUC to drop; published
30-day models on this dataset land roughly in the 0.63–0.70 band, so you
will still be in range, and you will be *comparable*.

---

## F8 — SERIOUS: one split, one seed, no intervals

Every number in the README derives from a single split with `random_state=42`.
No repeated splits, no bootstrap CIs, no variance.

Consequences:
- Is −0.0305 AUC degradation larger than split-to-split noise? Unknown.
- Is +0.0088 (v1→v2) real? Unknown.
- Is ρ=0.6537 (MI vs SHAP) stable? Unknown.
- Is 0.8954 robustness score reproducible? Unknown.

You applied exactly the right standard to CellTriage (repeated grouped
nested CV, mean ± std, bootstrap CIs) and none of it here.

**Fix:** repeat the full pipeline over ≥20 seeds for the split-dependent
results, report mean ± std, and bootstrap the test-set metrics.

---

## F9 — SERIOUS: the "7-stage" selection has three no-op stages

From `selector.log`:
```
Stage 4 Boruta      : Input 42 → Selected 1  (41 rejected)
Stage 5 SHAP        : Input 1  → Selected 1  (0 removed)
Stage 6 Stability   : Input 1  → Selected 1  (0 removed)
Stage 7 Consensus   : 78 features → 53 selected
```
Stages 5 and 6 operated on a **single feature** and removed nothing. The
final 53 features come from consensus voting over stages 1–3 only. The
README presents this as a sophisticated seven-stage pipeline.

Also undiagnosed: why did Boruta reject 41/42? On 63k rows with weak signal
and default depth, shadow features are competitive — but that is a
diagnosis you should have made and documented, not a result you should have
shipped.

**Fix:** either repair Boruta (tune depth/iterations, use `perc` threshold)
or remove stages 5–6 and describe the pipeline honestly as four stages plus
consensus. Add an ablation showing the selected 53 beat (a) all 78, (b) a
random 53, (c) top-53 by MI alone.

---

## F10 — SERIOUS: 265 tests, no multiple-testing correction

`data_drift.py` runs up to five tests × 53 features per window, and
`concept_drift.py` adds more. At α=0.01 across ~265 tests you expect
several false positives by chance alone. The claim "31/53 features drifted"
has no family-wise or FDR control.

**Fix:** Benjamini–Hochberg FDR at q=0.05, report both raw and adjusted
p-values, and re-state the drifted-feature count post-correction. This is a
30-minute fix that removes an obvious line of attack.

---

## F11 — SERIOUS: gradient attacks on tree ensembles are ill-defined

`attacks.py` implements FGSM and PGD using finite differences:
```python
gradients[:, j] = (p_plus - p_orig) / h    # h = 1e-3
```
LightGBM is piecewise constant. For almost every (sample, feature) pair, a
perturbation of 1e-3 does not cross a split threshold, so the numerator is
**exactly zero**. The "gradient" is zero almost everywhere.

Result: FGSM ASR = 0.29%, PGD ASR = 0.20%. The report interprets this as
*the model is robust to gradient attacks*. The correct interpretation is
**the attack did not execute**. You are reporting the null behaviour of a
broken method as a robustness result.

MASK_k5 achieving the highest ASR (4.51%) is consistent with this: the only
attack that actually moved features across split boundaries was the one
that replaced them wholesale.

Separately: **there is no threat model.** Who is the adversary against a
hospital readmission model, and what do they gain? The README never says.
A reviewer will ask, and "adversarial robustness" without a threat model
reads as method-shopping.

**What world-class looks like.** For tree ensembles the relevant literature
is Kantchelian, Tygar & Joseph (ICML 2016) — exact MILP evasion for tree
ensembles — and Chen et al. (ICML 2019) on robust tree training. For this
domain, the defensible reframing is **not** adversarial ML but
**data-quality robustness**: missingness injection, coding-drift
perturbation, unit errors, and delayed-lab scenarios. Those have a real
threat model (EHR data quality), a real literature, and real clinical
relevance.

---

## F12 — SERIOUS: conformal prediction is decorative

You asked directly whether the UQ is used for anything. It is not.

- `quantifier.py` produces prediction sets and coverage tables
- `threshold.py` uses raw calibrated probabilities, **not** conformal sets
- `alerting.py` does not consume coverage
- No decision anywhere changes as a function of the conformal output

The conformal module is a parallel artifact that reports on itself.

Compounding: mean set size at 90% coverage is **1.679**, meaning ~68% of
patients receive *both* labels — the system abstains on two-thirds of
cases. That is reported neutrally in a table. It is a poor efficiency
result and should be interpreted, not just printed.

**What world-class looks like.** UQ must gate a decision. The natural
design here: three-way clinical triage — confident-low → routine discharge,
confident-high → intervention, uncertain set → clinician review — with an
explicit review budget, and a demonstration that the abstention region
carries higher error than the confident regions. That is the same
decision-theoretic structure you designed for CellTriage. Port it.

---

## F13 — SERIOUS: the drift reference window is contaminated

`evaluator.py` selects the operating threshold by F1-max **on val**
(0.3958). `concept_drift.py` then treats **val as the reference production
window** and measures test degradation against it.

The reference window therefore has a threshold fitted on itself; the
production window does not. Part of the reported F1 collapse (−0.1436) and
precision collapse (−0.1560) is threshold optimism, not drift.

**Fix:** select the threshold on a held-out slice of train, or use nested
selection, and re-report the degradation. Expect the F1 gap to shrink. That
is a *finding*, and reporting it strengthens you.

---

## F14 — SERIOUS: reported "val coverage" for conformal is in-sample

`conformal_report_lgbm_v1.json` reports val_coverage 0.9041 vs test_coverage
0.9139 and concludes `drift_signal: false` for all nine predictors.

The conformal predictor was **calibrated on val**. Val coverage ≈ target by
construction — it is what the algorithm solves for. Comparing a fitted
quantity against a held-out quantity and concluding "no drift" is circular.

**Fix:** split val into calibration and coverage-audit halves, or compute
val coverage by cross-conformal. Only then is the val→test comparison
meaningful.

---

## F15 — SERIOUS: the eight evidence signals are not independent

Alert CD-003 states: *"Concept drift confirmed by 8 independent signals."*

They are not:
- `auc_drop`, `f1_drop`, `brier_increase`, `auc_slope_negative` — four
  views of one performance degradation
- `cusum_alarm`, `ph_alarm` — two detectors on the *same* error stream
- `prediction_drift`, `label_drift` — mechanically coupled; if P(Y) shifts
  and the model is calibrated, P(Ŷ) must shift

Honest count: roughly **three independent evidence families** (performance,
sequential error accumulation, distribution shift). "8/8" is the most
quotable number in your README and it is an overclaim.

**Fix:** group the signals into families, report family-level agreement,
and drop the word "independent."

---

# PART 2 — ENGINEERING GAPS RELATIVE TO 2025–2026 PRACTICE

## F17 — Drift detection is classical-only

Your stack: PSI (industry heuristic, no canonical citation), KS, Chi²,
Jensen–Shannon, Mann–Whitney, CUSUM (Page 1954), Page–Hinkley (1971).

All are legitimate and all are old. The current reference points a reviewer
will expect you to know:

| Method | Reference | Why it matters here |
|---|---|---|
| Black-box shift detection (BBSD) | Lipton et al. 2018; Rabanser et al., NeurIPS 2019 "Failing Loudly" | The standard empirical study of shift detection. Shows softmax-output KS with correction is a very strong baseline. **You are essentially doing a weaker version of this without the correction.** |
| Classifier two-sample test | Lopez-Paz & Oquab, ICLR 2017 | Train a discriminator to separate reference from production; AUC>0.5 with a permutation test is a principled, multivariate drift statistic |
| MMD two-sample test | Gretton et al., JMLR 2012 | Kernel test capturing joint (not marginal) shift — your per-feature tests miss interaction drift entirely |
| Conformal test martingales | Vovk et al. | Sequential exchangeability testing with anytime-valid guarantees, the principled successor to CUSUM here |
| Alibi Detect / Evidently | open-source | Reference implementations reviewers know |

**The gap that matters most:** every one of your tests is **univariate and
marginal**. You cannot detect a change in the *dependence structure*
between features. A classifier-2ST or MMD test would. Add one multivariate
detector and you close the largest methodological gap in the drift module.

## F18 — Static split conformal where adaptive conformal is the right tool

You use split conformal calibrated once on val. The project's entire subject
is distribution shift, and split conformal's exchangeability assumption is
precisely what shift violates.

**Adaptive Conformal Inference** (Gibbs & Candès, NeurIPS 2021) updates the
quantile online to maintain coverage under arbitrary shift. There is also
Barber et al. (2023) on conformal prediction beyond exchangeability, and
Tibshirani et al. (2019) on covariate-shift-weighted conformal.

Using static conformal, observing coverage happens to hold, and concluding
robustness is a weaker result than using the method designed for the
problem. This is the most obvious "why didn't you use the current method"
question in the whole repository.

## F21 — MLOps: what a 2026 production-grade repo has and this does not

| Expected | Present | Note |
|---|---|---|
| Experiment tracking (MLflow / W&B) | ✗ | Hand-rolled JSON logs |
| Model registry | Partial | Custom JSON; MLflow Model Registry is the standard |
| CI (GitHub Actions) | ✗ | **A "reliability toolkit" with no automated verification is a contradiction** |
| Containerisation | ✗ | No Dockerfile |
| Pinned dependencies + lockfile | Unclear | Must be exact for reproducibility claims |
| Tests | ✗ | Deleted |
| Data/artifact versioning (DVC) | ✗ | |
| Config management (Hydra) | Partial | Constants live in module headers |
| Pre-commit (ruff/black/mypy) | ✗ | |
| Model card / datasheet | ✗ | Mitchell et al. 2019; Gebru et al. 2021 — expected for clinical ML |

**The sharpest version of this criticism:** the project's thesis is that
production ML needs automated reliability guarantees, and the project itself
has no automated guarantees. A reviewer will enjoy pointing that out.

Minimum credible set: GitHub Actions running lint + a leakage test + a
smoke pipeline run; a Dockerfile; pinned requirements; MLflow (even local
file backend); a model card.

## F22 — Not deployed

FinRiskGuard has a live Hugging Face Space. DriftSentinel — a toolkit about
*production* reliability — has never run outside your laptop. This is the
first thing a reviewer will check and the easiest gap to close.

A monitoring dashboard Space (drift status, alert feed, per-patient
conformal set, retraining trigger log) would demonstrate the thesis rather
than describe it.

## Code structure

The module layout is genuinely good — clean separation, consistent logging,
sensible naming. Three dated patterns:
- constants at module top instead of config objects (Hydra/Pydantic
  settings is current practice)
- `pickle` for model artifacts (fragile across versions; joblib or ONNX)
- procedural `run_*()` scripts rather than composable, testable units with
  dependency injection

None of these are disqualifying. The absence of tests and CI is.

---

# PART 3 — SCIENTIFIC RIGOR

## F19 — No comparison to published results

Strack et al. 2014 is the canonical paper for this dataset and is cited in
your README — but never engaged. Dozens of published models exist. You
report AUC 0.6865 with no context for whether that is good, average, or
poor.

A reviewer cannot calibrate your result without a baseline table. Add one:
your model vs. published results on the same dataset and target.

## F20 — Related work is a bibliography, not a framing

Six references at the bottom, none discussed in text. There is no statement
of what prior work established, what it left open, and where you sit. As
written, the repository reads as disconnected from the literature —
engineering, not research.

**Required structure:** *drift detection [cite] established X; UQ under
shift [cite] established Y; clinical readmission prediction [cite]
established Z; what is missing is the integration into a monitored decision
system; that is our contribution.*

## Novelty claims

To your credit, the README does **not** overclaim novelty — it positions
itself as a toolkit and demonstration. That is defensible and you should
keep it. The unsupported claims are not about novelty but about *mechanism*:
"hospital billing codes change," "patient demographics shift over time,"
"readmission rates drop." These are stated as explanations for observed
shift with no evidence (see F1). Remove or substantiate them.

## Would a clinical ML expert find this convincing?

Partly. Strengths: patient-level splitting is correct and verified;
missingness-as-signal is right; ICD-9 chapter grouping is standard practice.

Weaknesses they will flag immediately:
1. Wrong target (F7) — `<30` is the task
2. `number_inpatient` and derived features make the problem near-tautological
   (F29): you are largely predicting frequent utilisers, which is
   well-known and clinically low-value
3. No calibration-in-the-large / decision curve analysis — the standard
   clinical evaluation (Vickers & Elkin 2006)
4. 97% alert rate (F5) is disqualifying in a clinical setting
5. No discussion of fairness across `race`/`gender`, which for a clinical
   deployment artifact in 2026 is an expected section, not an optional one
6. `payer_code` as a top predictor is an equity red flag — insurance status
   driving clinical risk prediction is exactly what algorithmic-bias
   reviewers look for (cf. Obermeyer et al., Science 2019)

Point 6 deserves emphasis: your single most "important drifted feature" is
insurance code. A reviewer in clinical ML will read that and ask whether
your model is encoding access-to-care disparities as clinical risk. You
currently have no answer.

## F28 — Repeated measures ignored

Patients contribute multiple encounters (46.2% multi-visit rows). Within a
split, rows from the same patient are correlated, so all reported standard
errors and test statistics are anti-conservative. Splitting by patient
handles leakage but not clustering.

**Fix:** cluster-robust CIs, or a patient-level bootstrap.

---

# PART 4 — WHAT WORLD-CLASS LOOKS LIKE

Condensed, per area:

| Area | You have | 2026 top-tier |
|---|---|---|
| Shift detection | 5 univariate marginal tests, no correction | + classifier-2ST / MMD for joint shift; BH-FDR; **known-ground-truth benchmark across shift regimes** |
| Sequential detection | CUSUM + PH, arbitrary params | ARL-calibrated thresholds; false-alarm rate on a no-drift stream; conformal test martingales |
| UQ | Static split conformal, decorative | Adaptive conformal (Gibbs & Candès 2021); UQ **gating a triage decision** with an abstention budget |
| Robustness | FGSM/PGD on trees (non-functional) | Threat-modelled data-quality perturbations; Kantchelian MILP if adversarial framing is kept |
| Decision layer | Cost threshold, arbitrary 5:1 | Decision curve analysis; cost sensitivity sweep; alert-budget constraint |
| Evaluation | One split, seed 42 | Repeated splits, bootstrap CIs, DeLong for AUC deltas, cluster-robust SEs |
| Clinical framing | Merged target, no fairness analysis | `<30` target; subgroup performance; calibration by subgroup; equity discussion of `payer_code` |
| Engineering | Scripts + JSON | CI, tests, container, MLflow, model card, deployed dashboard |
| Positioning | Citation list | Explicit related-work framing and a baseline comparison table |

---

# PART 5 — PRIORITIZED REMEDIATION

Ranked by (credibility damage if left) ÷ (effort to fix).

### Tier 0 — Do before anyone sees the repo (≈1 week)

| # | Action | Why |
|---|---|---|
| F2 | Remove the hardcoded `pipeline_ready = True` | Looks like disabling a quality gate |
| F3 | Delete in-sample v2 val metrics; report test-only with bootstrap CI | Currently an invalid comparison in three artifacts |
| F5 | Add alert-budget constraint; report feasible operating region | 97% flag rate is disqualifying |
| F4/F16 | Recalibrate L1 to a target clean-trigger rate; remove L3/L4; report detection at matched FPR | Headline metric currently conceals a 94% false-flag rate |
| F15 | Drop "independent"; group signals into families | Most quotable number in the README is an overclaim |
| F27 | Fix README/artifact inconsistencies (PSI 0.93 vs 0.84 vs 0) | Every number must be traceable to one named window |

### Tier 1 — The scientific core (≈2–3 weeks)

| # | Action | Why |
|---|---|---|
| F1 | Run the censoring diagnostic; re-characterise the shift; rewrite the narrative honestly | The project's central claim depends on it |
| F6 | Build the 4-regime × 8-signal ground-truth benchmark | Converts "we detected something" into a detection study; **highest value per hour in the audit** |
| F8 | Repeat over ≥20 seeds; mean ± std; bootstrap CIs | You already hold this standard for CellTriage |
| F7 | Switch primary target to `<30`; keep merged as secondary | Aligns with the clinical task and the literature |
| F10 | BH-FDR across all drift tests | 30-minute fix, removes an obvious attack |
| F13/F14 | Decontaminate threshold selection and conformal calibration | Both currently circular |

### Tier 2 — Modernisation (≈2 weeks)

| # | Action |
|---|---|
| F17 | Add classifier-2ST and MMD as multivariate detectors; benchmark against your classical suite in the F6 framework |
| F18 | Implement adaptive conformal (Gibbs & Candès) alongside split conformal; compare coverage under shift |
| F12 | Make conformal sets gate a three-way triage decision with an abstention budget |
| F11 | Replace FGSM/PGD with threat-modelled data-quality perturbations; state the threat model explicitly |
| F9 | Repair or honestly re-describe the selection pipeline; add the ablation |
| F23 | FE ablation: raw-only vs raw+FE, with CIs |
| F19/F20 | Baseline comparison table + a real related-work section |

### Tier 3 — Engineering credibility (≈1 week)

CI (lint + leakage test + smoke run), Dockerfile, pinned deps, MLflow,
model card, fairness/subgroup analysis, deployed monitoring Space.

### Caveat rather than fix

Some things are cheaper to state honestly than to solve:

- **Single dataset.** Say so; note that external validation on MIMIC or
  eICU is future work.
- **No true timestamps.** After F1, state plainly that temporal ordering is
  unverifiable in this dataset and that you therefore study cohort shift.
- **AUC ≈ 0.65.** Do not apologise. State that readmission is intrinsically
  hard, cite the published range, and note that the project's contribution
  is reliability infrastructure, not discrimination.
- **Adversarial robustness on tabular clinical data.** If you keep it,
  caveat that no realistic adversary exists and that the value is as a
  perturbation-sensitivity study.

---

# WHAT IS ACTUALLY GOOD — KEEP THIS

An audit that finds only faults is not calibrated. These are genuine
strengths and you should not refactor them away:

1. **Patient-level splitting is correctly implemented and verified.**
   Train∩Val=0, Train∩Test=0, Val∩Test=0, asserted in code. Most people get
   this wrong on this dataset.
2. **Module architecture is clean.** The `data / features / models / drift /
   uncertainty / adversarial / monitoring` decomposition is how a senior
   engineer would organise it.
3. **Logging discipline is unusually good.** Every module writes structured,
   readable logs. This is what made the audit possible.
4. **Missingness-as-signal** for `weight`, `A1Cresult`, `max_glu_serum` is
   correct clinical practice.
5. **Fit-on-train-only** is genuinely enforced in the preprocessor and
   encoders.
6. **Calibration module is correct.** ECE 0.2054 → 0.1173 via isotonic is a
   real, properly executed result.
7. **The registry retraining loop** is the right conceptual design, even
   though the comparison inside it is broken.
8. **Breadth.** Drift + UQ + robustness + monitoring in one coherent
   repository is more than most portfolio projects attempt.

The repository is a strong engineering artifact with weak scientific
controls. That is a fixable asymmetry, and it is the more favourable of the
two directions to be wrong in.

---

# THREE QUESTIONS TO PREPARE FOR

A reviewer will ask these. Have answers before the interview.

1. *"How do you know `encounter_id` is temporally ordered, and how did you
   rule out right-censoring as the cause of your label shift?"*
2. *"Would your eight evidence signals fire on a random split?"*
3. *"Your defense system flags 94% of clean traffic and your cost-optimal
   threshold flags 97% of patients. What operating point would you actually
   deploy?"*

If you can answer all three with evidence, the project is InnoCORE-grade.
