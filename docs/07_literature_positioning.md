# Literature Positioning

← [Back to README](../README.md)

Evidence: [`outputs/reports/literature_baselines.json`](../outputs/reports/literature_baselines.json)
· regenerate with `python src/investigation/literature_baselines.py`

---

## 1. What is not new here

Stating this first, plainly, because a specialist will know it within one
paragraph and the alternative is to look as though we did not:

- **The dataset is not novel.** UCI Diabetes 130-US Hospitals is one of the most
  heavily used public healthcare tabular benchmarks. It is a standard teaching
  and benchmarking set.
- **The task is not novel.** 30-day readmission prediction on this dataset is
  well-trodden — Strack et al. (2014) released it for exactly this question, and
  a continuous stream of comparison papers has followed.
- **Drift detection on this dataset is not novel either**, and more importantly
  it is not even well-posed without additional work: the dataset has no date
  column, so any drift study on it must first establish what "later" means. Most
  do not.
- **None of the detectors implemented here are new.** PSI, KS, χ², CUSUM,
  Page-Hinkley, split conformal, classifier two-sample testing, MMD, BBSD and
  Benjamini–Hochberg are all standard, and all are cited to their sources.

No result in this repository is a claim of methodological novelty. If the
contribution had to survive on "we detected drift," it would not survive.

## 2. What prior work established, and what it left open

| Study | What it established | What it left open |
|---|---|---|
| **Strack et al. (2014)**, *BioMed Res. Int.* 781670 | Released the dataset and the `<30` / `>30` / `NO` encoding. An **etiologic** study: multivariable logistic regression estimating the association between HbA1c measurement and early readmission, adjusted for covariates. Reports odds ratios. | **Reports no discrimination metric at all** — no AUC, no ROC. It is routinely cited as a predictive baseline on this dataset. **It is not one, and this project does not cite it as one.** It remains the correct citation for the dataset's provenance and for the target encoding; borrowing predictive authority from a paper that never made the claim would be the opposite of what this section is for. |
| **Liu, Sue & Wu (2024)**, *J. Med. AI* 7:23 | Seven models on the `<30` target at 11.2% prevalence, with **patient-grouped 5-fold CV** — encounters for one patient kept inside a single fold. AUROC 0.48 (SVM-RBF) to **0.64 (XGBoost, CI 0.64–0.65)**; tree ensembles and logistic regression all 0.62–0.64. | Establishes the achievable band under a correct grouping protocol. Does not ask whether the band moves when the *split regime* changes, because there is only one regime. |
| **Salim & Ibrahim (2026)**, *Healthcare* 14(9):1185 | Same dataset, same target, same 11.16% prevalence. Nested stratified 5-fold CV. XGBoost **0.664** (0.688 calibrated), LR 0.657, RF 0.650. | States as a limitation that the cross-validation **did not enforce patient-level splits**. Same task, higher number, one protocol dimension different. |

**The open question these three share.** All of them report a number. None of
them reports what that number would have been under a different split of the
same data. On a dataset where 46.2% of patients contribute more than one
encounter and where the ordering variable is an undated surrogate key, that is
the dimension along which the results actually move.

## 3. The result this section is built on

**Lead with this, not with the comparison table.** It is the strongest
quantitative claim the project makes, and it is a *controlled* one.

Hold the model, the features, the code and the seed fixed. Move only the split
regime:

| Regime | AUC | Δ vs deployed |
|---|---|---|
| Patient-grouped CV | **0.6806** [0.6759, 0.6851] | **+0.0396** |
| Random patient split (20 seeds) | **0.6785** ± 0.0067 | **+0.0376** |
| Entry-cohort held-out window (deployed) | **0.6410** [0.6255, 0.6556] | — |

**Nothing about the model changed.** A ~0.04 AUC swing is produced entirely by
how the data was divided. For scale, the gap between the best and worst
*tree-based* model across the two published comparison studies below is about
**0.024** — smaller than the effect of the split alone.

Two consequences, stated at deliberately different strengths:

- **Claimed.** A single AUC reported without naming its split regime is not
  comparable across papers on this dataset. Within this repository — one
  variable moving, everything else fixed — protocol moves the number by more
  than model choice plausibly does.
- **Not claimed.** That the difference between the two published numbers *is*
  leakage. Two studies is not a sample, and they differ in more than the
  grouping dimension. That the higher number comes from the study which states
  it did not group by patient is offered as **consistent with** the controlled
  result, never as evidence for it.

This is also why the project's own headline AUC is reported with its regime
attached everywhere it appears.

## 4. Baseline comparison

External rows are curated by hand and carry a DOI, a URL and a read date.
Internal rows are read at runtime from generated artifacts and are never typed —
if `headline_metrics_ci.json` changes, this table changes. See §8.

| Source | AUC | Protocol | Patient-grouped? |
|---|---|---|---|
| Strack et al. (2014) | *not reported* | logistic regression, ~70k filtered encounters | — |
| Liu, Sue & Wu (2024) | **0.64** [0.64, 0.65] | group 5-fold CV | ✅ yes |
| Salim & Ibrahim (2026) | **0.664** | nested stratified 5-fold CV | ❌ **no** (stated limitation) |
| **DriftSentinel** — deployed | **0.6410** [0.6255, 0.6556] | entry-cohort split, held-out test window | ✅ yes |
| **DriftSentinel** — protocol-matched | **0.6806** [0.6759, 0.6851] | StratifiedGroupKFold(5) × 2 | ✅ yes |
| **DriftSentinel** — random-patient control | **0.6785** ± 0.0067 | random patient split, 20 seeds | ✅ yes |

### 4.1 Where this project sits

Inside the published band on the deployed split; at or slightly above its upper
edge under the protocol-matched grouped CV. **Discrimination is not the
contribution and is not presented as one** — see §5, and the Limitations note on
the AUC ceiling for this task.

## 5. The contribution, stated precisely

> Not the model. Not the detectors. **The negative-control methodology and the
> systematic falsification design** — every phase carrying a stated condition
> under which the method *must* fire, run before the real result is interpreted.

The reason to build it this way is narrow and specific: **a null result is only
information if the instrument that produced it has been shown capable of
producing a positive one.** Without that, "we found no drift" and "our detector
is broken" are the same observation. Most applied drift work reports the former
and never tests for the latter — and this project's own original version was an
instance of that failure, reporting `8/8 signals CRITICAL` with no no-drift
baseline, of which two signals were later found to be structurally broken.

Every phase therefore ships a falsification arm:

| Phase | The null or claim | The condition under which it MUST fire | Result |
|---|---|---|---|
| **0.2** Random-split control | drift detected on the deployed split | — | signals fire at 0.25/6 mean where drift is impossible by construction, vs 3.30/6 on the entry-cohort split |
| **0.3** Synthetic positive controls | detectors are diagnostic | covariate / label / concept shift injected at swept magnitudes | power curves per signal; signals firing under all three are re-described as general alarms, not mechanism evidence |
| **2A.3** BH–FDR | "31/53 features drifted" | 318 tests corrected at q=0.05 | correction removes almost nothing; the binding constraint is **effect size**, not multiplicity |
| **2B.1** Adaptive conformal | ACI is unnecessary here | a synthetic hard label shift where static conformal must fail | static collapses to 0.412 coverage, ACI holds 0.898 — so the null on real data is a property of the data, not the implementation |
| **2B.2** Multivariate detection | the windows differ multivariately | all three detectors run on the random control | all three silent on the control, all three fire on entry-cohort and temporal |
| **2B.3** Uncertainty-gated triage | the gate routes what the model gets wrong | 10% of test labels replaced with noise; a working gate must route them more often | lift −0.004 — the gate cannot see label noise, and this is stated rather than hidden |
| **2B.4** Robustness | the model degrades gracefully | a destructive control that must register damage | destructive −0.0456 AUC vs worst realistic −0.0370 — the harness can detect damage when damage is real |
| **2C.1** Fairness audit | no racial disparity in discrimination is supportable | an injected disparity in one subgroup that the audit must flag | injected gap 0.1313 detected; real gap 0.0091 not — so the null is a measurement, not a blind spot |

Because each arm exists, this project's nulls are interpretable. That is the
whole of the claim, and it is a methodological one rather than an algorithmic
one.

## 6. Modern methods, positioned honestly

Four modern methods were implemented as the plan required, and each returned a
result. They resolve into **three findings — two nulls and one positive** (the
classifier two-sample test and MMD are two instruments answering one question, so
they are counted as one finding). **The nulls are reported as findings rather
than buried.**

Adopting a modern method and discovering it is unnecessary *on this data* is a
finding. Adopting it, finding it changes nothing, and presenting it anyway as
evidence of sophistication is method-shopping — which is the reflex this
remediation exists to correct.

### 6.1 Adaptive Conformal Inference — **NULL**

Gibbs & Candès (NeurIPS 2021). Static split conformal assumes exchangeability,
which is exactly what shift violates; ACI updates the quantile online to hold
coverage under arbitrary shift.

| Stream | Static coverage | ACI coverage | ACI mean set size |
|---|---|---|---|
| val audit half (held out) | 0.8943 | 0.8996 | 1.059 |
| test (entry-cohort) | 0.9134 | 0.9004 | 0.991 |
| **synthetic hard label shift** | **0.4120** | **0.8980** | 1.768 |
| changepoint at 50% | 0.6654 | 0.8987 | 1.378 |

On the real streams ACI buys **+0.013 coverage**, i.e. essentially nothing —
because static split conformal already holds. The val→test difference is cohort
composition and observation-window truncation, not a change in P(Y|X), so
exchangeability here is *bent, not broken*.

The prediction was **registered before the run**: static conformal already held
at 0.8943/0.9134 under the Tier 2A.4 decontaminated protocol, so ACI was
predicted to add little. The synthetic hard shift was included specifically as
the falsification condition, and it fires: static collapses to 0.412 while ACI
recovers to 0.898, moving α from 0.100 to 0.0051. **The null is therefore a
property of the data, not of the method or of our implementation of it** — a
distinction that is unavailable without the falsification arm.

ACI remains the correct default for a deployment whose shift is not known in
advance to be this benign. It is retained on that basis, and that basis is
stated.

### 6.2 Classifier two-sample test and MMD — **POSITIVE**

Lopez-Paz & Oquab (ICLR 2017); Gretton et al. (JMLR 2012).

Every shipped detector was univariate and marginal, so no change in the
*dependence structure* between features was detectable at all. This is the one
genuine capability gap, and closing it is the positive result:

| Regime | C2ST held-out AUC | C2ST perm. p | MMD² | MMD perm. p | Fires? |
|---|---|---|---|---|---|
| Random (negative control) | 0.5067 | 0.095 | 0.000183 | 0.184 | ❌ silent |
| Entry-cohort | **0.6932** | 0.048 | 0.002422 | 0.005 | ✅ |
| Temporal | **0.6792** | 0.048 | 0.002112 | 0.005 | ✅ |

Both detectors are **silent on the control and fire on both real regimes** —
which is precisely the calibration property the original 8-signal suite lacked.

The C2ST's held-out AUC is also an **effect size**: 0.693 says how
distinguishable the two windows are, on a bounded scale, from one number. That
directly answers the deficiency the FDR analysis exposed (below), where 141
feature-windows were statistically significant and 129 of them had PSI < 0.10.

Qualification, stated rather than glossed: on *this* data the multivariate tests
did not reveal drift the marginal tests missed — the marginals had already
flagged 43–50 of 53 features. The value delivered here is calibration and a
single interpretable effect size; the capability to catch a pure
dependence-structure change is real but was not exercised by this dataset.

### 6.3 Black-box shift detection with FDR — **NULL**

Lipton et al. (ICML 2018), benchmarked in Rabanser et al. (NeurIPS 2019,
*Failing Loudly*). Univariate tests on **model outputs** with multiple-testing
correction — the standard strong baseline that the shipped `prediction_drift`
signal approximates *without* the correction.

Implemented properly, it fires exactly where the uncorrected version fired
(KS p = 2.9e-62 on entry-cohort) and stays silent on the random control. Across
the full 318-test family, Benjamini–Hochberg at q=0.05 changes the verdict on
**one test at a matched α level** (269 → 268 at 0.05; 257 → 256 at 0.01).

**Adopting the principled version did not overturn a single conclusion.** The
honest reading is that multiplicity was never the binding constraint on this
evidence: with ~20k rows per window the tests are so overpowered that negligible
differences reach p < 1e-10. Of 141 feature-windows surviving FDR, **129 have
PSI < 0.10** — statistically certain and practically meaningless.

The correction ships anyway, because a project that reports 318 uncorrected
tests has no standing to argue that multiplicity did not matter *before*
measuring it. But it is reported as a null, and the real lesson — significance
plus a minimum effect, never significance alone — is reported as the finding.

### 6.4 Summary

| Method | Reference | Result | Kept? |
|---|---|---|---|
| Adaptive conformal inference | Gibbs & Candès 2021 | **null** — no gain on real streams; recovers on synthetic hard shift | yes, as the safe default |
| Classifier two-sample test | Lopez-Paz & Oquab 2017 | **positive** — closes the multivariate gap; calibrated on the control | yes |
| MMD two-sample test | Gretton et al. 2012 | **positive** — same pattern, kernel-based | yes |
| BBSD + Benjamini–Hochberg | Lipton 2018 / Rabanser 2019 | **null** — changes one verdict in 318 | yes, as discipline |

Four methods, three findings: **two nulls (§6.1, §6.3) and one positive (§6.2).**

## 7. Fairness: the `payer_code` question, and the Obermeyer frame

The audit raised it as an objection that would be made by any clinical ML
reviewer, so it is answered with a measurement rather than deflected.

**Obermeyer et al. (*Science*, 2019)** is the canonical demonstration of the
failure mode: an algorithm affecting millions of patients used *healthcare cost*
as a proxy for health need. Because less money is spent on Black patients at a
given level of illness, the algorithm systematically under-referred them —
without race being a feature. The lesson is not "do not use administrative
variables"; it is that a variable encoding **access to care** will be learned as
though it encoded **clinical severity**, and the resulting disparity is invisible
unless you look for it.

`payer_code` — insurance status — is structurally that kind of variable, and it
is both among the most drifted features (PSI 0.84–0.93 between windows) and
present in the model.

**The measured answer:** under the 30-day target, `payer_code` ranks **17 of 53**
by gain importance. It is not among the top five, which are
`discharge_disposition_id`, `FE_has_prior_inpatient`, `FE_lab_to_procedure_ratio`,
`FE_labs_per_day` and `FE_meds_per_day`.

This is a materially weaker version of the concern than the audit assumed — the
audit was working from the merged target, under which `payer_code` was a top
feature. It is not a dismissal:

- Rank 17 of 53 is still inside the model and still contributing.
- The top features are dominated by **prior-utilisation** variables, which makes
  the task **partly tautological**: the model substantially identifies frequent
  utilisers. Frequent utilisation is itself patterned by access, so the Obermeyer
  mechanism is not avoided by demoting `payer_code` — it is relocated.
- The disparity actually measured is **by age, not by insurance or race** — see
  the model card. The model is best at the group that needs it least.

That last point is the one that matters: the concern the audit raised turned out
to be smaller than expected, and a *different*, larger, quantified disparity was
found by looking. Reporting both is the position; reporting only the reassuring
half would not be.

## 8. How this table stays honest

R4 forbids hand-typed results. A literature comparison is the one table that
cannot be regenerated from this repository, because its inputs are in other
people's papers. Rather than exempt it, the table is split:

- **External rows** are declared as a reviewable constant in
  `src/investigation/literature_baselines.py`, each with DOI, URL, the exact
  protocol, and the date the value was read (`2026-08-04`).
- **Internal rows** are read at runtime from the named generated artifacts by
  JSON path. Nothing is typed. A missing key **raises** rather than defaulting —
  a fallback default would be indistinguishable from a real measurement (R6).
- **The contrast** between the two halves is computed, not asserted.

## 9. References

- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*, 51(2), 816–845.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289–300.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*, 44(3), 837–845.
- Gebru, T., et al. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
- Gibbs, I., & Candès, E. J. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *JMLR*, 13, 723–773.
- Lipton, Z. C., Wang, Y.-X., & Smola, A. (2018). Detecting and correcting for label shift with black box predictors. *ICML*.
- Liu, V. B., Sue, L. Y., & Wu, Y. (2024). Comparison of machine learning models for predicting 30-day readmission rates for patients with diabetes. *Journal of Medical Artificial Intelligence*, 7, 23. doi:10.21037/jmai-24-70
- Lopez-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests. *ICLR*.
- Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**, 220–229.
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453.
- Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing loudly: an empirical study of methods for detecting dataset shift. *NeurIPS*.
- Salim, S. S., & Ibrahim, A. A. (2026). A machine learning approach for predicting 30-day hospital readmission in patients with diabetes. *Healthcare*, 14(9), 1185. doi:10.3390/healthcare14091185
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*, 2014, 781670. doi:10.1155/2014/781670
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis. *Medical Decision Making*, 26(6), 565–574.
