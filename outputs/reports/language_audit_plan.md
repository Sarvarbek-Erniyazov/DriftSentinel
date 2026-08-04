# Phase 0.5 — Language correction plan (NOT APPLIED)

Policy: **surgical**. Phase 0.1 verdict: ordering **SUPPORTED**, split validity **NOT_TEMPORAL_BY_CONSTRUCTION**.

- occurrences: **134** across **15** files
- `KEEP`: 91  — refers to encounter_id chronology, which Phase 0.1 evidenced, or is part of the investigation itself
- `MANUAL_REVIEW`: 21  — ambiguous from the line alone — decide in context
- `REWRITE_MECHANISM`: 7  — asserts a CAUSE for the observed shift; Phase 0.1 evidenced chronology, not mechanism
- `REWRITE_SPLIT`: 15  — describes the SPLIT as temporal; the split sorts patients by entry cohort, so this is the claim Phase 0.1 refuted

## Proposed changes

### REWRITE_SPLIT (15)

- `CLAUDE.md:18` → `entry-cohort split`
  - `The project currently claims a **patient-level temporal split** and`
- `README.md:54` → `entry-cohort split`
  - `| **Split strategy** | Patient-level **entry-cohort** split — zero leakage, but *not* a temporal split (patients sorted by first `encounter_id`, so a 1999 entrant's 2008 encounters sit in train) |`
- `docs/01_eda_and_data_pipeline.md:144` → `entry-cohort split`
  - `ordering**, not a temporal split.`
- `docs/01_eda_and_data_pipeline.md:148` → `entry-cohort split`
  - `> That does **not** make this a temporal split: sorting *patients* by their`
- `docs/01_eda_and_data_pipeline.md:152`
  - `> `temporal` regime in `src/investigation/split_regimes.py`.`
- `docs/AUDIT.md:61`
  - `**The claim.** The README asserts a patient-level *temporal* split, and`
- `docs/AUDIT.md:83` → `entry-cohort split`
  - `temporal split. You split on *first* encounter_id per patient, which sorts`
- `docs/AUDIT.md:696`
  - `- **No true timestamps.** After F1, state plainly that temporal ordering is`
- `docs/REMEDIATION_PLAN.md:52`
  - `SUPPORTED / NOT SUPPORTED / INCONCLUSIVE for chronological ordering, each`
- `docs/REMEDIATION_PLAN.md:453`
  - `- Initial framing: patient-level "temporal" split, 8/8 signals, CRITICAL`
- `src/data/splitter.py:9` → `entry-cohort ordering`
  - `- Order patients by their FIRST encounter_id (temporal proxy)`
- `src/data/splitter.py:39` → `entry-cohort structure`
  - `This preserves temporal structure without requiring explicit timestamps.`
- `src/data/splitter.py:85` → `entry-cohort split`
  - `Patient-level temporal split of raw DataFrame.`
- `src/data/splitter.py:101` → `entry-cohort split`
  - `logger.info(f"Strategy       : patient-level temporal split")`
- `src/data/validator.py:282` → `entry-cohort split`
  - `f"monotonic_increasing={is_monotonic} (required for temporal split)")`

### REWRITE_MECHANISM (7)

- `CLAUDE.md:38` → `[REMOVE — mechanism claim without mechanism evidence]`
  - `"hospital billing codes changed" or "patient demographics shifted over`
- `README.md:39` → `[REMOVE — mechanism claim without mechanism evidence]`
  - `> attributed the shift to changing hospital billing codes and shifting patient`
- `docs/04_drift_detection.md:17` → `[REMOVE — mechanism claim without mechanism evidence]`
  - `> *causes* — shifting demographics, changing billing codes, falling disease`
- `docs/AUDIT.md:63` → `later-entering patients`
  - `"the world keeps changing," "newer patients have different insurance`
- `docs/AUDIT.md:112` → `[REMOVE — mechanism claim without mechanism evidence]`
  - `- the clinical framing ("hospital billing codes changed," "readmission`
- `docs/AUDIT.md:590` → `[REMOVE — mechanism claim without mechanism evidence]`
  - `"hospital billing codes change," "patient demographics shift over time,"`
- `docs/REMEDIATION_PLAN.md:121` → `[REMOVE — mechanism claim without mechanism evidence]`
  - ``drift began`, `newer patients`, `billing codes`, `demographics shift`.`

### MANUAL_REVIEW (21)

- `CLAUDE.md:109`
  - `> initial temporal-drift framing → falsification testing → discovery of the`
- `docs/01_eda_and_data_pipeline.md:147`
  - `> chronological (`outputs/reports/temporal_validity.json`, verdict SUPPORTED).`
- `docs/01_eda_and_data_pipeline.md:151`
  - `> A genuine chronological split of *encounters* is implemented separately as the`
- `docs/03_model_training.md:151`
  - `> finding — model performance under temporal drift". Two corrections: this is an`
- `docs/03_model_training.md:152`
  - `> entry-cohort contrast, not a temporal one; and Tier 0 showed the same pipeline`
- `docs/04_drift_detection.md:23`
  - `> `outputs/reports/temporal_validity.json`.`
- `docs/04_drift_detection.md:122`
  - `![Target shift across entry-cohort windows](../outputs/figure/13_target_temporal_shift.png)`
- `docs/04_drift_detection.md:124`
  - `says "temporal"; renaming a generated artifact is a pipeline change, not a`
- `docs/AUDIT.md:71`
  - ``encounter_id_monotonic = False`. You are *assuming* it is chronologically`
- `docs/AUDIT.md:124`
  - `> We initially framed the observed shift as temporal concept drift. On`
- `docs/REMEDIATION_PLAN.md:25`
  - `Create `src/investigation/temporal_validity.py`.`
- `docs/REMEDIATION_PLAN.md:29`
  - `temporal anchors** — clinical practice patterns whose timing is known`
- `docs/REMEDIATION_PLAN.md:33`
  - `the 2000s. If ordering is chronological, `A1Cresult` non-missing rate`
- `docs/REMEDIATION_PLAN.md:37`
  - `If ordering is chronological, the rosiglitazone rate should fall in the`
- `docs/REMEDIATION_PLAN.md:51`
  - `**Acceptance:** `outputs/reports/temporal_validity.json` with a verdict of`
- `docs/REMEDIATION_PLAN.md:57`
  - `names, docs, README. Do not leave the word "temporal" anywhere it is not`
- `docs/REMEDIATION_PLAN.md:120`
  - `Apply the Phase 0.1 verdict everywhere. Grep for `temporal`, `over time`,`
- `docs/REMEDIATION_PLAN.md:297`
  - `over time, interval/set size, and recovery speed after an induced shift.`
- `docs/REMEDIATION_PLAN.md:539`
  - `Begin with **Phase 0.1**. Print your plan for the temporal validity`
- `src/pipelines/pipeline.py:272`
  - `"scenario. NOT temporal drift: see "`
- `src/pipelines/pipeline.py:273`
  - `"outputs/reports/temporal_validity.json"),`

### KEEP (91)

- 91 occurrences in: `configs/split_regimes.yaml`, `configs/temporal_validity.yaml`, `docs/AUDIT.md`, `src/investigation/language_audit.py`, `src/investigation/split_regimes.py`, `src/investigation/temporal_validity.py`
