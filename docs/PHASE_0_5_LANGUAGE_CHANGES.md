<!-- Phase 0.5 -->

> **Note on two removed files.** This record cites `CLAUDE.md` and
> `docs/REMEDIATION_PLAN.md` by line number. Both were **build-tooling artifacts**
> — the working brief and the phased plan used to drive the remediation — and were
> removed from the repository once the work they described was complete. They were
> never research documentation and are not part of the evidence trail.
>
> The citations are left as they were rather than deleted: this document is a
> record of a scan performed at a point in time, and rewriting its findings to
> match a later repository state would make it a worse record, not a better one.
> The same applies to the generated artifacts behind it —
> `outputs/reports/language_audit_plan.{json,md}` and
> `temporal_language_inventory.json` still list both files among the 17 scanned,
> because **re-running the scan against a smaller repository would change the
> documented occurrence counts and quietly invalidate the numbers this record
> reports.** An evidence artifact that is regenerated to agree with the present is
> no longer evidence about the past.
>
> `docs/AUDIT.md` — the adversarial audit that drove all of this — is kept.

# Phase 0.5 — Language correction: applied changes and reasoning

Policy: **surgical**, approved after Phase 0.1 returned `SUPPORTED` for
chronological ordering of `encounter_id`.

**The principle, applied uniformly:** "temporal" survives only where it refers to
evidenced `encounter_id` chronology or to the investigation itself. Everywhere it
describes the **split**, it becomes *entry-cohort*. Mechanism claims are removed
regardless of the ordering verdict, because Tier 0 evidenced chronology, not causes.

Machine-readable inventory: `outputs/reports/language_audit_plan.json`
(regenerate with `python -m src.investigation.language_audit`).

---

## A bug in the inventory itself — found during application

The first inventory reported 98 occurrences in 13 files and **missed
`src/data/splitter.py` entirely** — the single most important file for this
rename. The skip-list matcher used a substring test:

```python
if any(rel.startswith(s) or f"/{s}/" in f"/{rel}" for s in skip_dirs):
```

with `data` in `skip_dirs` (intended: the top-level dataset directory). `"/data/"`
also matches `"/src/data/splitter.py"`, so the whole `src/data/` package was
silently excluded. Fixed to prefix-anchored matching in both
`src/investigation/language_audit.py` and `src/investigation/temporal_validity.py`;
5 further occurrences became visible, including the 4 in `splitter.py` that
describe the split as temporal in its module docstring, its function docstring,
and its runtime log line.

This is worth recording: a silent scope gap in an audit tool is the same class of
defect as the ones the audit was written to find.

---

## Files NOT edited, and why

`docs/AUDIT.md` and the remediation plan were **historical records**. Their
bodies are preserved verbatim; each received a resolution banner at the top
instead. The preservation rule — *never delete evidence of the original
framing* — makes editing them the wrong move: the audit's F1(a) is now known to be
wrong as framed, and a reader must be able to see both what was claimed and what
was found. 24 of the flagged occurrences live in these two files and were left
alone by this rule.

`outputs/` was excluded from the scan: it holds generated artifacts that are
rewritten by re-running the pipeline, not hand-edited. (Scanning it also made the
inventory self-referential — this report contains the word "temporal" many times,
and each run inflated the next run's count: 95 → 371 → 14,433 observed.)

---

## MANUAL_REVIEW — all 21, with the decision for each

Reviewable individually; the reasoning is the same principle in every row.

| # | Location | Decision | Reasoning |
|---|---|---|---|
| 1 | build-tooling file, since removed: "initial temporal-drift framing → falsification…" | **KEEP** | Describes the *arc*. "Initial framing" is historically accurate and is the thing being corrected |
| 2 | `README.md:46` "under temporal shift" | **REWRITE** → "under entry-cohort shift" | Describes the val→test contrast on the entry-cohort split |
| 3 | `docs/03_model_training.md:148` "under temporal drift" | **REWRITE** → entry-cohort framing + no-drift baseline note | Same contrast; also needed the 2.15/8 baseline caveat |
| 4 | `docs/04_drift_detection.md:115` `![Target Temporal Shift](…13_target_temporal_shift.png)` | **REWRITE alt text; KEEP filename** | Renaming a *generated artifact* is a pipeline change, not a language change. Deferred to Tier 1 with an inline comment so it is not lost |
| 5–6 | `docs/AUDIT.md:71,124` | **KEEP** (banner) | Historical record — see above |
| 7–15 | remediation plan, 9 occurrences (file since removed) | **KEEP** (banner) | Historical record. Several also referred to `temporal_validity.py`, which is the investigation and earns the term |
| 16 | `src/drift/concept_drift.py:3` "degradation over time windows" | **REWRITE** → "across sequential evaluation windows" | The windows are index-based slices of a concatenated stream, not calendar intervals. "Over time" overstates what is measured |
| 17 | `src/drift/concept_drift.py:155` "Simulates temporal performance monitoring" | **REWRITE** → "sequential", + explicit note that windows are index-based | Same reason; this is the function that actually cuts the windows |
| 18–19 | `src/models/evaluator.py:286,444` "Across Temporal Windows" (figure titles) | **REWRITE** → "Across Sequential Windows" | Axis is window index. A reader would otherwise read the x-axis as calendar time |
| 20 | `src/uncertainty/threshold.py:356` "threshold drift across temporal windows" | **REWRITE** → "sequential evaluation windows" | Same |
| 21 | `src/pipelines/pipeline.py:205` "expected temporal drift signals" | **REWRITE** → "expected entry-cohort shift signals" | Refers to the split |

## REWRITE_SPLIT (15) — applied

`README.md` (split-strategy row, repo tree), `docs/01_eda_and_data_pipeline.md`
(section heading + strategy line, with a note on why verified chronology still
does not make this a temporal split), `src/data/splitter.py` (module docstring,
strategy bullet, `_get_patient_order` docstring, `split()` docstring, runtime log
line), `src/data/validator.py` (the `encounter_id_monotonic` check no longer
claims to be "required for temporal split" — the entry-cohort split does not
require it), `src/pipelines/pipeline.py`. The build-tooling brief was rewritten more
substantially: it is a live instruction file whose "open question" section would
have misdirected future sessions.

## REWRITE_MECHANISM (7) — applied

`README.md` "The Problem" section (removed: "the world keeps changing", "patient
demographics shift over time", "hospital billing codes change"; replaced with the
window contrast actually measured, plus a note recording that the claims were
removed and why). `docs/04_drift_detection.md` intro and the `number_diagnoses`
interpretation row. The prohibition on such prose was strengthened from
conditional to absolute.

---

## Discrepancy found while editing — not previously recorded anywhere

`README.md` stated **"Target: Hospital readmission within 30 days (binary)"**
while the code computed `readmitted_binary = (readmitted != "NO")` — the *merged*
`<30`-or-`>30` target at 46.1% prevalence. The README described a target the
pipeline never implemented.

Audit F7 flagged the merged target as the wrong choice but did not notice the
README claimed the correct one. Every published number therefore sat under a
target label that did not match its computation. Marked as superseded in the
README rather than silently corrected, since the numbers beneath it are still the
merged-target numbers until the Tier 1 re-run.
