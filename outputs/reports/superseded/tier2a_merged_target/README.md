# Superseded: all artifacts under the MERGED target

Every file here was produced with `readmitted_binary = {"NO":0, "<30":1, ">30":1}`
(46.1% prevalence). Tier 2A.1 switched the primary target to 30-day readmission,
`{"NO":0, "<30":1, ">30":0}` (11.16% prevalence).

These are preserved, not deleted, because:

* the README previously LABELLED the target "readmission within 30 days" while
  the code computed the merged version — these artifacts are the evidence of
  that discrepancy, not just of the old numbers;
* the merged target remains a documented secondary analysis (derivable as
  `readmitted_multi > 0`);
* Tier 1.1-1.5 corrections (registry DeLong test, defense confusion matrix,
  threshold PPR, label-drift rule, signal retirement) were all computed on the
  merged target and must remain inspectable alongside their 30-day successors.

Inventory of what was regenerated and in what order:
`docs/TIER_2A_REGENERATION_INVENTORY.md`
