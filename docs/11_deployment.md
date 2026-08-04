<!-- Tier 3 -->

# Deployment — Streamlit Community Cloud

← [Back to README](../README.md)

Build is complete and verified locally. **Connecting `share.streamlit.io` is the
only remaining step and is left to the repository owner.**

---

## 1. What to enter on share.streamlit.io

| Field | Value |
|---|---|
| Repository | `Sarvarbek13/DriftSentinel` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.12` |

Nothing else needs configuring. Community Cloud installs the repository-root
`requirements.txt` automatically and reads `.streamlit/config.toml` for the
theme and server settings.

## 2. Why the root `requirements.txt` is small

Community Cloud installs **the root `requirements.txt` and nothing else**, into
a container with roughly 1 GB of memory. The full research environment —
lightgbm, scikit-learn, shap, scipy, seaborn, matplotlib — is several hundred
megabytes installed and **none of it is needed to serve the console**.

| File | Contents | Installed by |
|---|---|---|
| `requirements.txt` | streamlit, pandas, numpy, pyarrow, altair | Streamlit Community Cloud |
| `requirements-dev.txt` | the above **plus** the full research stack and pytest | CI, and anyone reproducing the artifacts |

`requirements-dev.txt` begins with `-r requirements.txt`, so one install covers
both and the two files cannot drift into disagreement about the runtime.

**This is enforced, not documented.** `tests/test_console_runtime.py` makes
`sklearn`, `lightgbm`, `shap`, `scipy`, `seaborn` and `matplotlib`
**unimportable**, then imports `app.py`. Importing the module executes the whole
Streamlit script top to bottom, so the test exercises every `st.*` call, not just
the import line. A separate assertion checks that the guard itself blocks — a
test that passes because nothing was blocked would prove nothing (CLAUDE.md R6).

CI runs this in its own `console` job that installs **only** the root
`requirements.txt`, so the deploy environment is reproduced in CI rather than
assumed.

## 3. Why the console loads no model

`app.py` reads a precomputed evidence bundle. It never unpickles a model.

| | |
|---|---|
| `app/demo_data/evidence.json` | 30 KB — every number the console displays |
| `app/demo_data/{val,test}_demo.parquet` | 2,500 rows each, 7 columns |
| `app/demo_data/top_features.json` | 0.5 KB |
| **Total payload** | **~60 KB** |

Three reasons, and the third is the one that matters:

1. **Cold start.** No model load, no scoring, no SHAP. Verified locally: health
   endpoint responds in **< 0.1 s** after boot, index served in **0.05 s**.
2. **It keeps the runtime slim**, which is what makes §2 possible at all.
3. **It avoids the pickle fragility documented in
   [TIER_1_7_SCOPE](TIER_1_7_SCOPE.md)** — version skew on unpickling, and
   `__main__`-bound class resolution for the isotonic calibrator. A public
   deployment is the worst possible place to discover either.

`tests/test_console_runtime.py` asserts the payload stays under 512 KB and that
`app.py` contains no `pickle.load`, `joblib.load` or reference to
`outputs/models`. Both are structural assertions: they cannot be satisfied by
accident.

### The hazard this creates, and the check for it

A precomputed bundle is a **second copy of the results, and copies go stale.**
This one did: it was built before the Tier 2C.6 threshold reconciliation, so the
console was serving subgroup numbers measured at an operating threshold the
repository had since withdrawn. That is the same defect class as hand-typed
prose inside a generated report, relocated to the demo layer — and R4 does not
exempt a number because it is on a dashboard.

`src/pipelines/demo_bundle.py` fixes that by declaring, per bundle section,
which artifact it comes from and where inside it:

```
python src/pipelines/demo_bundle.py          # verify  (CI-shaped; fails when stale)
python src/pipelines/demo_bundle.py --fix    # rewrite stale sections from source
```

Two kinds of section are handled differently, deliberately:

- **Verbatim copies** (`fairness`, `verdict`) are checked by equality.
- **Projections** (`anchors`, `multivariate`, `label_interval`) are trimmed views
  — the console needs four fields out of a forty-field anchor record — so they
  are checked **field by field** against the value each was projected from.
  Comparing a trimmed view by equality would alarm forever, and a check that
  always alarms is a check that gets ignored.

Sections that are reshaped for the console's layout are **listed by name** in the
report as unmapped rather than silently skipped. Current state: **IN_SYNC**,
5 sections mapped (2 copies + 3 projections, 32 projected fields matched).

## 4. Theme and server settings

`.streamlit/config.toml` is read automatically. One defect was fixed during
deployment prep and is worth recording:

> `enableCORS = false` alongside `enableXsrfProtection = true` is an
> **incompatible pair**. Streamlit silently overrides CORS back to `true` and
> logs a warning on every start — which it had been doing. XSRF protection is
> the one worth keeping on a public deployment, so the CORS line was removed
> rather than the warning suppressed.

Colour discipline, since the console's whole purpose is to display severity:
severity colours are defined once in `app.py` as `SEVERITY_COLORS` and never used
decoratively. The chrome is deliberately desaturated so the only saturated colour
on screen carries meaning.

## 5. Defects found by deployment prep

Both would have failed on Community Cloud while working perfectly here.

| Defect | Effect | Found by |
|---|---|---|
| `st.info(..., icon="◈")` and three siblings | `◈` (U+25C8), `⚠` and `✕` are **not valid emoji** to Streamlit's validator. Every one raised `StreamlitAPIException` **at script execution**, so the console crashed on load | `test_console_imports_without_the_research_stack` — importing `app.py` executes the script, so the exception surfaced in CI rather than in a browser |
| `enableCORS` / `enableXsrfProtection` conflict | warning on every start; CORS setting silently ignored | the same test run |
| Non-ASCII `print()` in a module's `__main__` | On Windows a **piped** stdout defaults to cp1252, so `evaluator.py` printing `Δ` raised `UnicodeEncodeError` and the stage exited non-zero **after** writing its artifact correctly. Invisible in a console, invisible on Linux CI — visible only when a Windows parent captures the pipe | `reproduce.py`, the first time the full chain was actually run end to end |

The third is the same shape as the hardcoded paths: **environment-dependent, and
silent in the environment it was written in.** Fixed at the runner
(`PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`, plus explicit `encoding=` on the
subprocess) rather than module by module, so it cannot recur the next time
somebody prints an arrow.

The first is the notable one: **the app was broken and looked fine**, because
nothing in the repository had ever executed the Streamlit script outside an
interactive session. The test that catches it does not check that the app
imports — it checks that the app imports *with the research stack removed*, which
is the condition the deploy target actually imposes.

## 6. Verified before hand-off

| Check | Result |
|---|---|
| `pytest tests -q` | **93 passed** |
| `ruff check src tests` (blocking rules) | clean |
| Console imports with sklearn / lightgbm / shap / scipy blocked | pass |
| Guard itself blocks (negative control) | pass |
| Root `requirements.txt` is exactly the slim runtime | pass |
| `requirements-dev.txt` includes the runtime via `-r` | pass |
| Demo payload under budget | ~60 KB of 512 KB |
| No model artifact referenced by `app.py` | pass |
| Evidence bundle matches its source artifacts | **IN_SYNC** |
| `streamlit run app.py` boots headless | health `ok`, index HTTP 200 in 0.05 s |
| CI job reproducing the deploy environment | `console` job added |

## 7. What is NOT claimed

- **Not load-tested.** One user, one session, locally.
- **Cold start measured locally, not on Community Cloud.** The container there is
  slower and the first request includes a cold pip environment. The claim made is
  the payload size and the absence of model loading, both of which are checked;
  the wall-clock figure on their infrastructure is not something this repository
  can verify.
- **The console shows a demo subset**, not the full test window: 2,500 rows per
  split. Every aggregate number it displays comes from the full-cohort artifacts
  via the evidence bundle, not from the subset — the subset is for the
  distribution plots only.
