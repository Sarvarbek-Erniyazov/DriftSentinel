"""
DriftSentinel — monitoring console (Streamlit)

DESIGN PRINCIPLE
    Every drift verdict is rendered against the NO-DRIFT BASELINE, always, by
    construction. `signal_panel()` cannot draw a firing rate without also
    drawing the random-control floor beside it — the contrast is not a caption,
    it is the component. This repository's central finding is that the original
    detector fired 2.15/8 signals on data where drift was impossible; a console
    that displayed "4/6 fired" without that floor would reproduce the exact
    error the project exists to correct.

STREAMLIT CONSTRAINTS HONOURED
    * module-level imports kept light (no sklearn / lightgbm / shap at import)
    * artifacts precomputed into app/demo_data (73 KB total); no model is loaded
    * @st.cache_data for parsed evidence, @st.cache_resource for the frames
    * st.session_state for every control that must survive a rerun
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
DEMO = APP_DIR / "app" / "demo_data"
if not DEMO.exists():                      # support app/app.py layout too
    DEMO = APP_DIR / "demo_data"

st.set_page_config(page_title="DriftSentinel — Monitoring Console",
                   page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ── semantic colour, defined ONCE ─────────────────────────────────────────
SEVERITY_COLORS = {"STABLE": "#3FB950", "NONE": "#3FB950", "MILD": "#8FA6C0",
                   "MODERATE": "#D29922", "HIGH": "#DB6D28", "CRITICAL": "#F85149"}
INK, MUTED, GRID = "#D8DEE9", "#8B97A8", "#2A3140"
BASELINE_COLOR = "#6E7A8A"     # the no-drift floor — deliberately neutral
FIRED_COLOR = "#4C8DF6"

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .ds-kicker {font-family: monospace; font-size: 0.72rem; letter-spacing: .12em;
              text-transform: uppercase; color: #8B97A8; margin-bottom: .15rem;}
  .ds-card {background:#171B24; border:1px solid #2A3140; border-radius:4px;
            padding: .85rem 1rem; height: 100%;}
  .ds-big {font-family: monospace; font-size: 1.75rem; line-height: 1.1;}
  .ds-sub {font-family: monospace; font-size: .78rem; color:#8B97A8;}
  .ds-pill {display:inline-block; font-family:monospace; font-size:.7rem;
            padding:.12rem .5rem; border-radius:3px; border:1px solid #2A3140;}
  .ds-note {border-left:2px solid #2A3140; padding-left:.8rem; color:#8B97A8;
            font-size:.86rem;}
  div[data-testid="stMetricValue"] {font-family: monospace;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Data access
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_evidence() -> dict:
    return json.loads((DEMO / "evidence.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_frames() -> dict:
    return {"val": pd.read_parquet(DEMO / "val_demo.parquet"),
            "test": pd.read_parquet(DEMO / "test_demo.parquet")}


E = load_evidence()
F = load_frames()

if "window" not in st.session_state:
    st.session_state.window = "test (entry-cohort)"
if "budget" not in st.session_state:
    st.session_state.budget = "budget_0.20"


def sev_color(s: str) -> str:
    return SEVERITY_COLORS.get(str(s).upper(), MUTED)


def card(label: str, value: str, sub: str = "", color: str = INK) -> str:
    return (f'<div class="ds-card"><div class="ds-kicker">{label}</div>'
            f'<div class="ds-big" style="color:{color}">{value}</div>'
            f'<div class="ds-sub">{sub}</div></div>')


# ══════════════════════════════════════════════════════════════════════════
# THE component: a signal is never drawn without its no-drift floor
# ══════════════════════════════════════════════════════════════════════════

def signal_panel(fired: dict, baseline: dict, height: int = 260):
    """
    Render voting signals against the random-control floor.

    The baseline is a REQUIRED argument. There is deliberately no code path
    that renders a firing rate on its own.
    """
    import altair as alt
    rows = []
    for sig, hit in fired.items():
        rows.append({"signal": sig, "series": "this window",
                     "value": 1.0 if hit else 0.0})
        rows.append({"signal": sig, "series": "no-drift floor",
                     "value": float(baseline.get(sig, 0.0))})
    d = pd.DataFrame(rows)
    # A zero-valued bar draws nothing, which reads as a BROKEN CHART rather
    # than as correct silence. Every row therefore carries an explicit label,
    # and zeros are labelled "0.00 silent" at the axis origin, so "did not
    # fire" is visually distinguishable from "did not render".
    d["label"] = np.where(d["value"] > 0, d["value"].map("{:.2f}".format),
                          "0.00 silent")
    base_enc = alt.Chart(d).encode(
              y=alt.Y("signal:N", sort=list(fired), title=None,
                      axis=alt.Axis(labelFont="monospace", labelColor=INK,
                                    labelFontSize=11, domainColor=GRID,
                                    tickColor=GRID)),
              x=alt.X("value:Q", title="firing rate",
                      scale=alt.Scale(domain=[0, 1]),
                      axis=alt.Axis(labelColor=MUTED, titleColor=MUTED,
                                    gridColor=GRID, domainColor=GRID)),
              yOffset=alt.YOffset("series:N"),
              color=alt.Color("series:N",
                              scale=alt.Scale(domain=["this window", "no-drift floor"],
                                              range=[FIRED_COLOR, BASELINE_COLOR]),
                              legend=alt.Legend(orient="top", title=None,
                                                labelColor=INK, labelFontSize=11)),
              tooltip=["signal", "series", alt.Tooltip("value:Q", format=".2f")])

    bars = base_enc.mark_bar(cornerRadius=1)
    # zero-anchor tick: guarantees a visible mark on every row, including zeros
    ticks = base_enc.mark_tick(thickness=2, size=9, opacity=0.55)
    labels = base_enc.mark_text(align="left", dx=4, fontSize=10,
                                font="monospace").encode(
        text=alt.Text("label:N"),
        color=alt.condition(alt.datum.value > 0, alt.value(INK), alt.value(MUTED)))

    ch = ((ticks + bars + labels)
          .properties(height=height)
          .configure_view(strokeWidth=0)
          .configure(background="#0E1117"))
    st.altair_chart(ch, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="ds-kicker">DriftSentinel</div>', unsafe_allow_html=True)
    st.markdown("### Monitoring console")
    st.caption("30-day readmission · UCI Diabetes 130-US · entry-cohort split")

    st.session_state.window = st.selectbox(
        "Production window",
        ["test (entry-cohort)", "random control (no drift possible)",
         "temporal (chronological)"],
        index=["test (entry-cohort)", "random control (no drift possible)",
               "temporal (chronological)"].index(st.session_state.window)
        if st.session_state.window in
        ["test (entry-cohort)", "random control (no drift possible)",
         "temporal (chronological)"] else 0,
        help="Reference window is always `val`. The random control is the "
             "falsification baseline: drift is impossible there by construction.")

    st.divider()
    v = E["verdict"]
    st.markdown('<div class="ds-kicker">Tier 0 verdict</div>', unsafe_allow_html=True)
    st.markdown(f"**Ordering:** `{v['ordering_verdict']}`  \n"
                f"**Split:** `{v['split_validity_verdict']}`")
    st.caption("encounter_id chronology verified against 3 external clinical "
               "anchors; the split is still an entry-cohort split.")

    st.divider()
    st.markdown('<div class="ds-kicker">Limitations · 1 click</div>',
                unsafe_allow_html=True)
    st.markdown("Open the **Method** tab →")
    st.caption("Target discrepancy · selector withdrawn · no-selection wins · "
               "two luck-not-design defects · schema ≠ adversarial")


REG_KEY = {"test (entry-cohort)": "entry_cohort",
           "random control (no drift possible)": "random",
           "temporal (chronological)": "temporal"}
regime = REG_KEY[st.session_state.window]
baseline = E["regimes"]["random"]["firing_rates"]
reg = E["regimes"][regime]

# For the live entry-cohort window we have the actual per-signal booleans;
# for the other regimes we show the seed-averaged firing rate as the verdict.
if regime == "entry_cohort":
    fired = E["live"]["evidence"]
    n_fired, n_vote = E["live"]["n_evidence"], E["live"]["n_voting"]
    severity = E["live"]["severity"]
    diagnostics = E["live"]["diagnostics"]
else:
    fired = {k: v >= 0.5 for k, v in reg["firing_rates"].items()}
    n_fired, n_vote = sum(fired.values()), len(fired)
    severity = max(reg["severity_counts"], key=reg["severity_counts"].get)
    diagnostics = {k: v >= 0.5 for k, v in reg["diagnostics"].items()}


# ══════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════

t_mon, t_diag, t_pred, t_unc, t_method = st.tabs(
    ["  Monitor  ", "  Diagnose  ", "  Predict  ", "  Uncertainty  ", "  Method  "])


# ─────────────────────────────────── MONITOR ──────────────────────────────
with t_mon:
    st.markdown("#### Drift status")
    st.info(
        "**Switch the window to `random control` in the sidebar** to see what "
        "these same detectors report on data where drift is **impossible by "
        "construction**. That contrast is the point of this console — before "
        "correction the detector fired 2.15 of 8 signals there.", icon="🔎")
    base_mean = E["regimes"]["random"]["n_evidence_mean"]
    base_std = E["regimes"]["random"]["n_evidence_std"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Signals fired", f"{n_fired}/{n_vote}",
                     f"no-drift floor {base_mean:.2f}/{n_vote}",
                     FIRED_COLOR), unsafe_allow_html=True)
    c2.markdown(card("Severity", severity,
                     f"CRITICAL ≥ {E['live']['severity_boundaries']['critical_min']}"
                     f" of {n_vote}", sev_color(severity)), unsafe_allow_html=True)
    c3.markdown(card("Excess over floor", f"{n_fired - base_mean:+.2f}",
                     "signals above the no-drift baseline"),
                unsafe_allow_html=True)
    c4.markdown(card("Control status",
                     list(E["regimes"]["random"]["alert_status_counts"])[0],
                     "20/20 seeds, drift impossible",
                     SEVERITY_COLORS["STABLE"]), unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="ds-kicker">Voting signals vs the no-drift floor</div>',
                    unsafe_allow_html=True)
        signal_panel(fired, baseline)
        if regime != "entry_cohort":
            st.caption("Display convention: for seeded regimes a signal is shown "
                       "as fired if its firing rate is ≥ 0.5 across 20 seeds. "
                       "The grey floor is always the raw rate, never thresholded.")
        st.markdown(
            '<div class="ds-note">The grey bar is the firing rate on a random '
            'patient split, where drift is <b>impossible by construction</b>. '
            'Any blue bar not clearly above its grey counterpart is not evidence '
            'of drift. Before correction this detector fired <b>2.15 of 8</b> '
            'signals on that control; it now fires '
            f'<b>{base_mean:.2f} of {n_vote}</b>.</div>',
            unsafe_allow_html=True)

    with right:
        st.markdown('<div class="ds-kicker">Retired detectors — diagnostics, not evidence</div>',
                    unsafe_allow_html=True)
        for sig, hit in diagnostics.items():
            ctrl = E["regimes"]["random"]["diagnostics"].get(sig, 0.0)
            st.markdown(
                f'<div class="ds-card" style="margin-bottom:.5rem">'
                f'<span class="ds-pill" style="color:{MUTED}">NOT EVIDENCE</span> '
                f'<code>{sig}</code><br>'
                f'<span class="ds-sub">this window: {"fired" if hit else "silent"} · '
                f'fires on {ctrl:.0%} of no-drift seeds</span></div>',
                unsafe_allow_html=True)
        st.markdown(
            '<div class="ds-note">CUSUM fired in <b>100% of no-drift runs</b> '
            '(101–207 alarms each) and Page-Hinkley\'s verdict is decided by '
            'whether the first 200 rows sit above or below the stream mean. '
            'Neither responded to any synthetic shift at any magnitude, so both '
            'were removed from the evidence count — retained here because the '
            'evidence that condemned them must stay inspectable.</div>',
            unsafe_allow_html=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    ls = E["live"]["label_shift"]
    m1.metric("AUC degradation", f"{E['live']['auc_deg']:+.4f}",
              help="val → test, threshold-free")
    m2.metric("Label shift (relative)", f"{ls['relative_change']:+.1%}",
              help=f"p = {ls['p_value']:.2e} · rule: {ls['rule']}")
    m3.metric("F1 degradation", f"{E['live']['f1_deg']:+.4f}",
              help=f"threshold source: {E['live']['threshold_source']}")
    st.caption(f"Operating threshold sourced from: {E['live']['threshold_source']}")


# ─────────────────────────────────── DIAGNOSE ─────────────────────────────
with t_diag:
    st.markdown("#### What did the alert actually detect?")
    st.markdown(
        '<div class="ds-note">An alert is only actionable if you know what the '
        'firing signal responds to. Each signal below was benchmarked against '
        'three synthetic shifts with known ground truth (covariate / label / '
        'concept), swept over magnitude.</div>', unsafe_allow_html=True)
    st.markdown("")

    rows = []
    for sig, d in E["diagnosticity"].items():
        rows.append({
            "signal": sig,
            "verdict": d["verdict"],
            "responds to": ", ".join(s.replace("_shift", "") for s in d["responds_to"])
                           or "— nothing —",
            "covariate": d["max_magnitude_firing_rate"]["covariate_shift"],
            "label": d["max_magnitude_firing_rate"]["label_shift"],
            "concept": d["max_magnitude_firing_rate"]["concept_shift"],
            "fired here": "✓" if fired.get(sig) else "",
        })
    df = pd.DataFrame(rows)

    def _style(v):
        return (f"color:{SEVERITY_COLORS['STABLE']}" if v == "DIAGNOSTIC"
                else f"color:{SEVERITY_COLORS['MODERATE']}" if v == "PARTIAL"
                else f"color:{SEVERITY_COLORS['CRITICAL']}")

    st.dataframe(
        df.style.map(_style, subset=["verdict"])
          .format({"covariate": "{:.2f}", "label": "{:.2f}", "concept": "{:.2f}"}),
        use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="ds-note"><b>Why <code>label_drift</code> shows PARTIAL.</b> '
        'The covariate-shift control resamples rows, which perturbs prevalence '
        'as a side effect — so <code>label_drift</code> responding to it is an '
        'artefact of how the synthetic control was constructed, not a defect in '
        'the signal. Read its label-shift column (1.00) as the real result.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="ds-note"><b>No signal is uniquely diagnostic of concept '
        'shift</b> — the mechanism this module is named for. <code>f1_drop</code> '
        'is diagnostic of covariate shift and <code>label_drift</code> of label '
        'shift; everything else responds to more than one mechanism and cannot '
        'localise a cause on its own.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="ds-kicker">Multivariate detectors — calibrated on the control</div>',
                unsafe_allow_html=True)
    mv = pd.DataFrame([
        {"regime": k.replace("_NEGATIVE_CONTROL", " (no drift)"),
         "classifier-2ST AUC": v["c2st_auc"], "c2ST p": v["c2st_p"],
         "MMD p": v["mmd_p"], "BBSD (FDR)": "fires" if v["bbsd"] else "silent"}
        for k, v in E["multivariate"].items()])
    st.dataframe(mv, use_container_width=True, hide_index=True)
    st.caption("Every shipped detector is univariate and marginal, so a change "
               "in the dependence structure between features is invisible to all "
               "of them simultaneously. These three see the joint distribution — "
               "and all three stay silent on the no-drift control.")


# ─────────────────────────────────── PREDICT ──────────────────────────────
with t_pred:
    st.markdown("#### Single-patient risk, at an honest operating point")
    tp = E["threshold_policy"]
    budgets = {k: v for k, v in tp["budgets"].items() if v.get("feasible")}

    cA, cB = st.columns([2, 3], gap="large")
    with cA:
        st.session_state.budget = st.radio(
            "Alert budget (max share of patients flagged)",
            list(budgets), index=list(budgets).index(st.session_state.budget)
            if st.session_state.budget in budgets else 1,
            format_func=lambda k: f"PPR ≤ {float(k.split('_')[1]):.0%}")
        b = budgets[st.session_state.budget]
        st.markdown(card("Threshold", f"{b['threshold']:.4f}",
                         f"PPR {b['predicted_positive_rate']:.1%} · "
                         f"{b['alerts_per_1000_patients']} alerts / 1000"),
                    unsafe_allow_html=True)
        st.markdown("")
        q1, q2, q3 = st.columns(3)
        q1.metric("Precision", f"{b['precision']:.3f}")
        q2.metric("Recall", f"{b['recall']:.3f}")
        q3.metric("Missed", f"{b['missed_positives']}")
        cap = float(st.session_state.budget.split("_")[1])
        if b["predicted_positive_rate"] < cap - 1e-6:
            st.caption(
                f"**This budget does not bind.** The optimum already sits at "
                f"PPR {b['predicted_positive_rate']:.1%}, below the {cap:.0%} cap, "
                f"so raising the cap changes nothing — itself an informative "
                f"result about the operating characteristics.")

    with cB:
        df_t = F["test"]
        i = st.slider("Demo patient (test window)", 0, len(df_t) - 1, 7,
                      help="Precomputed calibrated probability; no model runs in the app.")
        row = df_t.iloc[i]
        p_i = float(row["p"])
        flagged = p_i >= b["threshold"]
        st.markdown(card(
            "Predicted 30-day readmission risk", f"{p_i:.3f}",
            f"{'FLAGGED — above threshold' if flagged else 'not flagged'} · "
            f"actual outcome: {'readmitted <30d' if row['y'] else 'not readmitted'}",
            SEVERITY_COLORS["HIGH"] if flagged else MUTED), unsafe_allow_html=True)
        st.markdown("")
        # `age` is stored ordinal-encoded (0-9). Showing the code in a
        # clinician-facing table is meaningless; map it back to the band.
        AGE_BANDS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        show = [c for c in df_t.columns if c not in ("y", "p")]
        vals = []
        for c in show:
            v = row[c]
            if c == "age":
                i = int(round(float(v)))
                v = AGE_BANDS[i] if 0 <= i < len(AGE_BANDS) else f"code {v}"
            vals.append(v)
        st.dataframe(pd.DataFrame({"feature": show, "value": vals}),
                     use_container_width=True, hide_index=True, height=210)

    st.divider()
    legacy = tp["legacy"]
    if legacy["predicted_positive_rate"] > 0.60:
        st.error(
            f"**Degenerate operating point rejected.** The shipped "
            f"\"cost-optimal\" threshold {legacy['threshold']:.4f} flags "
            f"**{legacy['predicted_positive_rate']:.1%}** of all patients — an "
            f"alert on almost every admission carries no triage information and "
            f"would be switched off within a week. Its per-class cost "
            f"normalisation inflated the requested 5:1 ratio to an effective "
            f"**{legacy['effective_cost_ratio']:.1f}:1**. Any threshold above "
            f"60% PPR is flagged automatically.", icon="⚠️")
    st.caption(f"Feasible FN:FP cost ratios on this model: "
               f"{[r['cost_ratio'] for r in tp['sweep'] if not r['degenerate']]} — "
               f"every higher ratio produces a degenerate rule. Decision-curve "
               f"analysis shows net benefit over treat-all/treat-none across "
               f"threshold probabilities {tp['dca_range']}.")


# ────────────────────────────────── UNCERTAINTY ───────────────────────────
with t_unc:
    st.markdown("#### Conformal prediction — reported, not consumed")
    dec = E["conformal"]["decontaminated"]
    aud, tst = dec["audit_half_HELD_OUT"], dec["test_held_out"]

    u1, u2, u3, u4 = st.columns(4)
    u1.markdown(card("Coverage · HELD-OUT", f"{aud['coverage']:.4f}",
                     "audit half, target 0.90", SEVERITY_COLORS["STABLE"]),
                unsafe_allow_html=True)
    u2.markdown(card("Coverage · test", f"{tst['coverage']:.4f}",
                     "held out, target 0.90", SEVERITY_COLORS["STABLE"]),
                unsafe_allow_html=True)
    u3.markdown(card("Mean set size", f"{tst['mean_set_size']:.3f}",
                     f"both labels: {tst['share_both_labels']:.1%}"),
                unsafe_allow_html=True)
    u4.markdown(card("In-sample (invalid)",
                     f"{E['conformal']['contaminated_val']['coverage']:.4f}",
                     "calibration-set coverage — guaranteed by construction",
                     MUTED), unsafe_allow_html=True)

    st.markdown("")
    st.warning(
        "**Uncertainty does not gate the decision here, and that is a finding.** "
        "A trivial distance-to-threshold rule beats the conformal gate at every "
        "matched review budget. At 11.2% prevalence the model is confidently "
        "negative for almost everyone, so sets are singletons; pushing coverage "
        "high enough to produce a usable review rate routes **93% of at-risk "
        "patients** to human review — which is not triage. Conformal is retained "
        "as a diagnostic, not promoted to a decision gate it has not earned.",
        icon="🔎")

    comp = pd.DataFrame(E["conformal"]["triage_comparisons"])[
        ["alpha", "review_rate", "conformal_auto_f1", "baseline_auto_f1",
         "f1_conformal_minus_baseline", "at_risk_sent_to_review"]]
    comp.columns = ["α", "review rate", "conformal F1", "baseline F1",
                    "Δ F1", "at-risk → review"]
    st.dataframe(comp.style.format({
        "review rate": "{:.3f}", "conformal F1": "{:.4f}", "baseline F1": "{:.4f}",
        "Δ F1": "{:+.4f}", "at-risk → review": "{:.3f}"}),
        use_container_width=True, hide_index=True)
    st.caption("Every conformal operating point is degenerate: it makes zero "
               "positive decisions and auto-labels the remainder negative. Low "
               "error there is an artifact of not deciding, so F1 — which cannot "
               "be gamed by abstention — is the comparison metric.")


# ─────────────────────────────────── METHOD ───────────────────────────────
with t_method:
    st.markdown("#### The scientific story")
    st.markdown(
        '<div class="ds-note">This project began by claiming a temporal split '
        'and attributing an observed label shift to concept drift. Neither claim '
        'had been tested. What follows is what testing them produced.</div>',
        unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("##### 1 · Is `encounter_id` chronological?")
        st.markdown("**Verified — SUPPORTED.** Three external clinical/regulatory "
                    "anchors, each localising a known date to a rank position:")
        anc = pd.DataFrame([
            {"anchor": k.replace("_", " "),
             "event date": v["event_date"] or "trend anchor, no single event date",
             "p": f"{v['p_raw']:.2e}", "role": v["role"], "outcome": v["outcome"]}
            for k, v in E["anchors"].items()])
        st.dataframe(anc, use_container_width=True, hide_index=True)
        li = E["label_interval"]
        st.markdown(
            f"The anchor-derived calendar map reproduces the dataset's own "
            f"30-day boundary: median implied gap **"
            f"{li['implied_calendar_check']['<30']['median_implied_days']:.1f} days** "
            f"for `<30` vs **"
            f"{li['implied_calendar_check']['>30']['median_implied_days']:.1f} days** "
            f"for `>30` — a {li['ratio_gt30_over_lt30']:.1f}× separation from "
            f"anchors that never saw the label.")

    with s2:
        st.markdown("##### 2 · So is the split temporal?")
        st.error("**No — it is an entry-cohort split.** Sorting *patients* by "
                 "their *first* encounter puts an early entrant's later "
                 "encounters in train. True even under verified chronology.",
                 icon="🚫")
        st.markdown("##### 3 · Is the label shift concept drift?")
        le = E["censoring"]["last_enc"]
        lo = E["censoring"]["label_obs"]["by_label"]
        st.error(
            f"**No — it tracks observability.** `readmitted` is essentially an "
            f"in-extract successor indicator: only "
            f"**{lo['NO']['share_with_in_dataset_successor']:.1%}** of `NO` rows "
            f"have any later encounter. The final-observed-encounter share rises "
            f"{le['share_is_last_decile'][0]['rate']:.3f} → "
            f"{le['share_is_last_decile'][-1]['rate']:.3f} across the window, and "
            f"the `<30` gradient **reverses sign** "
            f"({le['lt30_slope_all_rows']:+.4f} → "
            f"{le['lt30_slope_non_last_rows_only']:+.4f}) once final encounters "
            f"are excluded.", icon="🚫")

    st.divider()
    st.markdown("##### The design principle: every method carries a condition under which it MUST fire")
    st.markdown(
        '<div class="ds-note">Three of the four modernisation phases returned '
        '<b>nulls</b> on real data. Each null is interpretable only because the '
        'instrument was proven able to detect the thing it found absent. '
        '"We found nothing" and "there is nothing to find" are different claims, '
        'and most work never earns the second.</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"phase": "2B.1 Adaptive conformal", "falsification arm": "synthetic hard label shift",
         "result": "static collapses 0.90 → 0.41, ACI holds 0.898 ✓"},
        {"phase": "2B.2 Multivariate drift", "falsification arm": "random split (drift impossible)",
         "result": "all three detectors silent ✓"},
        {"phase": "2B.3 Triage gating", "falsification arm": "10% label-corrupted subgroup",
         "result": "neither policy detects it — uncertainty ≠ wrongness ✓"},
        {"phase": "2B.4 Robustness", "falsification arm": "top feature → pure noise",
         "result": "−0.046 AUC, harness detects real damage ✓"},
        {"phase": "2C.1 Fairness", "falsification arm": "injected subgroup disparity",
         "result": "gap 0.131 flagged; real gender gap 0.009 not ✓"},
    ]), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Limitations & corrections — the things a reviewer should find first")
    L, R = st.columns(2, gap="large")
    with L:
        with st.expander("**Target discrepancy** — the README described a target the code never computed", True):
            st.markdown("The README stated *\"readmission within 30 days\"* while "
                        "the code computed the **merged** `<30 or >30` target "
                        "(46.1% prevalence). Every published number sat under a "
                        "label that did not match its computation. Now `<30`, "
                        "11.2% prevalence.")
        with st.expander("**The 7-stage selector claim is withdrawn**"):
            ab = E["ablation"]
            st.markdown(
                f"Stages 1–2 (variance + correlation) produce **byte-identical "
                f"output to the full pipeline in all 10 folds** — AUC difference "
                f"exactly `0.00000`, CI [0, 0]. It is a **two-stage filter with "
                f"five decorative stages**. A 4× prevalence change left the 53 "
                f"selected features identical.")
        with st.expander("**HEADLINE: no selection at all beats the shipped selector**"):
            c = E["ablation"]["contrasts"]
            st.markdown(
                f"Using **all 78 features** beats the 7-stage pipeline by "
                f"**{c['a_all_features_minus_shipped']['delta_auc_mean']:+.4f} AUC** "
                f"(CI {c['a_all_features_minus_shipped']['ci95']}), and an in-fold "
                f"target-aware selector by "
                f"**{c['d_target_aware_in_fold_minus_shipped']['delta_auc_mean']:+.4f}**. "
                f"The pipeline **costs accuracy** while claiming sophistication.")
    with R:
        with st.expander("**Two defects that worked by luck, not design**", True):
            st.markdown(
                "**1 · `TARGET_COLS` hash randomisation.** A `set` iterated to "
                "build column order, and Python randomises string hashing per "
                "process — so output column order changed every run. No seed "
                "could catch it; it was in serialisation, not modelling. The "
                "adversarial modules index positionally, so an artifact from one "
                "run applied to another's data would read the wrong column, "
                "silently.\n\n"
                "**2 · Boruta empty-candidate crash.** Selector stages 5–6 "
                "consume Boruta's confirmed set. Boruta confirms **exactly one** "
                "feature on full data and **zero** in some folds, where the fit "
                "raises. The pipeline was one feature from a hard failure.")
        with st.expander("**Schema violation ≠ adversarial detection**"):
            st.markdown(
                f"The defense system's apparent power came from a **schema check**. "
                f"Its violation count separated attacked from clean at AUC 0.94 — "
                f"but only because the attack added continuous noise to **binary** "
                f"columns. That is a data-type violation, not adversarial "
                f"detection. After repair it separates at **AUC 0.500**; the full "
                f"system reaches 0.617, detecting **0.064 at a 5% FPR**.\n\n"
                f"Relatedly, **{E['robustness']['zero_grad']:.1%} of "
                f"finite-difference gradients are exactly zero** on this "
                f"piecewise-constant ensemble — the shipped FGSM/PGD attacks "
                f"never executed. ASR ≈ 0 was the null behaviour of a broken "
                f"method, and that 'robustness score' is withdrawn.")
        with st.expander("**Fairness: a supported disparity by age**"):
            ages = E["fairness"]["age_band"]["levels"]
            rows = [{"band": k, "n": v["n"],
                     "prevalence": v.get("prevalence", {}).get("point"),
                     "AUC": v.get("auc", {}).get("point"),
                     "recall": v.get("recall", {}).get("point")}
                    for k, v in ages.items() if v["status"] == "OK"]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(
                "Patients **70+** carry the **highest** risk (9.6%) but receive "
                "the **worst** discrimination (AUC 0.626) and are flagged at less "
                "than half the rate of under-40s. Intervals do not overlap — this "
                "disparity is supported. Race disparities in AUC are **not** "
                "supported, but the cohort is **underpowered to rule them out**; "
                "three race levels are too small for any claim.")

    st.divider()
    st.caption(
        "Single dataset · no external validation · retrospective, no prospective "
        "outcomes · AUC ≈ 0.64 is within the published 0.63–0.70 band for 30-day "
        "readmission on this dataset and reflects an intrinsically hard task. "
        "The contribution is the negative-control methodology, not the detectors."
    )
