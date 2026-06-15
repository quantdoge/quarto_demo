#!/usr/bin/env python3
"""SLICE — Statistical Learning and Inference Compute Engine (Streamlit edition).

A Streamlit re-implementation of the Quarto SLICE dashboard, styled to look
like the Quarto static site. It reuses the *exact same* data-loading logic as
``generate_dashboard.py`` (via :func:`generate_dashboard.build_bundle`), so the
two front-ends stay in lock-step.

Run it with::

    streamlit run streamlit_app.py
    # then pick the project base directory in the sidebar
    # (defaults to ./sample_project — create it with make_sample_data.py)
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from generate_dashboard import build_bundle

PRIMARY = "#2c6fbb"
ACCENT = "#e07b39"
DEFAULT_BASE = "sample_project"


# --------------------------------------------------------------------------- #
# Page setup + theming (mirrors slice_theme.css from the Quarto build)
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="SLICE · Statistical Learning and Inference Compute Engine",
    page_icon="📊",
    layout="wide",
)

CSS = """
<style>
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }

.slice-header {
  background: linear-gradient(90deg, #1f4f87, #2c6fbb); color: #fff;
  padding: .75rem 1.2rem; border-radius: 10px; margin-bottom: 1rem;
  display: flex; align-items: baseline; justify-content: space-between;
  box-shadow: 0 2px 6px rgba(0,0,0,.15);
}
.slice-logo { font-size: 1.6rem; font-weight: 800; letter-spacing: 2px; }
.slice-sub { margin-left: .8rem; font-size: .95rem; opacity: .9; }
.slice-genstamp { font-size: .72rem; opacity: .75; }

.kpi { border-radius: 10px; padding: .9rem 1rem; color: #fff; box-shadow: 0 2px 5px rgba(0,0,0,.08); }
.kpi-label { font-size: .82rem; opacity: .92; }
.kpi-val { font-size: 2rem; font-weight: 700; margin-top: .2rem; line-height: 1.1; }
.kpi-a { background: linear-gradient(135deg, #2c6fbb, #4b8fd6); }
.kpi-b { background: linear-gradient(135deg, #6f54c9, #9173e0); }
.kpi-c { background: linear-gradient(135deg, #2f3b47, #51606e); }
.kpi-d { background: linear-gradient(135deg, #138a72, #1cb494); }

.slice-cardh { font-weight: 700; font-size: .95rem; margin: 0 0 .5rem; color: #2b3138; }

.slice-modelhead { margin-bottom: .6rem; }
.slice-pill { display: inline-block; background: #2c6fbb; color: #fff; border-radius: 999px;
  padding: .15rem .7rem; font-size: .76rem; font-weight: 600; margin-right: .35rem; }
.slice-pill-alt { background: #e07b39; }
.slice-artifact { color: #777; font-size: .76rem; margin-top: .4rem; }

.slice-table { width: 100%; border-collapse: collapse; font-size: .84rem; }
.slice-table th { text-align: left; background: #f1f3f5; padding: .4rem .55rem; border-bottom: 2px solid #dee2e6; }
.slice-table td { padding: .35rem .55rem; border-bottom: 1px solid #eee; vertical-align: top; }
.slice-table tbody tr:hover { background: #f8fbff; }
.slice-table code { background: #f1f3f5; padding: 0 .3rem; border-radius: 4px; font-size: .82em; }

.slice-tag { display: inline-block; font-size: .72rem; padding: .05rem .45rem; border-radius: 4px; background: #e9ecef; color: #495057; }
.slice-tag-categorical { background: #e3f0ff; color: #1c5fa8; }
.slice-tag-numerical { background: #fdeede; color: #b5651d; }
.slice-chip { background: #eef2f7; padding: 0 .3rem; border-radius: 4px; margin: 0 .15rem .15rem 0;
  display: inline-block; font-size: .8rem; }

.slice-summary { font-size: .92rem; line-height: 1.5; }
.slice-confline { margin-top: .5rem; font-size: .82rem; color: #555; }
.slice-conf { text-transform: uppercase; font-weight: 700; font-size: .72rem; padding: .05rem .45rem; border-radius: 4px; }
.slice-conf-high { background: #d8f5dd; color: #1b7a32; }
.slice-conf-medium { background: #fff3cd; color: #8a6d00; }
.slice-conf-low { background: #f8d7da; color: #a12330; }
.slice-kv td:first-child { color: #555; text-transform: capitalize; width: 45%; }
.slice-rationale { font-size: .85rem; color: #444; margin-top: .6rem; }

.slice-prio { font-size: .72rem; font-weight: 700; padding: .05rem .45rem; border-radius: 4px; }
.slice-prio-high { background: #f8d7da; color: #a12330; }
.slice-prio-medium { background: #fff3cd; color: #8a6d00; }
.slice-prio-low { background: #d8f5dd; color: #1b7a32; }

/* tighten the bordered "cards" so they read like Quarto cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: #fff; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,.05);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loading (same parser as the Quarto generator)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading model artifacts…")
def load_bundle(base_dir: str, top_n: int, max_inst: int) -> dict:
    return build_bundle(Path(base_dir), top_n, max_inst)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt3(x) -> str:
    return "—" if x is None else f"{float(x):.3f}"


def pct1(x) -> str:
    return "—" if x is None else f"{float(x) * 100:.1f}%"


# --------------------------------------------------------------------------- #
# HTML builders (reuse the Quarto CSS classes verbatim)
# --------------------------------------------------------------------------- #

def feature_table_html(desc: dict) -> str:
    rows = []
    for f in desc.get("categorical_predictors", []):
        rows.append(("categorical", f))
    for f in desc.get("numerical_predictors", []):
        rows.append(("numerical", f))
    if not rows:
        return "<em>No feature metadata available.</em>"
    body = "".join(
        f"<tr><td><code>{esc(f['name'])}</code></td>"
        f"<td><span class='slice-tag slice-tag-{t}'>{t}</span></td>"
        f"<td>{esc(f.get('description') or '—')}</td></tr>"
        for t, f in rows
    )
    return ("<table class='slice-table'><thead><tr><th>Feature</th><th>Type</th>"
            f"<th>Description</th></tr></thead><tbody>{body}</tbody></table>")


def hp_table_html(desc: dict) -> str:
    hp = desc.get("hyperparameters", [])
    if not hp:
        return "<em>No hyperparameter metadata available.</em>"
    body = "".join(
        f"<tr><td><code>{esc(p['name'])}</code></td>"
        f"<td><b>{esc('—' if p.get('value') is None else p['value'])}</b></td>"
        f"<td>{esc(p.get('interpretation') or '—')}</td></tr>"
        for p in hp
    )
    return ("<table class='slice-table'><thead><tr><th>Parameter</th><th>Value</th>"
            f"<th>Interpretation</th></tr></thead><tbody>{body}</tbody></table>")


def model_head_html(m: dict) -> str:
    return (
        "<div class='slice-modelhead'>"
        f"<span class='slice-pill'>{esc(m['horizon_label'])}</span>"
        f"<span class='slice-pill slice-pill-alt'>{esc(m['scope_label'])}</span>"
        f"<div class='slice-artifact'>artifact: <code>{esc(m['pkl'])}</code></div>"
        "</div>"
    )


def ai_summary_html(ai: dict) -> str:
    me = ai.get("model_explanation", {}) or {}
    conf = ""
    if ai.get("confidence"):
        c = esc(ai["confidence"])
        conf = (f"<div class='slice-confline'>confidence: "
                f"<span class='slice-conf slice-conf-{c}'>{c}</span></div>")
    kv = "".join(
        f"<tr><td>{esc(k.replace('_', ' '))}</td><td>{esc(v)}</td></tr>"
        for k, v in me.items() if k != "rationale"
    )
    out = (f"<div class='slice-summary'>{esc(ai.get('overall_summary') or 'No AI summary available.')}"
           f"{conf}</div>")
    if kv:
        out += ("<table class='slice-table slice-kv'><thead><tr><th>Aspect</th>"
                f"<th>Assessment</th></tr></thead><tbody>{kv}</tbody></table>")
    if me.get("rationale"):
        out += f"<p class='slice-rationale'><b>Rationale.</b> {esc(me['rationale'])}</p>"
    return out


def ai_shap_html(ai: dict) -> str:
    si = ai.get("shap_interpretation", {}) or {}
    tf = si.get("top_features", []) or []
    if not tf:
        out = "<em>No SHAP interpretation available.</em>"
    else:
        body = "".join(
            f"<tr><td>{esc('—' if f.get('global_importance_rank') is None else f['global_importance_rank'])}</td>"
            f"<td><code>{esc(f.get('feature'))}</code></td>"
            f"<td>{esc(f.get('effect_direction') or '—')}</td>"
            f"<td><span class='slice-tag'>{esc(f.get('effect_strength') or '—')}</span></td>"
            f"<td>{esc(f.get('notes') or '—')}</td></tr>"
            for f in tf
        )
        out = ("<table class='slice-table'><thead><tr><th>#</th><th>Feature</th>"
               "<th>Direction</th><th>Strength</th><th>Notes</th></tr></thead>"
               f"<tbody>{body}</tbody></table>")
    if si.get("overall_narrative"):
        out += f"<p class='slice-rationale'>{esc(si['overall_narrative'])}</p>"
    return out


def ai_audit_html(ai: dict) -> str:
    areas = (ai.get("audit_suggestion", {}) or {}).get("areas", []) or []
    if not areas:
        return "<em>No audit suggestions available.</em>"
    body = ""
    for a in areas:
        chips = " ".join(
            f"<code class='slice-chip'>{esc(x)}</code>"
            for x in (a.get("linked_features_or_findings") or [])
        )
        prio = esc((a.get("priority") or "").lower())
        body += (
            f"<tr><td><b>{esc(a.get('business_area') or '—')}</b></td>"
            f"<td><span class='slice-prio slice-prio-{prio}'>{esc(a.get('priority') or '—')}</span></td>"
            f"<td>{chips}</td>"
            f"<td>{esc(a.get('rationale') or '—')}</td>"
            f"<td>{esc(a.get('regulatory_relevance') or '—')}</td></tr>"
        )
    return ("<table class='slice-table'><thead><tr><th>Business area</th><th>Priority</th>"
            "<th>Linked features / findings</th><th>Rationale</th>"
            f"<th>Regulatory relevance</th></tr></thead><tbody>{body}</tbody></table>")


# --------------------------------------------------------------------------- #
# Plotly figures (server-side equivalents of the JS charts)
# --------------------------------------------------------------------------- #

def _base_layout(height: int = 320) -> dict:
    return dict(
        height=height, margin=dict(l=55, r=20, t=10, b=45),
        font=dict(size=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
    )


def fig_pr(m: dict, view: str) -> go.Figure:
    d = m.get("pr_data", [])
    fig = go.Figure()
    if view == "PR curve":
        fig.add_trace(go.Scatter(
            x=[r["recall"] for r in d], y=[r["precision"] for r in d], mode="lines",
            line=dict(color=PRIMARY, width=3),
            hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>"))
        base = m["metrics"].get("test_positive_rate")
        if base is not None:
            fig.add_hline(y=base, line=dict(color="#bbb", dash="dash"))
        fig.update_xaxes(title="Recall", range=[0, 1], gridcolor="#eee")
        fig.update_yaxes(title="Precision", range=[0, 1], gridcolor="#eee")
        fig.update_layout(showlegend=False, **_base_layout())
    else:
        t = [r for r in d if r.get("threshold") is not None]
        fig.add_trace(go.Scatter(x=[r["threshold"] for r in t], y=[r["precision"] for r in t],
                                 mode="lines", name="precision", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[r["threshold"] for r in t], y=[r["recall"] for r in t],
                                 mode="lines", name="recall", line=dict(color=ACCENT, width=3)))
        fig.update_xaxes(title="Threshold", range=[0, 1], gridcolor="#eee")
        fig.update_yaxes(title="Score", range=[0, 1], gridcolor="#eee")
        fig.update_layout(**_base_layout())
    return fig


def fig_imp(m: dict, labels: bool) -> go.Figure:
    d = list(reversed(m.get("shap_importance", [])[:15]))
    vals = [r["mean_abs_shap_value"] for r in d]
    fig = go.Figure(go.Bar(
        orientation="h", x=vals, y=[r["feature_name"] for r in d], marker_color=PRIMARY,
        text=[f"{v:.3f}" for v in vals] if labels else None,
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>mean|SHAP| %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="mean |SHAP value|", gridcolor="#eee")
    fig.update_layout(**{**_base_layout(), "margin": dict(l=190, r=45, t=10, b=40)})
    return fig


def fig_bee(m: dict, colour: bool) -> go.Figure:
    raw = m.get("shap_values", [])
    # only chart features that actually have points loaded (the top-N selection)
    present = {r["feature_name"] for r in raw}
    order = [r["feature_name"] for r in m.get("shap_importance", []) if r["feature_name"] in present]
    idx = {f: i for i, f in enumerate(order)}
    rng = {}
    for r in raw:
        if r.get("feature_value") is None or r["feature_name"] not in idx:
            continue
        c = rng.setdefault(r["feature_name"], [np.inf, -np.inf])
        c[0] = min(c[0], r["feature_value"]); c[1] = max(c[1], r["feature_value"])

    rs = np.random.default_rng(7)
    xs, ys, cs, texts = [], [], [], []
    for r in raw:
        f = r["feature_name"]
        if f not in idx:
            continue
        base = len(order) - 1 - idx[f]
        ys.append(base + (rs.random() - 0.5) * 0.7)
        xs.append(r["shap_value"])
        norm = 0.5
        c = rng.get(f)
        if c and c[1] > c[0] and r.get("feature_value") is not None:
            norm = (r["feature_value"] - c[0]) / (c[1] - c[0])
        cs.append(norm)
        texts.append(f)

    marker = dict(size=5, opacity=0.6)
    if colour:
        marker.update(color=cs, colorscale="RdBu", reversescale=True, cmin=0, cmax=1,
                      colorbar=dict(title="feature<br>value", thickness=12,
                                    tickvals=[0, 1], ticktext=["low", "high"]))
    else:
        marker.update(color=PRIMARY)

    fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="markers", marker=marker, text=texts,
                                 hovertemplate="%{text}<br>SHAP %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#999"))
    fig.update_xaxes(title="SHAP value (impact on model output)", gridcolor="#eee")
    fig.update_yaxes(tickvals=[len(order) - 1 - i for i in range(len(order))], ticktext=order)
    fig.update_layout(showlegend=False,
                      **{**_base_layout(520), "margin": dict(l=190, r=20, t=10, b=45)})
    return fig


# --------------------------------------------------------------------------- #
# Sidebar — data source + model selection
# --------------------------------------------------------------------------- #

st.sidebar.markdown("## Model selection")
base_dir = st.sidebar.text_input("Project base directory", value=DEFAULT_BASE,
                                  help="Folder containing Scripts/, Reports/, Data Dictionary/")
n_features = st.sidebar.slider(
    "Beeswarm features", min_value=5, max_value=30, value=12,
    help="How many top features (by mean |SHAP|) to display in the beeswarm.")

if not Path(base_dir).exists():
    st.markdown(
        "<div class='slice-header'><div><span class='slice-logo'>SLICE</span>"
        "<span class='slice-sub'>Statistical Learning and Inference Compute Engine</span></div></div>",
        unsafe_allow_html=True)
    st.warning(
        f"Base directory `{base_dir}` not found. Generate sample data first:\n\n"
        "```\npython make_sample_data.py --base-dir sample_project\n```")
    st.stop()

bundle = load_bundle(base_dir, top_n=n_features, max_inst=300)
app = bundle["app"]
horizons = bundle["options"]["horizons"]
scopes = bundle["options"]["scopes"]

h_label = st.sidebar.selectbox("Response variable", [h["label"] for h in horizons])
s_label = st.sidebar.selectbox("Product scope", [s["label"] for s in scopes])
h_key = next(h["key"] for h in horizons if h["label"] == h_label)
s_key = next(s["key"] for s in scopes if s["label"] == s_label)
m = bundle["models"].get(f"{h_key}|{s_key}")

if m is None:
    st.error("Selected model is not available in the data bundle.")
    st.stop()

d = m["description"]
st.sidebar.markdown(
    f"""<div class='slice-artifact' style='color:#444;font-size:.8rem;line-height:1.5'>
    <b>Artifact</b><br><code>{esc(m['pkl'])}</code>
    <div style='margin-top:.6rem'><b>Temporal split</b><br>
    datetime <code>{esc(d.get('datetime_field') or '—')}</code><br>
    split <code>{esc(d.get('train_test_split_date') or '—')}</code><br>
    cut-off <code>{esc(d.get('cutoff_date') or '—')}</code></div></div>""",
    unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption(f"**Objective.** {app['objective']}")


# --------------------------------------------------------------------------- #
# Header + KPI row
# --------------------------------------------------------------------------- #

st.markdown(
    f"""<div class='slice-header'>
      <div><span class='slice-logo'>SLICE</span>
      <span class='slice-sub'>{esc(app['full_name'])}</span></div>
      <div class='slice-genstamp'>generated {esc(bundle.get('generated_at',''))}</div>
    </div>""",
    unsafe_allow_html=True)

metrics = m["metrics"]
kpis = [
    ("kpi-a", "PR-AUC · Train (CV)", fmt3(metrics.get("cv_pr_auc"))),
    ("kpi-b", "PR-AUC · Test", fmt3(metrics.get("test_pr_auc"))),
    ("kpi-c", "Base Rate (test positives)", pct1(metrics.get("test_positive_rate"))),
    ("kpi-d", "Training cut-off", d.get("cutoff_date") or "—"),
]
for col, (cls, label, val) in zip(st.columns(4), kpis):
    col.markdown(
        f"<div class='kpi {cls}'><div class='kpi-label'>{esc(label)}</div>"
        f"<div class='kpi-val'>{esc(val)}</div></div>", unsafe_allow_html=True)

st.write("")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_perf, tab_shap, tab_ai = st.tabs(
    ["① Performance & Model", "② SHAP Beeswarm", "③ AI Summary"])

with tab_perf:
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("<div class='slice-cardh'>Model &amp; Features</div>", unsafe_allow_html=True)
        st.markdown(model_head_html(m), unsafe_allow_html=True)
        st.markdown(feature_table_html(d), unsafe_allow_html=True)
    with c2.container(border=True):
        st.markdown("<div class='slice-cardh'>Hyperparameters</div>", unsafe_allow_html=True)
        st.markdown(hp_table_html(d), unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3.container(border=True):
        st.markdown("<div class='slice-cardh'>Precision–Recall</div>", unsafe_allow_html=True)
        pr_view = st.radio("view", ["PR curve", "vs threshold"], horizontal=True,
                           label_visibility="collapsed", key="pr_view")
        st.plotly_chart(fig_pr(m, pr_view), width="stretch",
                        config={"displayModeBar": False})
    with c4.container(border=True):
        st.markdown("<div class='slice-cardh'>SHAP Feature Importance</div>", unsafe_allow_html=True)
        imp_labels = st.checkbox("Show values", value=True, key="imp_labels")
        st.plotly_chart(fig_imp(m, imp_labels), width="stretch",
                        config={"displayModeBar": False})

with tab_shap:
    with st.container(border=True):
        st.markdown(
            "<div class='slice-cardh'>SHAP Beeswarm — feature value vs impact on model output</div>",
            unsafe_allow_html=True)
        bee_colour = st.checkbox("Colour by feature value", value=True, key="bee_colour")
        st.plotly_chart(fig_bee(m, bee_colour), width="stretch",
                        config={"displayModeBar": False})

with tab_ai:
    ai = m.get("ai", {}) or {}
    a1, a2 = st.columns([4, 6])
    with a1.container(border=True):
        st.markdown("<div class='slice-cardh'>A · Model Summary</div>", unsafe_allow_html=True)
        st.markdown(ai_summary_html(ai), unsafe_allow_html=True)
    with a2.container(border=True):
        st.markdown("<div class='slice-cardh'>B · SHAP Interpretation (top features)</div>",
                    unsafe_allow_html=True)
        st.markdown(ai_shap_html(ai), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "<div class='slice-cardh'>C · Suggestable Audit Areas — based on top risk trend</div>",
            unsafe_allow_html=True)
        st.markdown(ai_audit_html(ai), unsafe_allow_html=True)
