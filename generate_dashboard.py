#!/usr/bin/env python3
"""Generate the SLICE Quarto dashboard as a self-contained static website.

The script reads, for each of the four selectable models, the artifacts laid
out in :mod:`slice.registry`:

* ``yaml_mapping_file``     -> features, train/test split & cut-off, hyperparams
* ``metrics_mapping_file``  -> PR-AUC (train/test) and base rate
* ``pr_data``               -> precision / recall / threshold curve
* SHAP importance summary   -> mean |SHAP| per feature
* SHAP values detail        -> beeswarm points
* ``AI_JSON_EXPLAIN``       -> AI summary / SHAP interpretation / audit areas
* the data dictionary       -> human descriptions for every feature used

It consolidates everything into one JSON bundle that is inlined (together with
the Plotly.js library) into a hand-rolled, custom-layout Quarto HTML page. A
small vanilla-JS controller reacts to the model selector and the per-chart
toggles. Because both the data and Plotly are embedded, the rendered page has
**no runtime CDN dependency** and works offline (even from a ``file://`` URL).

Unless ``--no-render`` is given, the script also invokes ``quarto render`` to
produce a single self-contained ``slice_dashboard.html``.

Usage
-----
    python generate_dashboard.py --base-dir <project root> --output-dir site

Missing input files never abort the run: the affected section is filled with a
clearly-marked placeholder and a warning is logged, so a partial data drop still
produces a working dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from slice import registry
from slice.registry import ModelSpec

# Pandas / yaml are only needed when actually reading data; import lazily-safe.
import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _warn(msg: str) -> None:
    print(f"  [warn] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"  {msg}")


def _num(x: Any) -> Optional[float]:
    """Coerce to a JSON-safe float, mapping NaN/inf/blank to None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _resolve(base: Path, rel: str) -> Path:
    # Accept both "/" and "\" separators from the spec.
    return base / Path(rel.replace("\\", "/"))


def _first_present(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first column in *df* matching any candidate (case-insensitive)."""
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# --------------------------------------------------------------------------- #
# Data-dictionary lookups
# --------------------------------------------------------------------------- #

def load_data_dictionary(base: Path) -> Dict[str, str]:
    """Build ``{column_name: description}`` from the data dictionary workbook.

    Per the spec we keep rows where ``Data Name == "Data"`` and key by the
    ``Column`` value. We fall back gracefully if the sheet uses slightly
    different header names.
    """
    path = _resolve(base, registry.DATA_DICTIONARY)
    if not path.exists():
        _warn(f"data dictionary not found: {path}")
        return {}

    try:
        df = pd.read_excel(path)
    except Exception as exc:  # noqa: BLE001 - want to keep generating
        _warn(f"could not read data dictionary ({path}): {exc}")
        return {}

    data_name_col = _first_present(df, "Data Name", "DataName", "Dataset")
    column_col = _first_present(df, "Column", "Field", "Variable", "Feature")
    desc_col = _first_present(
        df, "Description", "Data Description", "Definition", "Desc", "Meaning"
    )
    if column_col is None or desc_col is None:
        _warn(f"data dictionary missing Column/Description headers: {list(df.columns)}")
        return {}

    if data_name_col is not None:
        mask = df[data_name_col].astype(str).str.strip().str.lower() == "data"
        # If the filter wipes everything out, fall back to the full sheet.
        sub = df[mask] if mask.any() else df
    else:
        sub = df

    mapping: Dict[str, str] = {}
    for _, row in sub.iterrows():
        col = str(row[column_col]).strip()
        desc = row[desc_col]
        if col and col.lower() != "nan":
            mapping[col] = "" if pd.isna(desc) else str(desc).strip()
    return mapping


# --------------------------------------------------------------------------- #
# YAML config (yaml_mapping_file)
# --------------------------------------------------------------------------- #

def load_yaml_config(base: Path, spec: ModelSpec, ddict: Dict[str, str]) -> Dict[str, Any]:
    path = _resolve(base, spec.yaml)
    out: Dict[str, Any] = {
        "cutoff_date": None,
        "train_test_split_date": None,
        "datetime_field": None,
        "categorical_predictors": [],
        "numerical_predictors": [],
        "hyperparameters": [],
    }
    if not path.exists():
        _warn(f"yaml config not found: {path}")
        return out

    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not parse yaml ({path}): {exc}")
        return cfg_fallback(out)

    split = cfg.get("split", {}) or {}
    out["cutoff_date"] = split.get("cutoff_date")
    out["train_test_split_date"] = split.get("train_test_split_date")
    out["datetime_field"] = split.get("datetime_field")

    variables = cfg.get("variables", {}) or {}

    def _features(names: Any) -> List[Dict[str, str]]:
        result = []
        for name in (names or []):
            name = str(name)
            result.append({"name": name, "description": ddict.get(name, "")})
        return result

    out["categorical_predictors"] = _features(variables.get("categorical_predictors"))
    out["numerical_predictors"] = _features(variables.get("numerical_predictors"))

    # Hyperparameters can live under a few common keys.
    hp_block = (
        cfg.get("hyperparameters")
        or cfg.get("model_params")
        or (cfg.get("model", {}) or {}).get("params")
        or (cfg.get("model", {}) or {}).get("hyperparameters")
        or (cfg.get("training", {}) or {}).get("params")
        or {}
    )
    if isinstance(hp_block, dict):
        for k, v in hp_block.items():
            out["hyperparameters"].append({
                "name": str(k),
                "value": _scalar(v),
                "interpretation": registry.hyperparameter_note(k),
            })
    return out


def cfg_fallback(out: Dict[str, Any]) -> Dict[str, Any]:
    return out


def _scalar(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


# --------------------------------------------------------------------------- #
# Metrics (metrics_mapping_file)
# --------------------------------------------------------------------------- #

def load_metrics(base: Path, spec: ModelSpec) -> Dict[str, Optional[float]]:
    path = _resolve(base, spec.metrics)
    out = {"cv_pr_auc": None, "test_pr_auc": None, "test_positive_rate": None}
    if not path.exists():
        _warn(f"metrics file not found: {path}")
        return out
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not read metrics ({path}): {exc}")
        return out
    if df.empty:
        return out
    row = df.iloc[-1]  # take the latest row if several
    for key in out:
        col = _first_present(df, key)
        if col is not None:
            out[key] = _num(row[col])
    return out


# --------------------------------------------------------------------------- #
# PR data
# --------------------------------------------------------------------------- #

def load_pr_data(base: Path, spec: ModelSpec) -> List[Dict[str, Optional[float]]]:
    path = _resolve(base, spec.pr_data)
    if not path.exists():
        _warn(f"pr_data file not found: {path}")
        return []
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not read pr_data ({path}): {exc}")
        return []

    p = _first_present(df, "precision")
    r = _first_present(df, "recall")
    t = _first_present(df, "threshold")
    if p is None or r is None:
        _warn(f"pr_data missing precision/recall columns: {path}")
        return []

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "precision": _num(row[p]),
            "recall": _num(row[r]),
            "threshold": _num(row[t]) if t else None,
        })
    # Order by threshold (ascending) when available for clean line plots.
    if t:
        rows.sort(key=lambda d: (d["threshold"] is None, d["threshold"]))
    return rows


# --------------------------------------------------------------------------- #
# SHAP importance + values
# --------------------------------------------------------------------------- #

def load_shap_importance(base: Path, spec: ModelSpec) -> List[Dict[str, Any]]:
    path = _resolve(base, spec.shap_importance)
    if not path.exists():
        _warn(f"shap importance not found: {path}")
        return []
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not read shap importance ({path}): {exc}")
        return []

    rank = _first_present(df, "feature_rank", "rank")
    name = _first_present(df, "feature_name", "feature")
    val = _first_present(df, "mean_abs_shap_value", "mean_abs_shap", "importance")
    if name is None or val is None:
        _warn(f"shap importance missing columns: {path}")
        return []

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rows.append({
            "feature_rank": int(row[rank]) if rank and not pd.isna(row[rank]) else i,
            "feature_name": str(row[name]),
            "mean_abs_shap_value": _num(row[val]),
        })
    rows.sort(key=lambda d: d["feature_rank"])
    return rows


def load_shap_values(
    base: Path,
    spec: ModelSpec,
    top_features: List[str],
    max_instances: int,
) -> List[Dict[str, Any]]:
    """Load beeswarm points, restricted to *top_features* and down-sampled.

    Keeping the full values_detail file (instances x features) in the bundle
    would bloat the page; we keep only the most important features and at most
    *max_instances* points per feature.
    """
    path = _resolve(base, spec.shap_values)
    if not path.exists():
        _warn(f"shap values not found: {path}")
        return []
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not read shap values ({path}): {exc}")
        return []

    name = _first_present(df, "feature_name", "feature")
    fval = _first_present(df, "feature_value", "value")
    sval = _first_present(df, "shap_value", "shap")
    if name is None or sval is None:
        _warn(f"shap values missing columns: {path}")
        return []

    keep = set(top_features)
    df = df[df[name].astype(str).isin(keep)]

    rows: List[Dict[str, Any]] = []
    for feat, grp in df.groupby(name, sort=False):
        if len(grp) > max_instances:
            grp = grp.sample(max_instances, random_state=7)
        for _, row in grp.iterrows():
            rows.append({
                "feature_name": str(feat),
                "feature_value": _num(row[fval]) if fval else None,
                "shap_value": _num(row[sval]),
            })
    return rows


# --------------------------------------------------------------------------- #
# AI JSON (AI_JSON_EXPLAIN)
# --------------------------------------------------------------------------- #

def load_ai_json(base: Path, spec: ModelSpec) -> Dict[str, Any]:
    path = _resolve(base, spec.ai_json)
    if not path.exists():
        _warn(f"AI explain json not found: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _warn(f"could not parse AI json ({path}): {exc}")
        return {}

    # The spec sometimes spells it "model_explaination"; normalise to one key.
    if "model_explanation" not in data and "model_explaination" in data:
        data["model_explanation"] = data["model_explaination"]
    return data


# --------------------------------------------------------------------------- #
# Bundle assembly
# --------------------------------------------------------------------------- #

def build_bundle(base: Path, top_n_beeswarm: int, max_instances: int) -> Dict[str, Any]:
    ddict = load_data_dictionary(base)
    _info(f"data dictionary: {len(ddict)} columns")

    models: Dict[str, Any] = {}
    for spec in registry.MODELS:
        _info(f"model {spec.key}  ({spec.pkl})")
        cfg = load_yaml_config(base, spec, ddict)
        metrics = load_metrics(base, spec)
        pr_data = load_pr_data(base, spec)
        importance = load_shap_importance(base, spec)
        top_feats = [r["feature_name"] for r in importance[:top_n_beeswarm]]
        shap_values = load_shap_values(base, spec, top_feats, max_instances)
        ai = load_ai_json(base, spec)

        models[spec.key] = {
            "key": spec.key,
            "horizon": spec.horizon,
            "scope": spec.scope,
            "label": spec.label,
            "horizon_label": spec.horizon_label,
            "scope_label": spec.scope_label,
            "pkl": spec.pkl,
            "description": cfg,
            "metrics": metrics,
            "pr_data": pr_data,
            "shap_importance": importance,
            "shap_values": shap_values,
            "ai": ai,
        }

    return {
        "app": {
            "name": registry.APP_NAME,
            "full_name": registry.APP_FULL_NAME,
            "objective": registry.APP_OBJECTIVE,
        },
        "options": {"horizons": registry.HORIZONS, "scopes": registry.SCOPES},
        "models": models,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- #
# Static asset writers (QMD + CSS + offline JS controller)
# --------------------------------------------------------------------------- #

def write_assets(out_dir: Path, bundle: dict) -> None:
    """Write the .qmd, stylesheet and the two inline includes that make the
    rendered page fully self-contained and offline (no CDN required):

    * ``_plotly.html`` -- the entire Plotly.js library inlined in a <script>.
    * ``_app.html``    -- the data bundle + vanilla-JS controller inlined.

    The dashboard layout is hand-rolled (a custom-layout Quarto HTML page) so
    that no framework reorganises / duplicates the DOM; every element id the
    controller relies on is therefore unique and stable.
    """
    from plotly.offline import get_plotlyjs

    (out_dir / "slice_dashboard.qmd").write_text(QMD_TEMPLATE, encoding="utf-8")
    (out_dir / "slice_theme.css").write_text(CSS_THEME, encoding="utf-8")

    plotly_js = get_plotlyjs()
    (out_dir / "_plotly.html").write_text(
        "<script type=\"text/javascript\">\n" + plotly_js + "\n</script>\n",
        encoding="utf-8",
    )

    data_js = json.dumps(bundle, separators=(",", ":"))
    app = (
        "<script type=\"application/json\" id=\"slice-data\">\n"
        + data_js
        + "\n</script>\n<script type=\"text/javascript\">\n"
        + APP_JS
        + "\n</script>\n"
    )
    (out_dir / "_app.html").write_text(app, encoding="utf-8")


# --- Quarto source ----------------------------------------------------------
# A custom-layout HTML page. The body is one raw-HTML shell (passed through
# verbatim by pandoc); all dynamic content is filled by the inlined controller
# in _app.html, which keys off the ids declared here. No executable cells, so
# no Jupyter/R kernel is needed to render.

QMD_TEMPLATE = r'''---
title: "SLICE"
pagetitle: "SLICE · Statistical Learning and Inference Compute Engine"
format:
  html:
    page-layout: custom
    theme: cosmo
    css: slice_theme.css
    embed-resources: true
    include-in-header: _plotly.html
    include-after-body: _app.html
    toc: false
---

```{=html}
<div class="slice-app">
  <header class="slice-header">
    <div class="slice-brand">
      <span class="slice-logo">SLICE</span>
      <span class="slice-sub">Statistical Learning and Inference Compute Engine</span>
    </div>
    <div class="slice-genstamp" id="slice-genstamp"></div>
  </header>

  <div class="slice-body">
    <aside class="slice-aside">
      <h2>Model selection</h2>

      <label class="slice-lbl" for="sel-horizon">Response variable</label>
      <select id="sel-horizon" class="slice-select"></select>

      <label class="slice-lbl" for="sel-scope">Product scope</label>
      <select id="sel-scope" class="slice-select"></select>

      <div id="sb-meta" class="slice-sb-meta"></div>
      <hr>
      <div class="slice-objective">
        <b>Objective.</b> Using machine learning to derive the predictive
        factors that drive policy lapses within a short period.
      </div>
    </aside>

    <main class="slice-main">
      <section class="slice-kpis">
        <div class="kpi kpi-a"><div class="kpi-label">PR-AUC · Train (CV)</div><div class="kpi-val" id="kpi-cv">—</div></div>
        <div class="kpi kpi-b"><div class="kpi-label">PR-AUC · Test</div><div class="kpi-val" id="kpi-test">—</div></div>
        <div class="kpi kpi-c"><div class="kpi-label">Base Rate (test positives)</div><div class="kpi-val" id="kpi-base">—</div></div>
        <div class="kpi kpi-d"><div class="kpi-label">Training cut-off</div><div class="kpi-val" id="kpi-cutoff">—</div></div>
      </section>

      <nav class="slice-tabs">
        <button class="slice-tab is-active" data-tab="perf">① Performance &amp; Model</button>
        <button class="slice-tab" data-tab="shap">② SHAP Beeswarm</button>
        <button class="slice-tab" data-tab="ai">③ AI Summary</button>
      </nav>

      <section class="slice-panel is-active" data-panel="perf">
        <div class="slice-grid grid-5050">
          <div class="card">
            <div class="card-h">Model &amp; Features</div>
            <div class="card-b">
              <div id="model-head" class="slice-modelhead"></div>
              <div id="feature-table" class="slice-scroll"></div>
            </div>
          </div>
          <div class="card">
            <div class="card-h">Hyperparameters</div>
            <div class="card-b"><div id="hp-table" class="slice-scroll"></div></div>
          </div>
        </div>
        <div class="slice-grid grid-5050">
          <div class="card">
            <div class="card-h">Precision–Recall
              <span class="card-tools">
                <label><input type="radio" name="pr-view" value="curve" checked> PR curve</label>
                <label><input type="radio" name="pr-view" value="threshold"> vs threshold</label>
              </span>
            </div>
            <div class="card-b"><div id="pr-plot" class="slice-plot"></div></div>
          </div>
          <div class="card">
            <div class="card-h">SHAP Feature Importance
              <span class="card-tools"><label><input type="checkbox" id="imp-labels" checked> values</label></span>
            </div>
            <div class="card-b"><div id="imp-plot" class="slice-plot"></div></div>
          </div>
        </div>
      </section>

      <section class="slice-panel" data-panel="shap">
        <div class="card">
          <div class="card-h">SHAP Beeswarm — feature value vs impact on model output
            <span class="card-tools"><label><input type="checkbox" id="bee-color" checked> colour by feature value</label></span>
          </div>
          <div class="card-b"><div id="bee-plot" class="slice-plot slice-plot-tall"></div></div>
        </div>
      </section>

      <section class="slice-panel" data-panel="ai">
        <div class="slice-grid grid-4060">
          <div class="card">
            <div class="card-h">A · Model Summary</div>
            <div class="card-b"><div id="ai-summary"></div></div>
          </div>
          <div class="card">
            <div class="card-h">B · SHAP Interpretation (top features)</div>
            <div class="card-b"><div id="ai-shap"></div></div>
          </div>
        </div>
        <div class="card">
          <div class="card-h">C · Suggestable Audit Areas — based on top risk trend</div>
          <div class="card-b"><div id="ai-audit"></div></div>
        </div>
      </section>
    </main>
  </div>
</div>
```
'''


CSS_THEME = r'''
:root {
  --slice-primary: #2c6fbb;
  --slice-accent: #e07b39;
  --slice-ink: #2b3138;
  --slice-line: #e6e9ee;
}
/* Quarto's page-layout:custom turns <body> into a grid and emits a title
   block; neutralise both so our hand-rolled layout owns the page. */
header#title-block-header, .quarto-title-block { display: none !important; }
body { margin: 0 !important; background: #eef1f5; color: var(--slice-ink);
  display: block !important; }
.slice-app { font-size: 15px; width: 100%; }

.slice-header {
  background: linear-gradient(90deg, #1f4f87, #2c6fbb);
  color: #fff; padding: .7rem 1.2rem; display: flex; align-items: baseline;
  justify-content: space-between; box-shadow: 0 2px 6px rgba(0,0,0,.15);
}
.slice-logo { font-size: 1.55rem; font-weight: 800; letter-spacing: 2px; }
.slice-sub { margin-left: .8rem; font-size: .92rem; opacity: .9; }
.slice-genstamp { font-size: .72rem; opacity: .75; }

.slice-body { display: flex !important; gap: 1rem; padding: 1rem 1.2rem; align-items: flex-start; }

.slice-aside {
  flex: 0 0 300px;
  background: #fff; border: 1px solid var(--slice-line); border-radius: 10px;
  padding: 1rem; position: sticky; top: 1rem;
}
.slice-aside h2 { font-size: 1rem; margin: 0 0 .8rem; color: var(--slice-primary); }
.slice-lbl { display: block; font-weight: 600; font-size: .82rem; margin: .6rem 0 .25rem; }
.slice-select {
  width: 100%; padding: .45rem .5rem; border: 1px solid #ccd3da; border-radius: 6px;
  background: #fff; font-size: .9rem;
}
.slice-sb-meta { font-size: .8rem; color: #555; line-height: 1.5; margin-top: 1rem; }
.slice-sb-meta code, .slice-modelhead code, .slice-table code {
  background: #f1f3f5; padding: 0 .3rem; border-radius: 4px; font-size: .82em;
}
.slice-objective { font-size: .8rem; color: #555; line-height: 1.45; }
.slice-aside hr { border: 0; border-top: 1px solid var(--slice-line); margin: 1rem 0; }

.slice-main { flex: 1 1 auto; min-width: 0; }
.slice-kpis { display: grid !important; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1rem; }
.kpi { border-radius: 10px; padding: .9rem 1rem; color: #fff; box-shadow: 0 2px 5px rgba(0,0,0,.08); }
.kpi-label { font-size: .82rem; opacity: .92; }
.kpi-val { font-size: 2rem; font-weight: 700; margin-top: .2rem; }
.kpi-a { background: linear-gradient(135deg, #2c6fbb, #4b8fd6); }
.kpi-b { background: linear-gradient(135deg, #6f54c9, #9173e0); }
.kpi-c { background: linear-gradient(135deg, #2f3b47, #51606e); }
.kpi-d { background: linear-gradient(135deg, #138a72, #1cb494); }

.slice-tabs { display: flex; gap: .4rem; border-bottom: 2px solid var(--slice-line); margin-bottom: 1rem; }
.slice-tab {
  border: 0; background: transparent; font-size: .95rem; font-weight: 600; color: #6b7480;
  padding: .55rem .9rem; cursor: pointer; border-bottom: 3px solid transparent; margin-bottom: -2px;
}
.slice-tab:hover { color: var(--slice-primary); }
.slice-tab.is-active { color: var(--slice-primary); border-bottom-color: var(--slice-primary); }

.slice-panel { display: none; }
.slice-panel.is-active { display: block; }

.slice-grid { display: grid !important; gap: 1rem; margin-bottom: 1rem; }
.grid-5050 { grid-template-columns: 1fr 1fr; }
.grid-4060 { grid-template-columns: 4fr 6fr; }
@media (max-width: 1100px) { .grid-5050, .grid-4060, .slice-kpis { grid-template-columns: 1fr; }
  .slice-body { grid-template-columns: 1fr; } }

.card { background: #fff; border: 1px solid var(--slice-line); border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,.05); overflow: hidden; }
.card-h {
  font-weight: 700; font-size: .92rem; padding: .6rem .9rem; border-bottom: 1px solid var(--slice-line);
  background: #f7f9fc; display: flex; align-items: center; justify-content: space-between; gap: .5rem;
}
.card-tools { font-weight: 400; font-size: .8rem; color: #555; }
.card-tools label { margin-left: .8rem; cursor: pointer; }
.card-b { padding: .8rem .9rem; }
.slice-scroll { max-height: 320px; overflow: auto; }

.slice-plot { width: 100%; height: 320px; }
.slice-plot-tall { height: 520px; }

.slice-modelhead { margin-bottom: .7rem; }
.slice-pill { display: inline-block; background: var(--slice-primary); color: #fff; border-radius: 999px;
  padding: .15rem .7rem; font-size: .76rem; font-weight: 600; margin-right: .35rem; }
.slice-pill-alt { background: var(--slice-accent); }
.slice-artifact { color: #777; font-size: .76rem; margin-top: .4rem; }

.slice-table { width: 100%; border-collapse: collapse; font-size: .84rem; }
.slice-table th { text-align: left; background: #f1f3f5; padding: .4rem .55rem; border-bottom: 2px solid #dee2e6;
  position: sticky; top: 0; }
.slice-table td { padding: .35rem .55rem; border-bottom: 1px solid #eee; vertical-align: top; }
.slice-table tbody tr:hover { background: #f8fbff; }

.slice-tag { display: inline-block; font-size: .72rem; padding: .05rem .45rem; border-radius: 4px; background: #e9ecef; color: #495057; }
.slice-tag-categorical { background: #e3f0ff; color: #1c5fa8; }
.slice-tag-numerical { background: #fdeede; color: #b5651d; }
.slice-chip { background: #eef2f7; padding: 0 .3rem; border-radius: 4px; margin: 0 .15rem .15rem 0; display: inline-block; font-size: .8rem; }

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
'''


# --- Vanilla-JS controller (inlined into _app.html) -------------------------

APP_JS = r'''
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("slice-data").textContent);
  var PRIMARY = "#2c6fbb", ACCENT = "#e07b39";

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c];
    });
  }
  function fmt3(x) { return (x === null || x === undefined) ? "—" : Number(x).toFixed(3); }
  function pct1(x) { return (x === null || x === undefined) ? "—" : (Number(x) * 100).toFixed(1) + "%"; }

  function currentModel() {
    return DATA.models[$("sel-horizon").value + "|" + $("sel-scope").value];
  }
  function fillSelect(sel, opts) {
    sel.innerHTML = opts.map(function (o) {
      return "<option value=\"" + esc(o.key) + "\">" + esc(o.label) + "</option>";
    }).join("");
  }

  function updateKpis(m) {
    $("kpi-cv").textContent = fmt3(m.metrics.cv_pr_auc);
    $("kpi-test").textContent = fmt3(m.metrics.test_pr_auc);
    $("kpi-base").textContent = pct1(m.metrics.test_positive_rate);
    $("kpi-cutoff").textContent = m.description.cutoff_date || "—";
  }

  function updateHead(m) {
    var d = m.description;
    $("sb-meta").innerHTML =
      "<div><b>Artifact</b><br><code>" + esc(m.pkl) + "</code></div>" +
      "<div style=\"margin-top:.6rem\"><b>Temporal split</b><br>" +
      "datetime <code>" + esc(d.datetime_field || "—") + "</code><br>" +
      "split <code>" + esc(d.train_test_split_date || "—") + "</code><br>" +
      "cut-off <code>" + esc(d.cutoff_date || "—") + "</code></div>";
    $("model-head").innerHTML =
      "<span class=\"slice-pill\">" + esc(m.horizon_label) + "</span>" +
      "<span class=\"slice-pill slice-pill-alt\">" + esc(m.scope_label) + "</span>" +
      "<div class=\"slice-artifact\">artifact: <code>" + esc(m.pkl) + "</code></div>";
  }

  function updateFeatures(m) {
    var rows = [];
    (m.description.categorical_predictors || []).forEach(function (f) {
      rows.push({type: "categorical", name: f.name, description: f.description}); });
    (m.description.numerical_predictors || []).forEach(function (f) {
      rows.push({type: "numerical", name: f.name, description: f.description}); });
    if (!rows.length) { $("feature-table").innerHTML = "<em>No feature metadata available.</em>"; return; }
    var h = "<table class=\"slice-table\"><thead><tr><th>Feature</th><th>Type</th><th>Description</th></tr></thead><tbody>";
    rows.forEach(function (r) {
      h += "<tr><td><code>" + esc(r.name) + "</code></td><td><span class=\"slice-tag slice-tag-" + r.type +
           "\">" + r.type + "</span></td><td>" + esc(r.description || "—") + "</td></tr>";
    });
    $("feature-table").innerHTML = h + "</tbody></table>";
  }

  function updateHp(m) {
    var hp = m.description.hyperparameters || [];
    if (!hp.length) { $("hp-table").innerHTML = "<em>No hyperparameter metadata available.</em>"; return; }
    var h = "<table class=\"slice-table\"><thead><tr><th>Parameter</th><th>Value</th><th>Interpretation</th></tr></thead><tbody>";
    hp.forEach(function (p) {
      h += "<tr><td><code>" + esc(p.name) + "</code></td><td><b>" + esc(p.value == null ? "—" : p.value) +
           "</b></td><td>" + esc(p.interpretation || "—") + "</td></tr>";
    });
    $("hp-table").innerHTML = h + "</tbody></table>";
  }

  var LAYOUT_BASE = {
    margin: {l: 55, r: 20, t: 10, b: 45}, font: {size: 12},
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    legend: {orientation: "h", y: 1.12}
  };
  var CONFIG = {displayModeBar: false, responsive: true};

  function drawPr(m) {
    var d = m.pr_data || [], node = $("pr-plot");
    if (!d.length) { node.innerHTML = "<em>No PR data available.</em>"; return; }
    var view = (document.querySelector("input[name=pr-view]:checked") || {}).value || "curve";
    var traces, layout;
    if (view === "curve") {
      traces = [{x: d.map(function (r) { return r.recall; }), y: d.map(function (r) { return r.precision; }),
        mode: "lines", line: {color: PRIMARY, width: 3},
        hovertemplate: "recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>"}];
      layout = Object.assign({}, LAYOUT_BASE, {
        xaxis: {title: "Recall", range: [0, 1], gridcolor: "#eee"},
        yaxis: {title: "Precision", range: [0, 1], gridcolor: "#eee"},
        shapes: [{type: "line", x0: 0, x1: 1, y0: m.metrics.test_positive_rate,
          y1: m.metrics.test_positive_rate, line: {color: "#bbb", dash: "dash"}}], showlegend: false});
    } else {
      var t = d.filter(function (r) { return r.threshold !== null; });
      traces = [
        {x: t.map(function (r) { return r.threshold; }), y: t.map(function (r) { return r.precision; }),
         mode: "lines", name: "precision", line: {color: PRIMARY, width: 3}},
        {x: t.map(function (r) { return r.threshold; }), y: t.map(function (r) { return r.recall; }),
         mode: "lines", name: "recall", line: {color: ACCENT, width: 3}}];
      layout = Object.assign({}, LAYOUT_BASE, {
        xaxis: {title: "Threshold", range: [0, 1], gridcolor: "#eee"},
        yaxis: {title: "Score", range: [0, 1], gridcolor: "#eee"}});
    }
    Plotly.react(node, traces, layout, CONFIG);
  }

  function drawImp(m) {
    var node = $("imp-plot"), d = (m.shap_importance || []).slice(0, 15);
    if (!d.length) { node.innerHTML = "<em>No SHAP importance available.</em>"; return; }
    d = d.slice().reverse();
    var vals = d.map(function (r) { return r.mean_abs_shap_value; });
    var trace = {type: "bar", orientation: "h", x: vals, y: d.map(function (r) { return r.feature_name; }),
      marker: {color: PRIMARY},
      text: $("imp-labels").checked ? vals.map(function (v) { return v.toFixed(3); }) : null,
      textposition: "outside", cliponaxis: false,
      hovertemplate: "%{y}<br>mean|SHAP| %{x:.4f}<extra></extra>"};
    var layout = Object.assign({}, LAYOUT_BASE, {margin: {l: 190, r: 45, t: 10, b: 40},
      xaxis: {title: "mean |SHAP value|", gridcolor: "#eee"}, yaxis: {automargin: true}});
    Plotly.react(node, [trace], layout, CONFIG);
  }

  function drawBee(m) {
    var node = $("bee-plot"), raw = m.shap_values || [];
    if (!raw.length) { node.innerHTML = "<em>No SHAP value detail available.</em>"; return; }
    var order = (m.shap_importance || []).map(function (r) { return r.feature_name; });
    var idx = {}; order.forEach(function (f, i) { idx[f] = i; });
    var rng = {};
    raw.forEach(function (r) {
      if (r.feature_value === null || !(r.feature_name in idx)) return;
      var c = rng[r.feature_name] || {min: Infinity, max: -Infinity};
      c.min = Math.min(c.min, r.feature_value); c.max = Math.max(c.max, r.feature_value);
      rng[r.feature_name] = c;
    });
    var colour = $("bee-color").checked, xs = [], ys = [], cs = [], texts = [];
    raw.forEach(function (r) {
      if (!(r.feature_name in idx)) return;
      var base = order.length - 1 - idx[r.feature_name];
      ys.push(base + (Math.random() - 0.5) * 0.7); xs.push(r.shap_value);
      var c = rng[r.feature_name], norm = 0.5;
      if (c && c.max > c.min && r.feature_value !== null) norm = (r.feature_value - c.min) / (c.max - c.min);
      cs.push(norm); texts.push(r.feature_name);
    });
    var marker = colour
      ? {size: 5, opacity: 0.6, color: cs, colorscale: "RdBu", reversescale: true, cmin: 0, cmax: 1,
         colorbar: {title: "feature<br>value", thickness: 12, tickvals: [0, 1], ticktext: ["low", "high"]}}
      : {size: 5, opacity: 0.6, color: PRIMARY};
    var trace = {type: "scattergl", mode: "markers", x: xs, y: ys, marker: marker, text: texts,
      hovertemplate: "%{text}<br>SHAP %{x:.4f}<extra></extra>"};
    var layout = Object.assign({}, LAYOUT_BASE, {margin: {l: 190, r: 20, t: 10, b: 45},
      xaxis: {title: "SHAP value (impact on model output)", gridcolor: "#eee", zeroline: true, zerolinecolor: "#999"},
      yaxis: {tickvals: order.map(function (_, i) { return order.length - 1 - i; }), ticktext: order, automargin: true},
      showlegend: false});
    Plotly.react(node, [trace], layout, CONFIG);
  }

  function updateAi(m) {
    var ai = m.ai || {}, me = ai.model_explanation || {};
    var conf = ai.confidence ? "<div class=\"slice-confline\">confidence: <span class=\"slice-conf slice-conf-" +
      esc(ai.confidence) + "\">" + esc(ai.confidence) + "</span></div>" : "";
    var kv = Object.keys(me).filter(function (k) { return k !== "rationale"; }).map(function (k) {
      return "<tr><td>" + esc(k.replace(/_/g, " ")) + "</td><td>" + esc(me[k]) + "</td></tr>"; }).join("");
    var h = "<div class=\"slice-summary\">" + esc(ai.overall_summary || "No AI summary available.") + conf + "</div>";
    if (kv) h += "<table class=\"slice-table slice-kv\"><thead><tr><th>Aspect</th><th>Assessment</th></tr></thead><tbody>" + kv + "</tbody></table>";
    if (me.rationale) h += "<p class=\"slice-rationale\"><b>Rationale.</b> " + esc(me.rationale) + "</p>";
    $("ai-summary").innerHTML = h;

    var si = ai.shap_interpretation || {}, tf = si.top_features || [], sh;
    if (tf.length) {
      sh = "<table class=\"slice-table\"><thead><tr><th>#</th><th>Feature</th><th>Direction</th><th>Strength</th><th>Notes</th></tr></thead><tbody>";
      tf.forEach(function (f) {
        sh += "<tr><td>" + esc(f.global_importance_rank == null ? "—" : f.global_importance_rank) +
              "</td><td><code>" + esc(f.feature) + "</code></td><td>" + esc(f.effect_direction || "—") +
              "</td><td><span class=\"slice-tag\">" + esc(f.effect_strength || "—") + "</span></td><td>" +
              esc(f.notes || "—") + "</td></tr>";
      });
      sh += "</tbody></table>";
    } else { sh = "<em>No SHAP interpretation available.</em>"; }
    if (si.overall_narrative) sh += "<p class=\"slice-rationale\">" + esc(si.overall_narrative) + "</p>";
    $("ai-shap").innerHTML = sh;

    var areas = (ai.audit_suggestion || {}).areas || [], a;
    if (areas.length) {
      a = "<table class=\"slice-table\"><thead><tr><th>Business area</th><th>Priority</th><th>Linked features / findings</th><th>Rationale</th><th>Regulatory relevance</th></tr></thead><tbody>";
      areas.forEach(function (x) {
        var chips = (x.linked_features_or_findings || []).map(function (c) {
          return "<code class=\"slice-chip\">" + esc(c) + "</code>"; }).join(" ");
        a += "<tr><td><b>" + esc(x.business_area || "—") + "</b></td><td><span class=\"slice-prio slice-prio-" +
             esc((x.priority || "").toLowerCase()) + "\">" + esc(x.priority || "—") + "</span></td><td>" + chips +
             "</td><td>" + esc(x.rationale || "—") + "</td><td>" + esc(x.regulatory_relevance || "—") + "</td></tr>";
      });
      a += "</tbody></table>";
    } else { a = "<em>No audit suggestions available.</em>"; }
    $("ai-audit").innerHTML = a;
  }

  function activePanel() {
    var el = document.querySelector(".slice-panel.is-active");
    return el ? el.getAttribute("data-panel") : "perf";
  }
  function drawPanel(m, panel) {
    if (panel === "perf") { drawPr(m); drawImp(m); }
    else if (panel === "shap") { drawBee(m); }
  }

  function renderAll() {
    var m = currentModel(); if (!m) return;
    updateKpis(m); updateHead(m); updateFeatures(m); updateHp(m); updateAi(m);
    drawPanel(m, activePanel());
  }

  function init() {
    if ($("slice-genstamp") && DATA.generated_at)
      $("slice-genstamp").textContent = "generated " + DATA.generated_at;
    fillSelect($("sel-horizon"), DATA.options.horizons);
    fillSelect($("sel-scope"), DATA.options.scopes);
    $("sel-horizon").addEventListener("change", renderAll);
    $("sel-scope").addEventListener("change", renderAll);
    Array.prototype.forEach.call(document.querySelectorAll("input[name=pr-view]"), function (el) {
      el.addEventListener("change", function () { drawPr(currentModel()); }); });
    $("imp-labels").addEventListener("change", function () { drawImp(currentModel()); });
    $("bee-color").addEventListener("change", function () { drawBee(currentModel()); });

    Array.prototype.forEach.call(document.querySelectorAll(".slice-tab"), function (btn) {
      btn.addEventListener("click", function () {
        var tab = btn.getAttribute("data-tab");
        Array.prototype.forEach.call(document.querySelectorAll(".slice-tab"), function (b) {
          b.classList.toggle("is-active", b === btn); });
        Array.prototype.forEach.call(document.querySelectorAll(".slice-panel"), function (p) {
          p.classList.toggle("is-active", p.getAttribute("data-panel") === tab); });
        drawPanel(currentModel(), tab);  // (re)draw now that the panel has size
      });
    });
    renderAll();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
'''


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_with_quarto(out_dir: Path, quarto_bin: str) -> bool:
    qmd = out_dir / "slice_dashboard.qmd"
    quarto = shutil.which(quarto_bin) or quarto_bin
    _info(f"rendering with: {quarto}")
    try:
        res = subprocess.run(
            [quarto, "render", str(qmd.name)],
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        _warn(f"quarto not found ('{quarto_bin}'); wrote sources only. "
              f"Run: quarto render {qmd}")
        return False
    if res.returncode != 0:
        _warn("quarto render failed:")
        print(res.stdout[-3000:], file=sys.stderr)
        print(res.stderr[-3000:], file=sys.stderr)
        return False
    _info(f"rendered: {out_dir / 'slice_dashboard.html'}")
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the SLICE Quarto dashboard.")
    ap.add_argument("--base-dir", default=".",
                    help="Project root containing Scripts/, Reports/, Data Dictionary/.")
    ap.add_argument("--output-dir", default="site",
                    help="Where to write the .qmd, data bundle and rendered HTML.")
    ap.add_argument("--top-n-beeswarm", type=int, default=12,
                    help="Number of top features to include in the beeswarm.")
    ap.add_argument("--max-instances", type=int, default=300,
                    help="Max beeswarm points kept per feature.")
    ap.add_argument("--no-render", action="store_true",
                    help="Write sources but skip 'quarto render'.")
    ap.add_argument("--quarto-bin", default="quarto",
                    help="Path to the quarto executable.")
    ap.add_argument("--write-bundle", action="store_true",
                    help="Also write the raw slice_data.json (for debugging).")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"SLICE dashboard generator")
    print(f"  base   : {base}")
    print(f"  output : {out_dir}")

    bundle = build_bundle(base, args.top_n_beeswarm, args.max_instances)

    # Optional: also drop the raw bundle next to the sources for debugging.
    if args.write_bundle:
        (out_dir / "slice_data.json").write_text(
            json.dumps(bundle, indent=2), encoding="utf-8"
        )
        _info(f"wrote bundle: {out_dir / 'slice_data.json'}")

    write_assets(out_dir, bundle)
    _info("wrote slice_dashboard.qmd + slice_theme.css + _plotly.html + _app.html")

    if not args.no_render:
        render_with_quarto(out_dir, args.quarto_bin)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
