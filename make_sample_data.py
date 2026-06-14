#!/usr/bin/env python3
"""Create a realistic *sample* project tree for the SLICE dashboard.

This is a convenience for demos and CI: it fabricates data that matches the
exact file layout and column schemas described in the specification, so that
``generate_dashboard.py`` has something to consume end-to-end. Replace the
generated tree with your real ``Scripts/``, ``Reports/`` and ``Data Dictionary/``
folders to produce the production dashboard.

Usage
-----
    python make_sample_data.py --base-dir sample_project
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from slice import registry
from slice.registry import ModelSpec


CATEGORICAL = [
    "e_policy_option", "payment_mode_code", "source_of_business_code",
    "non_lapse_guarantee_flag", "agent_unit_code", "agent_leader_name",
    "po_same_la", "po_same_payer",
]
NUMERICAL = [
    "sum_assured", "agent_num_customers_sold", "inception_month",
    "po_age", "la_age", "po_income_num",
]

# product_code_BAP only appears in the BAP-scoped models.
BAP_EXTRA = ["product_code_BAP"]

DESCRIPTIONS = {
    "e_policy_option": "Electronic policy option elected at application.",
    "payment_mode_code": "Premium payment frequency (e.g. annual, monthly).",
    "source_of_business_code": "Distribution channel / source of the policy.",
    "non_lapse_guarantee_flag": "Whether a non-lapse guarantee rider applies.",
    "agent_unit_code": "Servicing agent's unit / branch code.",
    "agent_leader_name": "Name of the agent's leader / supervisor.",
    "po_same_la": "Flag: policy owner is also the life assured.",
    "po_same_payer": "Flag: policy owner is also the premium payer.",
    "sum_assured": "Face amount / sum assured of the policy.",
    "agent_num_customers_sold": "Number of customers the agent has sold to.",
    "inception_month": "Calendar month the policy incepted.",
    "po_age": "Age of the policy owner at inception.",
    "la_age": "Age of the life assured at inception.",
    "po_income_num": "Declared annual income of the policy owner.",
    "product_code_BAP": "Indicator for the BAP product line.",
}

HYPERPARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 600,
    "max_depth": 10,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 6.0,
    "random_state": 42,
}


def _write_yaml(path: Path, spec: ModelSpec) -> None:
    cats = CATEGORICAL + (BAP_EXTRA if spec.scope == "BAP" else [])
    lines = []
    lines.append("# Auto-generated SAMPLE training config for the SLICE demo.")
    lines.append("variables:")
    lines.append("  categorical_predictors:")
    for c in cats:
        lines.append(f"    - {c}")
    lines.append("  numerical_predictors:")
    for n in NUMERICAL:
        lines.append(f"    - {n}")
    lines.append("")
    lines.append("split:")
    lines.append("  datetime_field: inception_date_dt")
    lines.append('  cutoff_date: "2026-01-01"')
    lines.append('  train_test_split_date: "2025-01-01"')
    lines.append("")
    lines.append("hyperparameters:")
    for k, v in HYPERPARAMS.items():
        if isinstance(v, str):
            lines.append(f'  {k}: "{v}"')
        else:
            lines.append(f"  {k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _features_for(spec: ModelSpec) -> list[str]:
    feats = (CATEGORICAL + NUMERICAL).copy()
    if spec.scope == "BAP":
        feats = BAP_EXTRA + feats
    return feats


def _write_metrics(path: Path, spec: ModelSpec, rng: np.random.Generator) -> None:
    test_pr = float(rng.uniform(0.40, 0.55))
    cv_pr = test_pr + float(rng.uniform(0.08, 0.16))   # train a bit higher
    base = float(rng.uniform(0.10, 0.18))
    df = pd.DataFrame([{
        "run_id": spec.pkl.split("_")[-1].replace(".pkl", ""),
        "model_file": spec.pkl,
        "cv_pr_auc": round(cv_pr, 5),
        "test_pr_auc": round(test_pr, 5),
        "test_positive_rate": round(base, 6),
    }])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_pr_data(path: Path, spec: ModelSpec, rng: np.random.Generator) -> None:
    thresholds = np.linspace(0.0, 1.0, 101)
    base = 0.15
    rows = []
    for t in thresholds:
        # recall decreases with threshold; precision increases (noisy).
        recall = float(np.clip(1.0 - t ** 0.8 + rng.normal(0, 0.01), 0, 1))
        precision = float(np.clip(base + (1 - base) * t ** 0.7 + rng.normal(0, 0.01), 0, 1))
        rows.append({
            "run_id": spec.pkl.split("_")[-1].replace(".pkl", ""),
            "model_file": spec.pkl,
            "train_test_split_date": "2025-01-01",
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "threshold": round(float(t), 4),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_shap(imp_path: Path, val_path: Path, spec: ModelSpec,
                rng: np.random.Generator) -> list[str]:
    feats = _features_for(spec)
    # importance: BAP product dominates for BAP-scoped models.
    base_imp = rng.uniform(0.05, 1.0, size=len(feats))
    base_imp = np.sort(base_imp)[::-1]
    if spec.scope == "BAP":
        base_imp[0] = base_imp[1] * 6  # dominance warning territory
    # match importance order to a shuffled-ish feature order
    order = list(np.argsort(base_imp)[::-1])
    ordered = [feats[i] for i in order]
    ordered_imp = sorted(base_imp, reverse=True)

    imp_rows = []
    for rank, (f, v) in enumerate(zip(ordered, ordered_imp), start=1):
        imp_rows.append({
            "run_id": spec.pkl.split("_")[-1].replace(".pkl", ""),
            "model_file": spec.pkl,
            "feature_rank": rank,
            "feature_name": f,
            "mean_abs_shap_value": round(float(v), 6),
        })
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(imp_rows).to_csv(imp_path, index=False)

    # values_detail: per-instance feature_value & shap_value.
    n_inst = 400
    val_rows = []
    for f, v in zip(ordered, ordered_imp):
        fvals = rng.normal(0, 1, size=n_inst)
        # shap correlated with feature value, scaled by importance.
        shap = fvals * float(v) * 0.5 + rng.normal(0, float(v) * 0.2, size=n_inst)
        for i in range(n_inst):
            val_rows.append({
                "run_id": spec.pkl.split("_")[-1].replace(".pkl", ""),
                "model_file": spec.pkl,
                "instance_idx": i,
                "feature_name": f,
                "feature_value": round(float(fvals[i]), 5),
                "shap_value": round(float(shap[i]), 6),
            })
    val_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(val_rows).to_csv(val_path, index=False)
    return ordered


def _write_ai(path: Path, spec: ModelSpec, top_feats: list[str]) -> None:
    horizon = spec.horizon
    scope_txt = "BAP product" if spec.scope == "BAP" else "all products"
    dominance = (
        f"{top_feats[0]} has a Mean|SHAP| vastly exceeding all other features."
        if spec.scope == "BAP" else
        "No single feature dominates; importance is spread across several drivers."
    )
    payload = {
        "overall_summary": (
            f"The XGBoost lapse model for {horizon} ({scope_txt}) shows fair to "
            f"good out-of-sample ranking quality, with modest overfitting risk and "
            f"no obvious sign of leakage."
        ),
        "confidence": "high",
        "consistency_flags": [],
        "model_explanation": {
            "primary_use": "diagnostic",
            "secondary_use": "prediction",
            "predictive_strength": "good",
            "base_rate_lift": "3.0x",
            "generalization": "good",
            "temporal_stability": "stable",
            "leakage_risk": "possible",
            "rationale": (
                "Test PR-AUC materially exceeds the lapse base rate, giving a "
                "useful lift for risk scoring and diagnostic use, subject to "
                "ongoing temporal validation."
            ),
        },
        "tips_and_techniques": [
            {
                "category": "regularization",
                "recommendation": "Reduce max_depth from 10 toward 6-8 and consider increasing reg_lambda.",
                "priority": "high",
                "rationale": "The train-test PR-AUC gap indicates mild overfitting.",
            },
            {
                "category": "validation",
                "recommendation": "Implement walk-forward / rolling-origin time-series cross-validation.",
                "priority": "high",
                "rationale": "The model is used across time and the current split is a single boundary.",
            },
        ],
        "shap_interpretation": {
            "top_features": [
                {
                    "feature": top_feats[0],
                    "global_importance_rank": 1,
                    "effect_direction": "Higher values push the prediction toward lapse.",
                    "effect_strength": "dominant" if spec.scope == "BAP" else "strong",
                    "notes": f"Largest mean |SHAP| among the {len(top_feats)} features considered.",
                },
                {
                    "feature": top_feats[1],
                    "global_importance_rank": 2,
                    "effect_direction": "Absence of the safeguard pushes prediction toward lapse.",
                    "effect_strength": "strong",
                    "notes": "Clear separation between value groups in the beeswarm.",
                },
                {
                    "feature": top_feats[2],
                    "global_importance_rank": 3,
                    "effect_direction": "Higher prior sales volume associates with lower lapse risk.",
                    "effect_strength": "moderate",
                    "notes": "Strong negative direction overall.",
                },
            ],
            "dominance_warning": dominance,
            "interaction_notes": "Several continuous features show interaction with agent attributes.",
            "overall_narrative": (
                "The model primarily segments lapse risk by product type and "
                "guarantee design, with agent and demographic features refining "
                "the ranking."
            ),
        },
        "audit_suggestion": {
            "areas": [
                {
                    "business_area": "Product design and governance for BAP and related product codes",
                    "linked_features_or_findings": ["product_code_BAP", "e_policy_option"],
                    "rationale": "The model attributes high lapse risk to product type.",
                    "priority": "high",
                    "regulatory_relevance": "Relevant to product oversight and governance expectations.",
                },
                {
                    "business_area": "Non-lapse guarantee feature design and administration",
                    "linked_features_or_findings": ["non_lapse_guarantee_flag"],
                    "rationale": "Absence of the guarantee is a strong upward driver of lapse risk.",
                    "priority": "medium",
                    "regulatory_relevance": "Relevant to policyholder fair-treatment reviews.",
                },
                {
                    "business_area": "Agent conduct and persistency monitoring",
                    "linked_features_or_findings": ["agent_unit_code", "agent_num_customers_sold"],
                    "rationale": "Agent attributes meaningfully shift predicted lapse risk.",
                    "priority": "medium",
                    "regulatory_relevance": "Relevant to market-conduct supervision.",
                },
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_data_dictionary(path: Path) -> None:
    rows = []
    for col, desc in DESCRIPTIONS.items():
        rows.append({"Data Name": "Data", "Column": col, "Description": desc})
    # a couple of non-"Data" rows to exercise the filter
    rows.append({"Data Name": "Meta", "Column": "run_id", "Description": "Training run identifier."})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate sample SLICE project data.")
    ap.add_argument("--base-dir", default="sample_project")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    print(f"Writing sample project data to: {base}")

    _write_data_dictionary(base / "Data Dictionary" / "data_dictionary.xlsx")

    for i, spec in enumerate(registry.MODELS):
        rng = np.random.default_rng(args.seed + i)
        _write_yaml(base / spec.yaml.replace("\\", "/"), spec)
        _write_metrics(base / spec.metrics.replace("\\", "/"), spec, rng)
        _write_pr_data(base / spec.pr_data.replace("\\", "/"), spec, rng)
        top = _write_shap(
            base / spec.shap_importance.replace("\\", "/"),
            base / spec.shap_values.replace("\\", "/"),
            spec, rng,
        )
        _write_ai(base / spec.ai_json.replace("\\", "/"), spec, top)
        print(f"  wrote data for {spec.key}")

    print("Sample data complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
