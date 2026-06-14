"""Central registry of SLICE models and their data-file mappings.

Every path here is *relative to the project base directory* and is written
exactly as described in the original specification (the screenshots). Paths use
forward slashes; on Windows they resolve identically via ``pathlib``.

Both the dashboard generator (``generate_dashboard.py``) and the sample-data
maker (``make_sample_data.py``) import from this single source of truth so the
two never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --- App-level identity -----------------------------------------------------

APP_NAME = "SLICE"
APP_FULL_NAME = "Statistical Learning and Inference Compute Engine"
APP_OBJECTIVE = (
    "Using machine learning in deriving predictive factors that drive policy "
    "lapses within a short period."
)

# Relative path (from the project base dir) to the data dictionary workbook.
DATA_DICTIONARY = "Data Dictionary/data_dictionary.xlsx"


# --- Selection options shown to the user ------------------------------------

HORIZONS = [
    {"key": "12M", "label": "Policy Lapses Within 12M"},
    {"key": "4M", "label": "Policy Lapses Within 4M"},
]

SCOPES = [
    {"key": "All", "label": "Considering All Products"},
    {"key": "BAP", "label": "Considering Only BAP Product"},
]


@dataclass
class ModelSpec:
    """One selectable model and the files that describe it."""

    key: str                 # e.g. "12M|All" -- horizon|scope
    horizon: str             # "12M" / "4M"
    scope: str               # "All" / "BAP"
    pkl: str                 # model artifact name (identifier only)
    yaml: str                # training_config_*.yaml  (yaml_mapping_file)
    metrics: str             # metrics_*.csv           (metrics_mapping_file)
    pr_data: str             # pr_data_*.csv
    shap_importance: str     # SHAP importance_summary_*.csv
    shap_values: str         # SHAP values_detail_*.csv
    ai_json: str             # *aiexplain*.json        (AI_JSON_EXPLAIN)

    @property
    def horizon_label(self) -> str:
        return next(h["label"] for h in HORIZONS if h["key"] == self.horizon)

    @property
    def scope_label(self) -> str:
        return next(s["label"] for s in SCOPES if s["key"] == self.scope)

    @property
    def label(self) -> str:
        return f"{self.horizon_label} — {self.scope_label}"


MODELS: List[ModelSpec] = [
    ModelSpec(
        key="12M|All",
        horizon="12M",
        scope="All",
        pkl="lapsed_lte12m_lapse_after_12m_20260615_000615.pkl",
        yaml="Scripts/YAML_Archived/training_config_TS_12m.yaml",
        metrics="Reports/metrics_lapsed_lte12m_lapse_after_12m_20260615_000615.csv",
        pr_data="Reports/pr_data_lapsed_lte12m_lapse_after_12m_20260615_000615.csv",
        shap_importance="Reports/SHAP/lapsed_lte12m_importance_summary_20260615_001035.csv",
        shap_values="Reports/SHAP/lapsed_lte12m_values_detail_20260615_001035.csv",
        ai_json="Reports/lapsed_lte12m_aiexplain_20260615_000615_20260615_001355.json",
    ),
    ModelSpec(
        key="12M|BAP",
        horizon="12M",
        scope="BAP",
        pkl="lapsed_lte12m_bap_lapse_after_12m_20260615_002255.pkl",
        yaml="Scripts/YAML_Archived/training_config_TS_12mbap.yaml",
        metrics="Reports/metrics_lapsed_lte12m_bap_lapse_after_12m_20260615_002255.csv",
        pr_data="Reports/pr_data_lapsed_lte12m_bap_lapse_after_12m_20260615_002255.csv",
        shap_importance="Reports/SHAP/lapsed_lte12m_bap_importance_summary_20260615_003106.csv",
        shap_values="Reports/SHAP/lapsed_lte12m_bap_values_detail_20260615_003106.csv",
        ai_json="Reports/lapsed_lte12m_bap_aiexplain_20260615_002255_20260615_003831.json",
    ),
    ModelSpec(
        key="4M|All",
        horizon="4M",
        scope="All",
        pkl="lapsed_lte4m_lapse_after_4m_20260614_230458.pkl",
        yaml="Scripts/YAML_Archived/training_config_TS_4m.yaml",
        metrics="Reports/metrics_lapsed_lte4m_lapse_after_4m_20260614_230458.csv",
        pr_data="Reports/pr_data_lapsed_lte4m_lapse_after_4m_20260614_230458.csv",
        shap_importance="Reports/SHAP/lapsed_lte4m_importance_summary_20260614_230943.csv",
        shap_values="Reports/SHAP/lapsed_lte4m_values_detail_20260614_230943.csv",
        ai_json="Reports/lapsed_lte4m_aiexplain_20260614_230458_20260614_231415.json",
    ),
    ModelSpec(
        key="4M|BAP",
        horizon="4M",
        scope="BAP",
        pkl="lapsed_lte4m_bap_lapse_after_4m_20260614_234248.pkl",
        yaml="Scripts/YAML_Archived/training_config_TS_4mbap.yaml",
        metrics="Reports/metrics_lapsed_lte4m_bap_lapse_after_4m_20260614_234248.csv",
        pr_data="Reports/pr_data_lapsed_lte4m_bap_lapse_after_4m_20260614_234248.csv",
        shap_importance="Reports/SHAP/lapsed_lte4m_bap_importance_summary_20260614_234630.csv",
        shap_values="Reports/SHAP/lapsed_lte4m_bap_values_detail_20260614_234630.csv",
        ai_json="Reports/lapsed_lte4m_bap_aiexplain_20260614_234248_20260614_234927.json",
    ),
]


# --- Hyperparameter interpretation hints ------------------------------------
# Used to turn raw tuning values into the short plain-English notes required by
# the spec ("hyperparameters used for tuning: to get and *interpret* ...").

HYPERPARAMETER_NOTES = {
    "n_estimators": "Number of boosting rounds (trees). More trees increase "
                    "capacity but risk overfitting.",
    "max_depth": "Maximum tree depth. Higher values capture interactions but "
                 "can overfit; lower values regularise.",
    "learning_rate": "Step size shrinkage per boosting round. Lower values "
                     "learn more slowly but generalise better.",
    "eta": "Step size shrinkage per boosting round (alias of learning_rate).",
    "subsample": "Row sampling fraction per tree. <1 adds randomness and "
                 "reduces overfitting.",
    "colsample_bytree": "Column sampling fraction per tree. <1 decorrelates "
                        "trees and regularises.",
    "colsample_bylevel": "Column sampling fraction per split level.",
    "min_child_weight": "Minimum summed instance weight in a child. Higher "
                        "values make the model more conservative.",
    "gamma": "Minimum loss reduction required to split. Higher values prune "
             "weak splits.",
    "reg_alpha": "L1 regularisation on weights. Encourages sparsity.",
    "reg_lambda": "L2 regularisation on weights. Shrinks weights smoothly.",
    "scale_pos_weight": "Balances positive/negative classes for imbalanced "
                        "targets.",
    "num_leaves": "Maximum leaves per tree (LightGBM). Controls complexity.",
    "objective": "Training objective / loss function.",
    "eval_metric": "Metric optimised during early stopping / evaluation.",
    "early_stopping_rounds": "Stops training when validation metric stops "
                             "improving for N rounds.",
    "random_state": "Seed for reproducibility.",
    "seed": "Seed for reproducibility.",
    "tree_method": "Tree construction algorithm.",
}


def hyperparameter_note(name: str) -> str:
    """Best-effort plain-English note for a hyperparameter name."""
    return HYPERPARAMETER_NOTES.get(str(name).strip().lower(), "")
