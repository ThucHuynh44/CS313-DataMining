"""
BRCA IDC vs ILC classification with nested Group-Stratified CV and hold-out test.

Bundle v4 (IDC vs ILC):
- Preprocess: log2(1+x) (sparse-safe) -> VarianceThreshold -> MaxAbsScaler -> SelectKBest -> (optional) CorrelationFilter
- Saves BOTH:
  * {prefix}_input_feature_cols.json  (FULL input genes; REQUIRED for inference with saved pipeline)
  * {prefix}_selected_feature_cols.json (genes remaining after var+kbest+corr for best estimator)
  * {prefix}_metadata.json (config, best params, metrics, versions, timestamps)
  * {prefix}_model.joblib (best pipeline)

Default label map (edit if your labels differ):
  - "Lobular carcinoma"               -> 0  (ILC)
  - "Infiltrating duct carcinoma"     -> 1  (IDC)

Main APIs:
- run_brca_experiment(cfg) -> results dict
- save_inference_bundle(results, out_dir, prefix="brca_idc_ilc") -> paths
- load_inference_bundle(out_dir, prefix="brca_idc_ilc") -> (model, input_cols, selected_cols, metadata)
- align_features(X_df, input_cols) -> reorder/fill to full input gene list
- create_gene_mapper(...) + plot_top_features(...)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Tuple, List, Optional, Callable

import warnings
warnings.filterwarnings("ignore")

import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import sparse

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    xgboost = None  # type: ignore
    XGBClassifier = None  # type: ignore


# Default: ILC=0, IDC=1 (edit if your dataset uses other strings)
LABEL_MAP_DEFAULT = {
    "Lobular carcinoma": 1,                 # ILC
    "Infiltrating duct carcinoma": 0,       # IDC
}

CLASS_NAMES_DEFAULT = ("ILC (1)", "IDC (0)")


@dataclass
class ExperimentConfig:
    MATRIX_PATH: str = "./brca_matrix_with_label_subtyping.tsv"
    META_PATH: str = "./breast_cancer_annotation.csv"

    MODEL: str = "rf"  # "logreg_l2" | "linear_svm" | "rf" | "xgb"

    # Labeling / reporting
    LABEL_MAP: Dict[str, int] = field(default_factory=lambda: dict(LABEL_MAP_DEFAULT))
    CLASS_NAMES: Tuple[str, str] = CLASS_NAMES_DEFAULT  # (class0_name, class1_name)

    # Preprocess / feature selection
    K_FEATURE_LIST: Tuple[int, ...] | int = (100, 500, 1000, 2000)
    VAR_THRESHOLD: float = 0.0

    USE_CORR_FILTER: bool = True
    CORR_THRESHOLD: float = 0.95

    # CV & Search
    OUTER_SPLITS: int = 5
    OUTER_FOLD_INDEX_AS_TEST: int = 0
    INNER_SPLITS: int = 5
    SCORING: str = "average_precision"  # or "roc_auc"
    N_JOBS: int = -1

    # Reproducibility
    RANDOM_STATE_OUTER: int = 42
    RANDOM_STATE_INNER: int = 123
    RANDOM_STATE_MODEL: int = 42

    # XGB defaults
    XGB_TREE_METHOD: str = "hist"


# -----------------------------
# Gene mapping + plotting helpers
# -----------------------------
def create_gene_mapper(
    csv_path: str,
) -> Tuple[Callable[[str], Optional[str]], Callable[[List[str]], List[Optional[str]]]]:
    df = pd.read_csv(csv_path)
    if not {"gene_id", "gene_name"}.issubset(df.columns):
        raise ValueError("gene map CSV must contain columns: 'gene_id' and 'gene_name'")
    mapping = dict(zip(df["gene_id"].astype(str), df["gene_name"].astype(str)))

    def mapper(gene_id: str) -> Optional[str]:
        gid = str(gene_id)
        if gid in mapping:
            return mapping[gid]
        base = gid.split(".", 1)[0]
        return mapping.get(base, None)

    def map_genes(gene_ids: List[str]) -> List[Optional[str]]:
        return [mapper(gid) for gid in gene_ids]

    return mapper, map_genes


def plot_top_features(
    feat_df: pd.DataFrame,
    top_n: int = 10,
    gene_mapper: Optional[Callable[[str], Optional[str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    if "importance_abs" in feat_df.columns:
        plot_df = feat_df[["feature", "importance_abs"]].head(top_n).iloc[::-1]
        values = plot_df["importance_abs"].values
        default_title = f"Top {top_n} Features (|weight|)"
    elif "importance" in feat_df.columns:
        plot_df = feat_df[["feature", "importance"]].head(top_n).iloc[::-1]
        values = plot_df["importance"].values
        default_title = f"Top {top_n} Features (importance)"
    else:
        raise ValueError("feat_df must contain 'importance_abs' or 'importance' column.")

    ids = plot_df["feature"].astype(str).values
    if gene_mapper is None:
        labels = ids
        ylab = "Feature"
    else:
        labels = [gene_mapper(gid) or gid for gid in ids]
        ylab = "Gene"

    if figsize is None:
        figsize = (10, max(4, 0.35 * top_n))

    plt.figure(figsize=figsize)
    plt.barh(labels, values)
    plt.title(title or default_title)
    plt.xlabel("Score")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Inference utilities
# -----------------------------
def align_features(X_df: pd.DataFrame, input_feature_cols: List[str]) -> pd.DataFrame:
    """Align DataFrame to FULL training input gene list (order + fill missing with 0)."""
    return X_df.reindex(columns=input_feature_cols, fill_value=0.0).astype(np.float32)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_tuple_klist(klist: Tuple[int, ...] | int) -> Tuple[int, ...]:
    if isinstance(klist, int):
        return (int(klist),)
    return tuple(klist)


# -----------------------------
# Correlation filter (stateful, safe for CV + inference)
# -----------------------------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Drop highly correlated features (absolute Pearson corr > threshold).
    Stores keep-mask during fit so transform is consistent.

    Note: computes full correlation matrix on dense data -> use AFTER SelectKBest.
    """
    def __init__(self, threshold: float = 0.95):
        self.threshold = float(threshold)

    def fit(self, X, y=None):
        X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)

        corr = np.corrcoef(X_dense, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        n = corr.shape[0]
        to_drop = np.zeros(n, dtype=bool)

        # Only compare against features that are NOT dropped so far
        for j in range(1, n):
            kept = np.where(~to_drop[:j])[0]
            if kept.size == 0:
                continue
            if np.any(np.abs(corr[kept, j]) > self.threshold):
                to_drop[j] = True

        self.keep_mask_ = ~to_drop
        self.n_features_in_ = n
        return self

    def transform(self, X):
        if not hasattr(self, "keep_mask_"):
            raise RuntimeError("CorrelationFilter is not fitted yet.")
        idx = np.where(self.keep_mask_)[0]
        return X[:, idx]

    def get_support(self):
        return self.keep_mask_


# -----------------------------
# Core pipeline pieces
# -----------------------------
def _log2p_preserve_sparse(A):
    if sparse.issparse(A):
        A = A.copy()
        A.data = np.log2(A.data + 1.0)
        return A
    return np.log2(A + 1.0)


def make_preprocess(cfg: ExperimentConfig) -> Pipeline:
    k_list = _as_tuple_klist(cfg.K_FEATURE_LIST)

    steps = [
        ("log2p", FunctionTransformer(_log2p_preserve_sparse, validate=False)),
        ("var", VarianceThreshold(threshold=cfg.VAR_THRESHOLD)),
        ("scale", MaxAbsScaler()),
        ("kbest", SelectKBest(score_func=f_classif, k=k_list[0])),
    ]
    if cfg.USE_CORR_FILTER:
        steps.append(("corr", CorrelationFilter(threshold=cfg.CORR_THRESHOLD)))

    return Pipeline(steps=steps)


def load_data(cfg: ExperimentConfig):
    df = pd.read_csv(cfg.MATRIX_PATH, sep="\t")
    meta = pd.read_csv(cfg.META_PATH)

    if not {"File Name", "label"}.issubset(df.columns):
        raise ValueError("Matrix file must include columns: 'File Name' and 'label'.")
    if not {"File Name", "Case ID"}.issubset(meta.columns):
        raise ValueError("Metadata file must include columns: 'File Name' and 'Case ID'.")

    meta_cols = ["File Name", "label"]
    input_feature_cols = [c for c in df.columns if c not in meta_cols]

    X_df = (
        df[input_feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )

    y_str = df["label"].astype(str).str.strip()
    y = y_str.map(cfg.LABEL_MAP)

    if y.isna().any():
        bad = sorted(y_str[y.isna()].unique())
        raise ValueError(f"Unexpected labels. Allowed {sorted(cfg.LABEL_MAP.keys())}. Found: {bad}")
    y = y.astype(int).values

    case_map = meta[["File Name", "Case ID"]].drop_duplicates()
    join_df = df[["File Name"]].merge(case_map, on="File Name", how="left")
    if join_df["Case ID"].isna().any():
        miss = join_df.loc[join_df["Case ID"].isna(), "File Name"].tolist()
        raise ValueError(f"Missing Case ID for {len(miss)} samples. Examples: {miss[:5]} ...")
    groups = join_df["Case ID"].astype(str).values

    zero_frac = float((X_df.values == 0).mean())
    X = sparse.csr_matrix(X_df.values) if zero_frac >= 0.5 else X_df.values

    return X, y, groups, input_feature_cols, zero_frac


def make_model_and_grid(cfg: ExperimentConfig, scale_pos_weight: float):
    k_list = _as_tuple_klist(cfg.K_FEATURE_LIST)
    common_pre: Dict[str, List[Any]] = {"pre__kbest__k": list(k_list)}

    if cfg.MODEL == "logreg_l2":
        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_STATE_MODEL,
        )
        grid = {**common_pre, "clf__C": [0.1, 1, 3]}

    elif cfg.MODEL == "linear_svm":
        model = LinearSVC(
            class_weight="balanced",
            max_iter=2000,
            random_state=cfg.RANDOM_STATE_MODEL,
        )
        grid = {**common_pre, "clf__C": [0.1, 1, 3]}

    elif cfg.MODEL == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=cfg.N_JOBS,
            class_weight="balanced_subsample",
            random_state=cfg.RANDOM_STATE_MODEL,
        )
        grid = {
            **common_pre,
            "clf__n_estimators": [300, 600, 1000],
            "clf__max_depth": [None, 10, 20, 50],
        }

    elif cfg.MODEL == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not available. Install it or choose another MODEL.")

        model = XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            tree_method=cfg.XGB_TREE_METHOD,
            n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_STATE_MODEL,
        )
        grid = {
            **common_pre,
            "clf__n_estimators": [300, 600, 1000],
            "clf__max_depth": [3, 4, 6],
            #"clf__learning_rate": [0.03, 0.05],
        }

    else:
        raise ValueError("MODEL must be one of: logreg_l2 | linear_svm | rf | xgb")

    return model, grid


def outer_holdout_split(X, y: np.ndarray, groups: np.ndarray, cfg: ExperimentConfig):
    outer = StratifiedGroupKFold(
        n_splits=cfg.OUTER_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE_OUTER
    )
    splits = list(outer.split(X, y, groups=groups))
    k = cfg.OUTER_FOLD_INDEX_AS_TEST
    if k < 0 or k >= len(splits):
        raise ValueError(f"OUTER_FOLD_INDEX_AS_TEST={k} out of range (0..{len(splits)-1}).")
    return splits[k]


def _get_scores_for_best_config(gs: GridSearchCV) -> List[float]:
    best_i = int(gs.best_index_)
    split_cols = [c for c in gs.cv_results_.keys() if c.startswith("split") and c.endswith("_test_score")]
    split_cols = sorted(split_cols, key=lambda s: int(s.split("_")[0].replace("split", "")))
    return [float(gs.cv_results_[c][best_i]) for c in split_cols]


def _get_y_score(best_pipe: Pipeline, X_test):
    if hasattr(best_pipe, "predict_proba"):
        return best_pipe.predict_proba(X_test)[:, 1]
    if hasattr(best_pipe, "decision_function"):
        return best_pipe.decision_function(X_test)
    return best_pipe.predict(X_test).astype(float)


def extract_selected_feature_names(best_pipe: Pipeline, input_feature_names: List[str]) -> List[str]:
    pre = best_pipe.named_steps["pre"]
    var = pre.named_steps["var"]
    kbest = pre.named_steps["kbest"]

    names_after_var = np.array(input_feature_names)[var.get_support()]
    names_after_kbest = names_after_var[kbest.get_support()]

    corr = pre.named_steps.get("corr", None)
    if corr is not None and hasattr(corr, "get_support"):
        names_after_kbest = names_after_kbest[corr.get_support()]

    return names_after_kbest.astype(str).tolist()


def feature_importance_dataframe(clf, selected_names: List[str], positive_class_name: str = "IDC") -> pd.DataFrame:
    selected_arr = np.array(selected_names, dtype=str)

    if hasattr(clf, "coef_"):
        coef = np.ravel(clf.coef_)
        return (
            pd.DataFrame(
                {
                    "feature": selected_arr,
                    "weight": coef,
                    "importance_abs": np.abs(coef),
                    "direction": np.where(
                        coef >= 0,
                        f"↑ toward {positive_class_name} (1)",
                        f"↓ toward {positive_class_name} (1)",
                    ),
                }
            )
            .sort_values("importance_abs", ascending=False)
        )

    if hasattr(clf, "feature_importances_"):
        imp = np.ravel(clf.feature_importances_)
        return (
            pd.DataFrame({"feature": selected_arr, "importance": imp})
            .sort_values("importance", ascending=False)
        )

    if hasattr(clf, "get_booster"):
        imp = np.array(clf.feature_importances_)
        return (
            pd.DataFrame({"feature": selected_arr, "importance": imp})
            .sort_values("importance", ascending=False)
        )

    raise RuntimeError("Cannot extract feature importance for this classifier.")


def run_brca_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    X, y, groups, input_feature_cols, zero_frac = load_data(cfg)

    overview = {
        "n_samples": int(y.shape[0]),
        "n_features": int(len(input_feature_cols)),
        "label_distribution": {0: int((y == 0).sum()), 1: int((y == 1).sum())},
        "zero_fraction": float(zero_frac),
        "n_groups": int(len(np.unique(groups))),
        "using_sparse": bool(sparse.issparse(X)),
    }

    train_idx, test_idx = outer_holdout_split(X, y, groups, cfg)
    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test, g_test = X[test_idx], y[test_idx], groups[test_idx]

    # For XGB only (computed on TRAIN)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg) / max(1.0, float(pos))

    preprocess = make_preprocess(cfg)
    model, grid = make_model_and_grid(cfg, scale_pos_weight=spw)
    pipe = Pipeline([("pre", preprocess), ("clf", model)])

    inner = StratifiedGroupKFold(
        n_splits=cfg.INNER_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE_INNER
    )

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring=cfg.SCORING,
        cv=inner,
        n_jobs=cfg.N_JOBS,
        refit=True,
        verbose=0,
        error_score=np.nan,
    )
    gs.fit(X_train, y_train, groups=g_train)

    best_pipe = gs.best_estimator_
    fold_scores = _get_scores_for_best_config(gs)

    y_score = _get_y_score(best_pipe, X_test)
    y_pred = best_pipe.predict(X_test)

    test_metrics = {
        "ROC-AUC": float(roc_auc_score(y_test, y_score)),
        "PR-AUC": float(average_precision_score(y_test, y_score)),
        "F1": float(f1_score(y_test, y_pred)),
        "ACC": float(accuracy_score(y_test, y_pred)),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    report = classification_report(
        y_test, y_pred, digits=4, target_names=list(cfg.CLASS_NAMES)
    )

    selected_feature_cols = extract_selected_feature_names(best_pipe, input_feature_cols)
    clf = best_pipe.named_steps["clf"]
    # positive class display name (e.g., "IDC" from "IDC (1)")
    pos_name = cfg.CLASS_NAMES[1].split()[0]
    feat_df = feature_importance_dataframe(clf, selected_feature_cols, positive_class_name=pos_name)

    return {
        "config": cfg,
        "overview": overview,
        "input_feature_cols": input_feature_cols,
        "selected_feature_cols": selected_feature_cols,
        "outer_split": {
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "train_groups": int(len(np.unique(g_train))),
            "test_groups": int(len(np.unique(g_test))),
        },
        "inner_cv": {
            "best_score": float(gs.best_score_),
            "best_params": gs.best_params_,
            "best_fold_scores": fold_scores,
        },
        "holdout": {
            "metrics": test_metrics,
            "confusion_matrix": cm,
            "confusion_matrix_row_norm": cm_norm,
            "classification_report": report,
        },
        "feature_importance": feat_df,
        "best_estimator": best_pipe,
        "grid_search": gs,
    }


def save_inference_bundle(results: Dict[str, Any], out_dir: str, prefix: str = "brca_idc_ilc") -> Dict[str, str]:
    """
    Save:
      - {prefix}_model.joblib
      - {prefix}_input_feature_cols.json  (FULL, required for inference)
      - {prefix}_selected_feature_cols.json (post Var+KBest+Corr)
      - {prefix}_metadata.json
    """
    import joblib

    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, f"{prefix}_model.joblib")
    input_cols_path = os.path.join(out_dir, f"{prefix}_input_feature_cols.json")
    selected_cols_path = os.path.join(out_dir, f"{prefix}_selected_feature_cols.json")
    meta_path = os.path.join(out_dir, f"{prefix}_metadata.json")

    joblib.dump(results["best_estimator"], model_path)

    with open(input_cols_path, "w", encoding="utf-8") as f:
        json.dump(list(results["input_feature_cols"]), f, ensure_ascii=False)

    with open(selected_cols_path, "w", encoding="utf-8") as f:
        json.dump(list(results["selected_feature_cols"]), f, ensure_ascii=False)

    cfg: ExperimentConfig = results["config"]
    metadata = {
        "created_utc": _utc_now_iso(),
        "label_map": cfg.LABEL_MAP,
        "positive_label": 1,
        "class_names": list(cfg.CLASS_NAMES),
        "config": asdict(cfg),
        "overview": results.get("overview", {}),
        "best_params": results.get("inner_cv", {}).get("best_params", {}),
        "best_score_cv": results.get("inner_cv", {}).get("best_score", None),
        "holdout_metrics": results.get("holdout", {}).get("metrics", {}),
        "selected_k_final": len(results["selected_feature_cols"]),
        "artifacts": {
            "model": os.path.basename(model_path),
            "input_feature_cols": os.path.basename(input_cols_path),
            "selected_feature_cols": os.path.basename(selected_cols_path),
            "metadata": os.path.basename(meta_path),
        },
        "versions": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": getattr(xgboost, "__version__", None) if xgboost is not None else None,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "model": model_path,
        "input_feature_cols": input_cols_path,
        "selected_feature_cols": selected_cols_path,
        "metadata": meta_path,
    }


def load_inference_bundle(out_dir: str, prefix: str = "brca_idc_ilc"):
    import joblib

    model_path = os.path.join(out_dir, f"{prefix}_model.joblib")
    input_cols_path = os.path.join(out_dir, f"{prefix}_input_feature_cols.json")
    selected_cols_path = os.path.join(out_dir, f"{prefix}_selected_feature_cols.json")
    meta_path = os.path.join(out_dir, f"{prefix}_metadata.json")

    model = joblib.load(model_path)

    with open(input_cols_path, "r", encoding="utf-8") as f:
        input_cols = json.load(f)
    with open(selected_cols_path, "r", encoding="utf-8") as f:
        selected_cols = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, input_cols, selected_cols, metadata


def print_summary(results: Dict[str, Any], top_n: int = 10) -> None:
    ov = results["overview"]
    osplit = results["outer_split"]
    icv = results["inner_cv"]
    hold = results["holdout"]
    feat = results["feature_importance"]
    cfg: ExperimentConfig = results["config"]

    print("===== OVERVIEW =====")
    print(f"Số mẫu: {ov['n_samples']:,} | Số gene/features: {ov['n_features']:,}")
    print(f"Phân bố nhãn: {ov['label_distribution']} (0={cfg.CLASS_NAMES[0]}, 1={cfg.CLASS_NAMES[1]})")
    print(f"Tỉ lệ số 0 toàn ma trận: {ov['zero_fraction']:.3f}")
    print(f"Số Case ID duy nhất (nhóm): {ov['n_groups']:,}")
    print(f"Dùng sparse: {ov['using_sparse']}")

    print("\n===== OUTER HOLD-OUT =====")
    print(
        f"train={osplit['train_size']}, test={osplit['test_size']} | "
        f"train groups={osplit['train_groups']}, test groups={osplit['test_groups']}"
    )

    print("\n===== INNER CV (GridSearch) =====")
    print(f"best {cfg.SCORING}={icv['best_score']:.4f}")
    print(f"best_params={icv['best_params']}")
    print("Fold scores:", ", ".join(f"{s:.4f}" for s in icv["best_fold_scores"]))

    print("\n===== FINAL TEST (HOLD-OUT) =====")
    for k, v in hold["metrics"].items():
        print(f"{k:7s}: {v:.4f}")

    cm = hold["confusion_matrix"]
    cmn = hold["confusion_matrix_row_norm"]
    print("\n--- Confusion Matrix (counts) ---")
    print(pd.DataFrame(cm, index=[f"True {cfg.CLASS_NAMES[0]}", f"True {cfg.CLASS_NAMES[1]}"],
                       columns=[f"Pred {cfg.CLASS_NAMES[0]}", f"Pred {cfg.CLASS_NAMES[1]}"]))
    print("\n--- Confusion Matrix (row-normalized) ---")
    print(pd.DataFrame(np.round(cmn, 4), index=[f"True {cfg.CLASS_NAMES[0]}", f"True {cfg.CLASS_NAMES[1]}"],
                       columns=[f"Pred {cfg.CLASS_NAMES[0]}", f"Pred {cfg.CLASS_NAMES[1]}"]))

    print("\n--- Classification report ---")
    print(hold["classification_report"])

    print(f"\n--- Top {top_n} features ---")
    print(feat.head(top_n).to_string(index=False))
