"""
Streamlit demo for survival risk scoring on a single sample.
- Upload a gene expression file (TSV/CSV) with gene rows.
- Provide the clinical features expected by the trained pipelines.
- Choose one of the pre-trained artifacts: CoxPH or Coxnet.

This file is additive and does not modify the existing clinical dashboard app.
"""

import io
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

from utils_clinical import read_star_htseq_tsv

st.set_page_config(page_title="BRCA Survival Demo", layout="wide")


def _ensure_remainder_cols_list():
    """Backport sklearn's private _RemainderColsList for older pickles."""
    from sklearn.compose import _column_transformer

    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _column_transformer._RemainderColsList = _RemainderColsList


def _patch_column_transformer(ct):
    """Add missing attrs for older pickled ColumnTransformer to work on newer sklearn."""
    try:
        from sklearn.compose import ColumnTransformer
    except Exception:
        return ct

    if not isinstance(ct, ColumnTransformer):
        return ct

    # Newer sklearn expects this attr after fit; older pickles may miss it
    if not hasattr(ct, "_name_to_fitted_passthrough"):
        ct._name_to_fitted_passthrough = {}

    return ct

# ------------------------------
# Caching helpers
# ------------------------------

@st.cache_resource(show_spinner=False)
def load_artifact(path: Path) -> Dict:
    _ensure_remainder_cols_list()
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pre_clin" in obj:
        obj["pre_clin"] = _patch_column_transformer(obj["pre_clin"])
    return obj


def _get_clinical_schema(pre_clin) -> Tuple[List[str], List[str], List[str]]:
    """Extract the raw column names expected by the fitted ColumnTransformer."""
    cat_cols = []
    num_cols = []
    bin_cols = []
    for name, transformer, cols in pre_clin.transformers:
        if name == "cat":
            cat_cols = list(cols)
        elif name == "num":
            num_cols = list(cols)
        elif name == "bin":
            bin_cols = list(cols)
    return cat_cols, num_cols, bin_cols


def _get_cat_options(pre_clin, cat_cols: List[str]) -> Dict[str, List[str]]:
    """Return known categories per categorical column if available."""
    opts: Dict[str, List[str]] = {}
    enc = pre_clin.named_transformers_.get("cat")
    if enc is None:
        return opts
    cats = getattr(enc, "categories_", [])
    for col, arr in zip(cat_cols, cats):
        opts[col] = [str(x) for x in arr.tolist()] if arr is not None else []
    return opts

def _log2p(A):
    # A có thể là numpy dense hoặc sparse
    if sparse.issparse(A):
        A = A.copy()
        A.data = np.log2(A.data + 1.0)
        return A
    return np.log2(A + 1.0)


def _nearest_probs(surv_df: pd.DataFrame, days: List[int]) -> Dict[int, float]:
    """Pick nearest survival prob for each day from a survival function DF."""
    out: Dict[int, float] = {}
    if surv_df.empty:
        return out
    for d in days:
        idx = surv_df.index.get_indexer([d], method="nearest")
        if idx.size and idx[0] >= 0:
            out[d] = float(surv_df.iloc[idx[0], 0])
    return out


def _sksurv_step_to_df(step_fn) -> pd.DataFrame:
    """Convert sksurv StepFunction to DataFrame for plotting."""
    xs = np.asarray(step_fn.x, dtype=float)
    ys = np.asarray(step_fn.y, dtype=float)
    return pd.DataFrame({"survival": ys}, index=xs)


def _median_from_step(step_fn):
    ys = np.asarray(step_fn.y, dtype=float)
    xs = np.asarray(step_fn.x, dtype=float)
    mask = ys <= 0.5
    if not mask.any():
        return None
    idx = np.argmax(mask)
    return float(xs[idx])


def _median_from_surv_df(surv_df: pd.DataFrame):
    if surv_df is None or surv_df.empty:
        return None
    ys = surv_df.iloc[:, 0].to_numpy(dtype=float)
    xs = surv_df.index.to_numpy(dtype=float)
    mask = ys <= 0.5
    if not mask.any():
        return None
    idx = np.argmax(mask)
    return float(xs[idx])


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode("utf-8")


# ------------------------------
# UI
# ------------------------------

st.title("BRCA Survival Risk — Single Sample Demo")
st.write(
    "Upload a gene-expression file, fill clinical fields, then score using your saved CoxPH/Coxnet artifacts."
)

artifact_choice = st.radio(
    "Select model artifact",
    options=[
        ("CoxPH (lifelines)", "cph_streamlit_artifact.joblib"),
        ("Coxnet (sksurv)", "coxnet_streamlit_artifact.joblib"),
    ],
    format_func=lambda x: x[0],
    horizontal=True,
)
artifact_path = Path(artifact_choice[1])
if not artifact_path.exists():
    st.error(f"Artifact not found: {artifact_path}")
    st.stop()

artifact = load_artifact(artifact_path)
model = artifact["model"]

gene_cols: List[str] = artifact.get("gene_cols", [])
clin_feature_names: List[str] = artifact.get("clin_feature_names", [])
feature_names: List[str] = artifact.get("feature_names", gene_cols + clin_feature_names)
selected_gene_idx = artifact.get("selected_gene_idx")
if selected_gene_idx is not None:
    selected_gene_idx = np.asarray(selected_gene_idx, dtype=int)

gene_pre = artifact["gene_pre"]
pre_clin = artifact["pre_clin"]
t1 = artifact.get("t1")
t2 = artifact.get("t2")
requires_dense = bool(artifact.get("requires_dense_X", False))

cat_cols, num_cols, bin_cols = _get_clinical_schema(pre_clin)
cat_options = _get_cat_options(pre_clin, cat_cols)

st.markdown("### 1) Upload gene counts")
up = st.file_uploader("Gene expression file (TSV/CSV with gene rows)", type=["tsv", "csv", "txt"])

st.markdown("### 2) Clinical features")
clin_inputs: Dict[str, object] = {}

col1, col2 = st.columns(2)
with col1:
    for c in cat_cols:
        choices = ["(blank)"] + cat_options.get(c, [])
        val = st.selectbox(c, choices, index=0)
        clin_inputs[c] = "" if val == "(blank)" else val
with col2:
    for c in num_cols:
        clin_inputs[c] = st.number_input(c, value=0.0, step=1.0, format="%0.3f")
    for c in bin_cols:
        val = st.selectbox(c, ["No", "Yes"], index=0)
        clin_inputs[c] = 1 if val == "Yes" else 0

st.markdown("### 3) Predict")
run = st.button("Compute risk")

if run:
    if up is None:
        st.error("Please upload a gene expression file first.")
        st.stop()

    # --- Gene preprocessing ---
    try:
        gene_df = read_star_htseq_tsv(up)
    except Exception as exc:
        st.error(f"Could not read gene file: {exc}")
        st.stop()

    if selected_gene_idx is not None:
        base_genes = np.asarray(gene_cols)[selected_gene_idx].tolist()
    else:
        base_genes = gene_cols

    # Align to expected gene columns
    gene_aligned = gene_df.reindex(columns=base_genes, fill_value=0.0)
    Xg = gene_pre.transform(gene_aligned.to_numpy(dtype=float))
    if sparse.issparse(Xg):
        Xg = sparse.csr_matrix(Xg)

    # --- Clinical preprocessing ---
    clin_df = pd.DataFrame([clin_inputs], columns=cat_cols + num_cols + bin_cols)
    Xc = pre_clin.transform(clin_df)
    if sparse.issparse(Xc):
        Xc = sparse.csr_matrix(Xc)

    # --- Combine ---
    if sparse.issparse(Xg) or sparse.issparse(Xc):
        X = sparse.hstack([Xg, Xc]).tocsr()
    else:
        X = np.hstack([Xg, Xc])

    # --- Prepare for model type ---
    feature_order = (
        artifact.get("feature_names_cph")
        or artifact.get("feature_names")
        or (gene_cols + clin_feature_names)
    )

    df_for_pred = None
    sksurv_surv_df = None
    sksurv_step = None
    sksurv_err = None

    if hasattr(model, "predict_partial_hazard"):
        # lifelines CoxPH expects dense DataFrame with named columns
        if sparse.issparse(X):
            X_arr = X.toarray()
        else:
            X_arr = np.asarray(X)
        df_for_pred = pd.DataFrame(X_arr, columns=feature_order)
        risk = float(model.predict_partial_hazard(df_for_pred).to_numpy().ravel()[0])
    else:
        if requires_dense and sparse.issparse(X):
            X = X.toarray()
        risk = float(np.asarray(model.predict(X)).ravel()[0])
        # If sksurv supports survival_function
        if hasattr(model, "predict_survival_function"):
            try:
                surv_list = model.predict_survival_function(X if not sparse.issparse(X) else X.toarray())
                if surv_list:
                    sksurv_step = surv_list[0]
                    sksurv_surv_df = _sksurv_step_to_df(sksurv_step)
                else:
                    sksurv_err = "No baseline survival stored in artifact (empty survival list)."
            except Exception as exc:
                sksurv_surv_df = None
                sksurv_err = str(exc)
        # Fallback: Coxnet with fit_baseline_model=False -> try cumulative_hazard_function
        if sksurv_surv_df is None and hasattr(model, "predict_cumulative_hazard_function"):
            try:
                chf_list = model.predict_cumulative_hazard_function(X if not sparse.issparse(X) else X.toarray())
                if chf_list:
                    step = chf_list[0]
                    xs = np.asarray(step.x, dtype=float)
                    ys = np.asarray(step.y, dtype=float)
                    surv = np.exp(-ys)
                    sksurv_step = step
                    sksurv_surv_df = pd.DataFrame({"survival": surv}, index=xs)
                else:
                    sksurv_err = sksurv_err or "No cumulative hazard stored in artifact."
            except Exception as exc:
                sksurv_err = sksurv_err or str(exc)

    # Risk group
    if t1 is not None and t2 is not None:
        group = "LOW" if risk <= t1 else ("MED" if risk <= t2 else "HIGH")
    else:
        group = "N/A"

    st.success("Prediction ready")
    st.metric("Risk score", f"{risk:0.4f}")
    st.metric("Risk group (t1/t2 from training)", group)

    # --- Survival curve & milestones (lifelines CoxPH only) ---
    if df_for_pred is not None and hasattr(model, "predict_survival_function"):
        surv_func = model.predict_survival_function(df_for_pred)
        surv_df = pd.DataFrame({"survival": surv_func.iloc[:, 0].values}, index=surv_func.index)

        st.markdown("#### Survival curve (model-based)")
        st.line_chart(surv_df)

        st.download_button(
            "Download survival curve (CSV)",
            data=_df_to_csv_bytes(surv_df),
            file_name="survival_curve_lifelines.csv",
            mime="text/csv",
        )

        # Median survival
        if hasattr(model, "predict_median"):
            try:
                median_survival = model.predict_median(df_for_pred)
                median_val = float(np.asarray(median_survival).ravel()[0])
                if np.isfinite(median_val):
                    st.caption(f"Median survival time: {median_val:.0f} days")
            except Exception:
                pass

        # Milestone probs
        milestones = [365, 365 * 3, 365 * 5]
        probs = _nearest_probs(surv_df, milestones)
        if probs:
            st.markdown("##### Survival probability milestones")
            for d in milestones:
                if d in probs:
                    st.write(f"{d} days (~{d/365:.0f}y): {probs[d]*100:.1f}%")

    # sksurv Coxnet survival plot
    if sksurv_surv_df is not None:
        st.markdown("#### Survival curve (sksurv)")
        st.line_chart(sksurv_surv_df)

        st.download_button(
            "Download survival curve (CSV, sksurv)",
            data=_df_to_csv_bytes(sksurv_surv_df),
            file_name="survival_curve_sksurv.csv",
            mime="text/csv",
        )

        milestones = [365, 365 * 3, 365 * 5]
        probs = _nearest_probs(sksurv_surv_df, milestones)
        if probs:
            st.markdown("##### Survival probability milestones")
            for d in milestones:
                if d in probs:
                    st.write(f"{d} days (~{d/365:.0f}y): {probs[d]*100:.1f}%")

        med = _median_from_surv_df(sksurv_surv_df)
        if med is not None and np.isfinite(med):
            st.caption(f"Median survival time (sksurv): {med:.0f} days")
    elif sksurv_err:
        st.info(f"Survival curve not available for Coxnet: {sksurv_err}")
        if "fit_baseline_model" in sksurv_err.lower() or "baseline" in sksurv_err.lower():
            st.warning(
                "Artifact was trained without baseline hazards. To enable survival curves, "
                "retrain Coxnet with fit_baseline_model=True and resave the artifact."
            )

    st.caption(
        "Gene columns aligned to training set: "
        f"{len(base_genes)} genes; clinical cols: {cat_cols + num_cols + bin_cols}"
    )

st.markdown("---")
st.write(
    "Tips: the gene file should contain TPM/count columns per gene. Clinical inputs are kept simple; "
    "missing values can remain blank (categorical) or 0 (numeric/binary)."
)
