# app.py
import io
import sys
import types
import re

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter


# ============================================================
# 0) COMPAT: Inject module "main" so joblib can unpickle
#    artifacts saved from module name "main".
# ============================================================
TIME_COL_DEFAULT = "OS_time_days"
EVENT_COL_DEFAULT = "OS_event"


def _extract_time_event(y):
    if hasattr(y, "dtype") and y.dtype.names is not None:
        if "time" in y.dtype.names and "event" in y.dtype.names:
            return y["time"].astype(float), y["event"].astype(bool)
    raise ValueError("y phải là Surv.from_arrays(event=..., time=...).")


def cindex_survival_score(X, y):
    time, event = _extract_time_event(y)
    if sparse.issparse(X):
        X = X.tocsc()

    n_features = X.shape[1]
    scores = np.zeros(n_features, dtype=float)
    pvals = np.full(n_features, np.nan, dtype=float)

    for j in range(n_features):
        if sparse.issparse(X):
            xj = X.getcol(j).toarray().ravel()
        else:
            xj = np.asarray(X[:, j]).ravel()

        xj = np.nan_to_num(xj, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(xj) == 0:
            scores[j] = 0.0
            continue

        ci1 = concordance_index_censored(event, time, xj)[0]
        ci2 = concordance_index_censored(event, time, -xj)[0]
        scores[j] = max(0.0, max(ci1, ci2) - 0.5)

    return scores, pvals


def _log2p(A):
    if sparse.issparse(A):
        B = A.copy()
        B.data = np.log2(B.data + 1.0)
        return B
    return np.log2(A + 1.0)


class TopKVariance(BaseEstimator, TransformerMixin):
    def __init__(self, k=10000):
        self.k = int(k)
        self.idx_ = None

    def fit(self, X, y=None):
        if sparse.issparse(X):
            Xc = X.tocsc()
            mean = np.array(Xc.mean(axis=0)).ravel()
            mean2 = np.array(Xc.power(2).mean(axis=0)).ravel()
            var = mean2 - mean**2
        else:
            var = np.var(np.asarray(X), axis=0)

        k = min(self.k, var.shape[0])
        self.idx_ = np.argsort(var)[::-1][:k]
        return self

    def transform(self, X):
        if self.idx_ is None:
            raise RuntimeError("TopKVariance chưa fit.")
        return X[:, self.idx_]


class AddOrdinalStageTNM(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        def _make(col, mapping, num_col, unk_col):
            if col not in df.columns:
                return
            s = df[col].astype("string").str.strip().str.lower()
            num = s.map(mapping)
            df[num_col] = num.fillna(-1).astype(int)
            df[unk_col] = num.isna().astype(int)

        _make(
            "stage_group",
            {"i": 1, "ii": 2, "iii": 3, "iv": 4},
            "stage_num",
            "stage_unk",
        )
        _make(
            "ajcc_pathologic_t", {"t1": 1, "t2": 2, "t3": 3, "t4": 4}, "t_num", "t_unk"
        )
        _make(
            "ajcc_pathologic_n", {"n0": 0, "n1": 1, "n2": 2, "n3": 3}, "n_num", "n_unk"
        )
        _make("ajcc_pathologic_m", {"m0": 0, "m1": 1}, "m_num", "m_unk")
        return df


class DenseFloat64(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64, order="C")


class CoxPHSklearn(BaseEstimator):
    def __init__(self, penalizer=0.1, l1_ratio=0.0):
        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.model_ = None
        self.feature_names_ = None

    @staticmethod
    def _to_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64, order="C")

    def fit(self, X, y):
        Xd = self._to_dense(X)
        t = y["time"].astype(float)
        e = y["event"].astype(bool).astype(int)

        p = Xd.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(p)]
        df_fit = pd.DataFrame(Xd, columns=self.feature_names_)
        df_fit[TIME_COL_DEFAULT] = t
        df_fit[EVENT_COL_DEFAULT] = e

        self.model_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model_.fit(
            df_fit, duration_col=TIME_COL_DEFAULT, event_col=EVENT_COL_DEFAULT
        )
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model chưa fit.")
        Xd = self._to_dense(X)
        dfX = pd.DataFrame(Xd, columns=self.feature_names_)
        return self.model_.predict_partial_hazard(dfX).to_numpy().ravel()


_main = types.ModuleType("main")
_main._log2p = _log2p
_main.TopKVariance = TopKVariance
_main.AddOrdinalStageTNM = AddOrdinalStageTNM
_main.DenseFloat64 = DenseFloat64
_main.cindex_survival_score = cindex_survival_score
_main.CoxPHSklearn = CoxPHSklearn
sys.modules["main"] = _main
# ============================================================


# ============================================================
# 1) Clinical cleaning code (your logic)
# ============================================================
NA_LIKE = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not reported",
    "not available",
    "unknown",
    "unspecified",
    "--",
    "-",
    ".",
    "missing",
}


def _normalize_colname(c: str) -> str:
    c = str(c).strip()
    c = c.replace(" ", "_")
    c = c.replace("-", "_")
    return c


def _clean_text_series(s: pd.Series) -> pd.Series:
    s2 = s.astype("string").str.strip()
    s2 = s2.replace({x: pd.NA for x in NA_LIKE})
    s2 = s2.str.lower()
    return s2


def clean_clinical(
    df: pd.DataFrame,
    target_time: str = "OS_time_days",
    target_event: str = "OS_event",
    id_col: str = "submitter_id",
    file_col: str = "File Name",
    drop_missing_col_threshold: float = 0.95,
    eps_time: float = 0.5,
):
    df = df.copy()

    # 1) normalize col names
    df.columns = [_normalize_colname(c) for c in df.columns]

    # normalize file_col/id/targets names (but inference may not have them)
    if file_col in df.columns:
        file_col_norm = file_col
    else:
        file_col_norm = _normalize_colname(file_col)
    file_col = file_col_norm
    id_col = _normalize_colname(id_col)
    target_time = _normalize_colname(target_time)
    target_event = _normalize_colname(target_event)

    # 2) clean text cols
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        df[c] = _clean_text_series(df[c])

    # 3) coerce targets if exist
    if target_time in df.columns:
        df[target_time] = pd.to_numeric(df[target_time], errors="coerce")

    if target_event in df.columns:
        raw = df[target_event].astype("string").str.strip().str.lower()
        mapped = raw.replace(
            {
                "true": "1",
                "false": "0",
                "yes": "1",
                "no": "0",
                "alive": "0",
                "dead": "1",
            }
        )
        df[target_event] = pd.to_numeric(mapped, errors="coerce")

    # 4) filter bad outcomes only if those cols exist
    before_rows = len(df)
    if target_time in df.columns:
        df = df[df[target_time].notna()].copy()
        df = df[df[target_time] >= 0].copy()
        df.loc[df[target_time] == 0, target_time] = eps_time

    if target_event in df.columns:
        df = df[df[target_event].isin([0, 1])].copy()
        df[target_event] = df[target_event].astype(int)

    # 5) dedup by id if present
    if id_col in df.columns:
        row_missing = df.isna().mean(axis=1)
        df["_row_missing_rate"] = row_missing

        sort_cols = [id_col, "_row_missing_rate"]
        ascending = [True, True]
        if file_col in df.columns:
            sort_cols.append(file_col)
            ascending.append(True)

        df = df.sort_values(sort_cols, ascending=ascending)
        df = df.groupby(id_col, as_index=False).first()
        df = df.drop(columns=["_row_missing_rate"], errors="ignore")

    after_rows = len(df)

    # 6) drop columns with too much missing (IMPORTANT: set threshold=1.0 for single-row inference)
    miss = df.isna().mean()
    drop_cols = miss[miss > drop_missing_col_threshold].index.tolist()
    protect = {id_col, target_time, target_event, file_col}
    drop_cols = [c for c in drop_cols if c not in protect]
    df = df.drop(columns=drop_cols)

    report = {
        "rows_before": int(before_rows),
        "rows_after": int(after_rows),
        "dropped_columns_missing_gt_threshold": drop_cols,
        "missing_rate_top10": miss.sort_values(ascending=False).head(10).to_dict(),
        "event_rate": (
            float(df[target_event].mean())
            if target_event in df.columns and len(df)
            else None
        ),
    }
    return df, report


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _nanize(s: pd.Series) -> pd.Series:
        s = s.astype("string").str.strip().str.lower()
        s = s.replace({"unknown": pd.NA, "not reported": pd.NA, "not available": pd.NA})
        return s

    if "race" in df.columns:
        s = _nanize(df["race"]).replace({"american indian or alaska native": "other"})
        df["race"] = s.fillna("unknown")

    for c in ["ER_status", "PR_status"]:
        if c in df.columns:
            df[c] = _nanize(df[c]).fillna("unknown")

    if "HER2_status" in df.columns:
        s = _nanize(df["HER2_status"]).replace(
            {
                "test value reported": "unknown",
                "copy number reported": "unknown",
            }
        )
        df["HER2_status"] = s.fillna("unknown")

    if "ajcc_pathologic_stage" in df.columns:
        s = _nanize(df["ajcc_pathologic_stage"]).replace({"stage x": pd.NA})

        def stage_group(x):
            if pd.isna(x):
                return "unknown"
            m = re.match(r"stage\s+([ivx]+)", x)
            if not m:
                return "unknown"
            g = m.group(1)
            return "unknown" if g == "x" else g.upper()

        def stage_sub(x):
            if pd.isna(x):
                return "unknown"
            m = re.match(r"stage\s+([ivx]+)([abc])$", x)
            if m and m.group(2) in ["a", "b", "c"]:
                return m.group(2)
            m2 = re.match(r"stage\s+([ivx]+)$", x)
            return "none" if m2 else "unknown"

        df["stage_group"] = s.map(stage_group)
        df["stage_sub"] = s.map(stage_sub)
        df = df.drop(columns=["ajcc_pathologic_stage"], errors="ignore")

    if "ajcc_pathologic_t" in df.columns:
        s0 = _nanize(df["ajcc_pathologic_t"]).replace({"tx": pd.NA})
        t_suffix = s0.str.extract(r"^t\d([a-z])$")[0]
        t_suffix = t_suffix.where(t_suffix.isin(["a", "b", "c", "d"]), pd.NA)
        t_base = s0.str.replace(r"^(t[0-9]).*$", r"\1", regex=True)

        df["ajcc_pathologic_t"] = t_base.fillna("unknown")
        df["t_suffix"] = t_suffix.fillna("none")
        df.loc[df["ajcc_pathologic_t"] == "unknown", "t_suffix"] = "unknown"

    if "ajcc_pathologic_n" in df.columns:
        s0 = _nanize(df["ajcc_pathologic_n"]).replace({"nx": pd.NA})

        n_is_mi = s0.str.fullmatch(r"n1mi", na=False).astype(int)
        n0_i_plus = s0.str.contains(r"^n0\s*\(i\+\)$", regex=True, na=False).astype(int)
        n0_mol_plus = s0.str.contains(r"^n0\s*\(mol\+\)$", regex=True, na=False).astype(
            int
        )

        n_suffix = s0.str.extract(r"^n\d([abc])$")[0]
        n_suffix = n_suffix.where(n_suffix.isin(["a", "b", "c"]), pd.NA)

        s = s0.str.replace(r"^n0.*$", "n0", regex=True)
        s = s.str.replace(r"^(n[0-9]).*$", r"\1", regex=True)
        df["n_base"] = s.fillna("unknown")

        df["n_suffix"] = n_suffix.fillna("none")
        df["n_is_mi"] = n_is_mi
        df["n0_i_plus"] = n0_i_plus
        df["n0_mol_plus"] = n0_mol_plus

        df["n_micrometastasis"] = (
            (df["n_is_mi"] == 1) | (df["n0_i_plus"] == 1) | (df["n0_mol_plus"] == 1)
        ).astype(int)

        unk = df["n_base"] == "unknown"
        df.loc[unk, ["n_is_mi", "n0_i_plus", "n0_mol_plus", "n_micrometastasis"]] = 0
        df.loc[unk, "n_suffix"] = "unknown"

        df = df.drop(columns=["ajcc_pathologic_n"], errors="ignore")
        df = df.rename(columns={"n_base": "ajcc_pathologic_n"})

    if "ajcc_pathologic_m" in df.columns:
        s = _nanize(df["ajcc_pathologic_m"]).replace({"mx": pd.NA})
        s = s.replace({"cm0 (i+)": "m0"})
        s = s.where(s.isin(["m0", "m1"]), pd.NA)
        df["ajcc_pathologic_m"] = s.fillna("unknown")

    if "laterality" in df.columns:
        df["laterality"] = _nanize(df["laterality"]).fillna("unknown")
        df.loc[df["laterality"] == "unknown", "laterality"] = "left"

    if "method_of_diagnosis" in df.columns:
        s = _nanize(df["method_of_diagnosis"])
        biopsy_set = {
            "core biopsy",
            "fine needle aspiration",
            "excisional biopsy",
            "incisional biopsy",
            "ultrasound guided biopsy",
            "biopsy",
        }
        s = s.apply(lambda x: "biopsy" if x in biopsy_set else x)
        s = s.replace({"surgical resection": "surgery"})
        df["method_of_diagnosis"] = s.fillna("unknown")

    if "menopause_status" in df.columns:
        df["menopause_status"] = _nanize(df["menopause_status"]).fillna("unknown")

    for c in ["ER_status", "PR_status"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"equivocal": "unknown"})
                .fillna("unknown")
            )

    if "race" in df.columns:
        df["race"] = (
            df["race"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"other": "unknown"})
            .fillna("unknown")
        )

    return df


def clean_numeric(df: pd.DataFrame, add_missing_flags: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["positive_ratio"], errors="ignore")

    cols = ["age_at_diagnosis_years", "lymph_nodes_positive", "lymph_nodes_tested"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "age_at_diagnosis_years" in df.columns:
        df.loc[
            (df["age_at_diagnosis_years"] < 0) | (df["age_at_diagnosis_years"] > 120),
            "age_at_diagnosis_years",
        ] = np.nan

    if "lymph_nodes_positive" in df.columns:
        df.loc[df["lymph_nodes_positive"] < 0, "lymph_nodes_positive"] = np.nan

    if "lymph_nodes_tested" in df.columns:
        df.loc[df["lymph_nodes_tested"] < 0, "lymph_nodes_tested"] = np.nan

    if "lymph_nodes_positive" in df.columns and "lymph_nodes_tested" in df.columns:
        bad = (
            df["lymph_nodes_positive"].notna()
            & df["lymph_nodes_tested"].notna()
            & (df["lymph_nodes_positive"] > df["lymph_nodes_tested"])
        )
        df.loc[bad, "lymph_nodes_tested"] = df.loc[bad, "lymph_nodes_positive"]

    if add_missing_flags:
        for c in cols:
            if c in df.columns:
                df[c + "_was_missing"] = df[c].isna().astype(int)

    for c in cols:
        if c in df.columns:
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(med)

    return df


# ============================================================
# 2) Gene quant TSV loader (gene_id + tpm_unstranded) -> 1-row wide
# ============================================================
def load_quant_tpm_geneid_to_wide(
    uploaded_file,
    file_name: str,
    gene_cols_expected: list[str],
    agg: str = "sum",
) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep="\t", comment="#", dtype=str)

    if "gene_id" not in df.columns:
        raise ValueError(f"TSV thiếu cột 'gene_id'. Columns: {list(df.columns)}")
    if "tpm_unstranded" not in df.columns:
        raise ValueError("TSV thiếu cột 'tpm_unstranded'.")

    df = df[~df["gene_id"].astype(str).str.startswith("N_")].copy()

    expected_has_version = any("." in g for g in gene_cols_expected)
    strip_version = not expected_has_version

    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    if strip_version:
        df["gene_id"] = df["gene_id"].str.replace(r"\.\d+$", "", regex=True)

    df = df[df["gene_id"].notna() & (df["gene_id"] != "")].copy()
    df["tpm_unstranded"] = pd.to_numeric(df["tpm_unstranded"], errors="coerce").fillna(
        0.0
    )

    if agg == "sum":
        df = df.groupby("gene_id", as_index=False)["tpm_unstranded"].sum()
    elif agg == "max":
        df = df.groupby("gene_id", as_index=False)["tpm_unstranded"].max()
    else:
        raise ValueError("agg must be 'sum' or 'max'")

    wide = df.set_index("gene_id")["tpm_unstranded"].to_frame().T
    wide.index = [0]
    wide.insert(0, "File Name", str(file_name))
    return wide


def align_to_expected_genes(
    df_one: pd.DataFrame, gene_cols_expected: list[str]
) -> pd.DataFrame:
    df = df_one.copy()
    for g in gene_cols_expected:
        if g not in df.columns:
            df[g] = 0.0
    keep = ["File Name"] + gene_cols_expected
    return df[keep]


# ============================================================
# 3) Pipeline helpers (extract expected cols, ensure cols, predict)
# ============================================================
def _read_joblib_from_upload(uploaded_file):
    buf = io.BytesIO(uploaded_file.getvalue())
    return joblib.load(buf)


def _extract_required_cols_from_pipeline(fitted_pipeline):
    prep = fitted_pipeline.named_steps["prep"]
    features = prep.named_steps["features"]

    gene_cols = []
    clin_use_cols = []
    clin_pre = None

    for name, trans, cols in features.transformers_:
        if name == "gene":
            gene_cols = list(cols)
        elif name == "clin":
            clin_use_cols = list(cols)
            clin_pre = trans

    cat_cols, num_cols, bin_cols = [], [], []
    if clin_pre is not None and hasattr(clin_pre, "transformers_"):
        for name, trans, cols in clin_pre.transformers_:
            if name == "cat":
                cat_cols = list(cols)
            elif name == "num":
                num_cols = list(cols)
            elif name == "bin":
                bin_cols = list(cols)

    return gene_cols, clin_use_cols, cat_cols, num_cols, bin_cols


def _ensure_columns(
    df: pd.DataFrame, gene_cols, clin_cols, cat_cols, num_cols, bin_cols
):
    df = df.copy()
    for c in gene_cols:
        if c not in df.columns:
            df[c] = 0.0
    for c in clin_cols:
        if c not in df.columns:
            if c in bin_cols:
                df[c] = 0
            elif c in num_cols:
                df[c] = np.nan
            else:
                df[c] = pd.NA
    return df


def _predict_risk(artifact: dict, df_one: pd.DataFrame) -> np.ndarray:
    pipe = artifact["pipeline"]
    pred = pipe.predict(df_one)
    if isinstance(pred, np.ndarray) and pred.ndim == 2:
        alpha_idx = int(artifact.get("alpha_idx", 0))
        alpha_idx = max(0, min(alpha_idx, pred.shape[1] - 1))
        return np.asarray(pred[:, alpha_idx]).ravel()
    return np.asarray(pred).ravel()


def _risk_to_group(risk_scalar: float, t1: float, t2: float) -> str:
    if risk_scalar <= t1:
        return "LOW"
    if risk_scalar <= t2:
        return "MED"
    return "HIGH"


# ============================================================
# 4) Streamlit UI
# ============================================================
st.set_page_config(
    page_title="Survival Demo (raw clinical -> auto clean)", layout="wide"
)
st.title("Survival Demo — nhập clinical THÔ (15 ô) -> auto clean -> predict")

with st.sidebar:
    st.header("1) Model artifact (.joblib)")
    model_choice = st.radio(
        "Model", ["CoxPH (lifelines wrapper)", "Coxnet (scikit-survival)"], index=0
    )

    artifact_upload = st.file_uploader(
        "Upload artifact .joblib", type=["joblib"], accept_multiple_files=False
    )
    use_local = st.checkbox("Dùng artifact local path", value=False)
    local_path = st.text_input(
        "Local path",
        value=(
            "coxph_streamlit_artifact.joblib"
            if "CoxPH" in model_choice
            else "coxnet_streamlit_artifact.joblib"
        ),
        disabled=not use_local,
    )

    st.header("2) Gene quant TSV (1 sample)")
    gene_up = st.file_uploader(
        "Upload TSV (gene_id + tpm_unstranded)",
        type=["tsv", "txt"],
        accept_multiple_files=False,
    )


# ---- Load artifact
artifact = None
artifact_src = None
try:
    if artifact_upload is not None:
        artifact = _read_joblib_from_upload(artifact_upload)
        artifact_src = f"uploaded: {artifact_upload.name}"
    elif use_local and local_path.strip():
        artifact = joblib.load(local_path.strip())
        artifact_src = f"local: {local_path.strip()}"
except Exception as e:
    st.error(f"Không load được artifact: {e}")

if artifact is None:
    st.warning("Hãy upload/chọn artifact trước.")
    st.stop()

pipe = artifact["pipeline"]
FILE_COL = artifact.get("file_col", "File Name")
t1 = float(artifact.get("t1", np.nan))
t2 = float(artifact.get("t2", np.nan))

gene_cols, clin_use_cols, cat_cols, num_cols, bin_cols = (
    _extract_required_cols_from_pipeline(pipe)
)

with st.expander("Artifact summary", expanded=True):
    st.write(f"Source: `{artifact_src}`")
    st.write(
        {
            "best_params": artifact.get("best_params", {}),
            "best_cv_cindex": artifact.get("best_cv_cindex", None),
            "test_cindex": artifact.get("test_cindex", None),
            "t1": t1,
            "t2": t2,
            "alpha_idx": artifact.get("alpha_idx", None),
            "alpha_value": artifact.get("alpha_value", None),
            "n_gene_expected": len(gene_cols),
            "n_clin_expected": len(clin_use_cols),
        }
    )

if gene_up is None:
    st.info("Upload gene quant TSV để demo dự đoán.")
    st.stop()

# ---- Load quant TSV -> 1-row gene wide
try:
    selected_file = gene_up.name
    gene_row = load_quant_tpm_geneid_to_wide(
        gene_up,
        file_name=selected_file,
        gene_cols_expected=gene_cols,
        agg="sum",
    )
    gene_row = align_to_expected_genes(gene_row, gene_cols_expected=gene_cols)
except Exception as e:
    st.error(f"Lỗi đọc gene TSV quant: {e}")
    st.stop()

st.subheader("Gene input")
st.write(f"Using sample: **{selected_file}**")
st.caption(
    "Gene values = tpm_unstranded, pivot theo gene_id; auto strip version nếu model dùng ENSG không version."
)


# ============================================================
# 5) Raw clinical form (15 fields) -> auto clean -> predict
# ============================================================
st.subheader("Nhập clinical THÔ (15 ô)")

help_text = """
Ví dụ input (không bắt buộc):
- race: 'white', 'asian', ...
- ER_status / PR_status: 'positive'/'negative'/'unknown'
- HER2_status: 'positive'/'negative'/'unknown'
- ajcc_pathologic_stage: 'Stage II', 'stage iia', ...
- ajcc_pathologic_t: 'T2', 't1c', ...
- ajcc_pathologic_n: 'N0', 'N1mi', 'N0 (i+)', ...
- ajcc_pathologic_m: 'M0'/'M1'
- laterality: 'left'/'right'
- method_of_diagnosis: 'core biopsy', 'biopsy', 'surgical resection', ...
- menopause_status: 'premenopausal'/'postmenopausal'/'unknown'
- primary_diagnosis_y: text label của bạn (vd: 'infiltrating duct carcinoma')
"""
st.info(help_text)


def select_or_other(label: str, options: list[str], default: str = "unknown") -> str:
    opts = options[:]  # copy
    if default not in opts:
        default = opts[0] if opts else "unknown"
    idx = opts.index(default)

    choice = st.selectbox(label, opts + ["other (type)"], index=idx)
    if choice == "other (type)":
        return st.text_input(f"{label} (custom)", value="")
    return choice


with st.form("raw_clinical_form"):
    st.caption(
        "Bạn có thể chọn từ dropdown. Nếu không có giá trị phù hợp, chọn **other (type)** để tự nhập."
    )

    # ===== Option lists (bạn có thể tùy chỉnh thêm) =====
    race_opts = [
        "unknown",
        "white",
        "black or african american",
        "asian",
        "other",
    ]

    status_opts = ["unknown", "positive", "negative", "equivocal"]

    stage_opts = [
        "unknown",
        "stage i",
        "stage ia",
        "stage ib",
        "stage ii",
        "stage iia",
        "stage iib",
        "stage iii",
        "stage iiia",
        "stage iiib",
        "stage iiic",
        "stage iv",
        "stage x",
    ]

    t_opts = [
        "unknown",
        "tx",
        "t0",
        "t1",
        "t1a",
        "t1b",
        "t1c",
        "t2",
        "t2a",
        "t2b",
        "t3",
        "t4",
    ]
    n_opts = [
        "unknown",
        "nx",
        "n0",
        "n0 (i+)",
        "n0 (mol+)",
        "n1",
        "n1a",
        "n1b",
        "n1c",
        "n1mi",
        "n2",
        "n2a",
        "n2b",
        "n3",
        "n3a",
        "n3b",
        "n3c",
    ]
    m_opts = ["unknown", "mx", "m0", "m1", "cm0 (i+)"]

    laterality_opts = ["unknown", "left", "right"]
    method_opts = [
        "unknown",
        "biopsy",
        "core biopsy",
        "fine needle aspiration",
        "excisional biopsy",
        "incisional biopsy",
        "ultrasound guided biopsy",
        "surgical resection",
        "surgery",
    ]
    menopause_opts = ["unknown", "premenopausal", "perimenopausal", "postmenopausal"]

    c1, c2 = st.columns(2)

    with c1:
        race = select_or_other("race", race_opts, default="unknown")
        ER_status = select_or_other("ER_status", status_opts, default="unknown")
        PR_status = select_or_other("PR_status", status_opts, default="unknown")
        HER2_status = select_or_other("HER2_status", status_opts, default="unknown")

        ajcc_pathologic_stage = select_or_other(
            "ajcc_pathologic_stage", stage_opts, default="unknown"
        )
        ajcc_pathologic_t = select_or_other(
            "ajcc_pathologic_t", t_opts, default="unknown"
        )
        ajcc_pathologic_n = select_or_other(
            "ajcc_pathologic_n", n_opts, default="unknown"
        )
        ajcc_pathologic_m = select_or_other(
            "ajcc_pathologic_m", m_opts, default="unknown"
        )

    with c2:
        laterality = select_or_other("laterality", laterality_opts, default="unknown")
        method_of_diagnosis = select_or_other(
            "method_of_diagnosis", method_opts, default="unknown"
        )

        miss_age = st.checkbox("age_at_diagnosis_years is missing", value=False)
        age_at_diagnosis_years = (
            np.nan
            if miss_age
            else st.number_input("age_at_diagnosis_years", value=0.0, step=1.0)
        )

        miss_pos = st.checkbox("lymph_nodes_positive is missing", value=False)
        lymph_nodes_positive = (
            np.nan
            if miss_pos
            else st.number_input("lymph_nodes_positive", value=0.0, step=1.0)
        )

        miss_tested = st.checkbox("lymph_nodes_tested is missing", value=False)
        lymph_nodes_tested = (
            np.nan
            if miss_tested
            else st.number_input("lymph_nodes_tested", value=0.0, step=1.0)
        )

        menopause_status = select_or_other(
            "menopause_status", menopause_opts, default="unknown"
        )

        # primary_diagnosis_y thường đa dạng -> để text input + optional dropdown “common”
        primary_common = [
            "other (type)",
            "infiltrating duct carcinoma",
            "infiltrating lobular carcinoma",
            "unknown",
        ]
        pd_choice = st.selectbox(
            "primary_diagnosis_y (optional list)", primary_common, index=0
        )
        if pd_choice == "other (type)":
            primary_diagnosis_y = st.text_input(
                "primary_diagnosis_y (custom)", value="unknown"
            )
        else:
            primary_diagnosis_y = pd_choice

    submitted = st.form_submit_button("Auto-clean & Predict")

if not submitted:
    st.stop()

# ---- Build raw clinical df (1 row)
df_clin_raw = pd.DataFrame(
    [
        {
            "race": race,
            "ER_status": ER_status,
            "PR_status": PR_status,
            "HER2_status": HER2_status,
            "ajcc_pathologic_stage": ajcc_pathologic_stage,
            "ajcc_pathologic_t": ajcc_pathologic_t,
            "ajcc_pathologic_n": ajcc_pathologic_n,
            "ajcc_pathologic_m": ajcc_pathologic_m,
            "laterality": laterality,
            "method_of_diagnosis": method_of_diagnosis,
            "age_at_diagnosis_years": age_at_diagnosis_years,
            "lymph_nodes_positive": lymph_nodes_positive,
            "lymph_nodes_tested": lymph_nodes_tested,
            "menopause_status": menopause_status,
            "primary_diagnosis_y": primary_diagnosis_y,
        }
    ]
)

# ---- Apply your clinical cleaning pipeline (inference-safe)
try:
    df_clin_clean, report = clean_clinical(
        df_clin_raw,
        drop_missing_col_threshold=1.0,  # IMPORTANT: don't drop cols in single-row inference
    )
    df_clin_clean = clean_categories(df_clin_clean)
    df_clin_clean = clean_numeric(df_clin_clean, add_missing_flags=True)
except Exception as e:
    st.error(f"Lỗi clean clinical: {e}")
    st.stop()

with st.expander("Clinical cleaning report / preview", expanded=False):
    st.write("Report:", report)
    st.write("After clean (columns):", list(df_clin_clean.columns))
    st.dataframe(df_clin_clean, use_container_width=True)

# ---- Combine gene + cleaned clinical into df_one
df_one = gene_row.copy()
for col in df_clin_clean.columns:
    df_one[col] = df_clin_clean.iloc[0][col]

# Ensure all expected columns exist (for pipeline safety)
df_one = _ensure_columns(df_one, gene_cols, clin_use_cols, cat_cols, num_cols, bin_cols)

# ---- Predict
try:
    risk = float(_predict_risk(artifact, df_one)[0])
    group = _risk_to_group(risk, t1, t2)
except Exception as e:
    st.error(f"Lỗi predict: {e}")
    st.stop()

st.success(f"✅ Risk = **{risk:.6g}**  → group: **{group}**")

# Download single-row prediction
out = pd.DataFrame(
    {
        FILE_COL: [selected_file],
        "risk": [risk],
        "risk_group": [group],
    }
)
st.download_button(
    "Download prediction CSV (1 row)",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="prediction_single_sample.csv",
    mime="text/csv",
)
