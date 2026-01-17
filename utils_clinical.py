# utils_clinical.py
from typing import List, Dict, Any, Tuple
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest

# ------------------------------
# Parsing helpers
# ------------------------------

def _pick_primary_diagnosis(diags: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not diags:
        return {}
    # Prefer explicit primary
    prims = [d for d in diags if str(d.get("diagnosis_is_primary_disease")).lower() == "true"]
    if prims:
        return prims[0]
    # Otherwise choose with days_to_diagnosis==0 else first
    zero = [d for d in diags if d.get("days_to_diagnosis") in (0, "0", 0.0)]
    return zero[0] if zero else diags[0]

def _max_followup_days(fups: List[Dict[str, Any]]) -> float:
    if not fups:
        return np.nan
    vals = []
    for f in fups:
        v = f.get("days_to_follow_up")
        if v is not None:
            try:
                vals.append(float(v))
            except Exception:
                pass
    return float(np.nanmax(vals)) if vals else np.nan

def _first_progression_days(fups: List[Dict[str, Any]]) -> float:
    """Earliest progression or recurrence days, if present."""
    if not fups:
        return np.nan
    vals = []
    for f in fups:
        if str(f.get("progression_or_recurrence", "")).lower() == "yes":
            v = f.get("days_to_progression") or f.get("days_to_recurrence") or f.get("days_to_follow_up")
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
    return float(np.nanmin(vals)) if vals else np.nan

def _receptor_from_molecular_tests(fups: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    """Return (ER_status, PR_status, HER2_status) from molecular_tests in follow_ups."""
    er, pr, her2 = [], [], []
    for f in (fups or []):
        for test in f.get("molecular_tests", []) or []:
            gene = str(test.get("gene_symbol", "")).upper()
            res  = str(test.get("test_result", "")).title()
            if gene == "ESR1":   # Estrogen receptor
                er.append(res)
            elif gene == "PGR":  # Progesterone receptor
                pr.append(res)
            elif gene == "ERBB2":# HER2
                her2.append(res)
    def _normalize(vals):
        if not vals:
            return np.nan
        # Prefer Positive/Negative; otherwise Indeterminate
        for v in vals:
            if v in {"Positive", "Negative"}:
                return v
        return "Indeterminate"
    return _normalize(er), _normalize(pr), _normalize(her2)

# ------------------------------
# Public parse function
# ------------------------------

def parse_tcga_clinical_json(cases: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in cases:
        case_id = row.get("case_id")
        submitter_id = row.get("submitter_id")
        proj = (row.get("project") or {}).get("project_id")
        disease_type = row.get("disease_type")
        primary_site = row.get("primary_site")
        demo = row.get("demographic") or {}
        fups = row.get("follow_ups") or []
        diags = row.get("diagnoses") or []

        dprim = _pick_primary_diagnosis(diags)
        # core fields
        stage = dprim.get("ajcc_pathologic_stage")
        tnmt = (dprim.get("ajcc_pathologic_t"), dprim.get("ajcc_pathologic_n"), dprim.get("ajcc_pathologic_m"))
        ydiag = dprim.get("year_of_diagnosis")
        laterality = dprim.get("laterality")
        pdx = dprim.get("primary_diagnosis")

        # followup & survival proxy
        dlf = dprim.get("days_to_last_follow_up")
        last_fu = _max_followup_days(fups)
        followup_days = float(last_fu if not np.isnan(last_fu) else (dlf if dlf is not None else np.nan))

        # receptors
        er, pr, her2 = _receptor_from_molecular_tests(fups)

        rows.append({
            "case_id": case_id,
            "submitter_id": submitter_id,
            "project_id": proj,
            "disease_type": disease_type,
            "primary_site": primary_site,
            "gender": demo.get("gender"),
            "race": demo.get("race"),
            "ethnicity": demo.get("ethnicity"),
            "vital_status": demo.get("vital_status"),
            "age_at_index_years": demo.get("age_at_index"),
            "days_to_death": demo.get("days_to_death"),
            "days_to_birth": demo.get("days_to_birth"),
            "primary_diagnosis": pdx,
            "ajcc_pathologic_stage": stage,
            "ajcc_pathologic_t": tnmt[0],
            "ajcc_pathologic_n": tnmt[1],
            "ajcc_pathologic_m": tnmt[2],
            "year_of_diagnosis": ydiag,
            "laterality": laterality,
            "days_to_last_follow_up": dprim.get("days_to_last_follow_up"),
            "max_days_to_follow_up": last_fu,
            "follow_up_days": followup_days,
            "ER_status": er,
            "PR_status": pr,
            "HER2_status": her2,
        })
    df = pd.DataFrame(rows)
    # Consistent types
    for c in ["age_at_index_years", "days_to_death", "days_to_birth", "year_of_diagnosis",
              "days_to_last_follow_up", "max_days_to_follow_up", "follow_up_days"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------------
# Derivations
# ------------------------------

def derive_os_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # OS event: 1 if vital_status == Dead and days_to_death is present
    df["OS_event"] = ((df["vital_status"] == "Dead") & df["days_to_death"].notna()).astype(int)
    # OS time: if event then days_to_death, else use follow_up_days (proxy)
    df["OS_time"] = np.where(df["OS_event"] == 1, df["days_to_death"], df["follow_up_days"])
    return df

def format_stage(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    # normalize like 'Stage IIA' -> 'IIA'
    m = re.search(r"Stage\s+([IVX]+[ABC]?)", s, flags=re.I)
    return m.group(1) if m else s

def summarize_receptor_status(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rec_col, rec_name in [("ER_status","ER"), ("PR_status","PR"), ("HER2_status","HER2")]:
        counts = (df[rec_col]
                  .fillna("Unknown")
                  .value_counts(dropna=False)
                  .rename_axis("Status")
                  .reset_index(name="Count"))
        counts["Receptor"] = rec_name
        out.append(counts)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["Status","Count","Receptor"])

# ====== CLASSIFIER UTILS ======
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

def log2p_preserve_sparse(x):
    """log2(x+1) cho ndarray/DataFrame; tương thích khi model kỳ vọng log2p."""
    return np.log2(np.asarray(x, dtype=float) + 1.0)

# def read_star_htseq_tsv(tsv_path_or_buf) -> pd.DataFrame:
#     """
#     Đọc TSV 1 mẫu (file bạn gửi kiểu STAR/HTSeq/gene-counts).
#     Trả về DataFrame: 1 hàng (sample) × nhiều cột (gene symbol/ID).
#     - Bỏ dòng kỹ thuật: '__no_feature', 'N_unmapped', ...
#     - Nếu tên gene dạng 'ENSG...|TP53' → lấy phần sau dấu '|'
#     """
#     df = pd.read_csv(tsv_path_or_buf, sep="\t", comment="#", dtype=str)
#     bad_prefixes = ("__", "N_")
#     df = df[~df.iloc[:, 0].astype(str).str.startswith(bad_prefixes)]

#     # đoán cột numeric làm giá trị
#     num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
#     if len(num_cols) < 2:
#         gene_col, val_col = df.columns[0], df.columns[1]
#     else:
#         gene_col = df.columns[0]
#         counts = [(c, pd.to_numeric(df[c], errors="coerce").notna().sum()) for c in df.columns[1:]]
#         val_col = sorted(counts, key=lambda x: -x[1])[0][0]

#     vals = pd.to_numeric(df[val_col], errors="coerce")
#     genes = df[gene_col].astype(str)
#     if genes.str.contains(r"\|").any():
#         genes = genes.str.split("|").str[-1]

#     s = pd.Series(vals.values, index=genes.values, dtype=float).groupby(level=0).mean()
#     out = pd.DataFrame([s])
#     out.index = ["sample_1"]
#     return out

def read_star_htseq_tsv(tsv_path_or_buf, prefer="gene_name") -> pd.DataFrame:
    df = pd.read_csv(tsv_path_or_buf, sep="\t", comment="#", dtype=str)

    # cột giá trị
    val_col = None
    for cand in ["tpm_unstranded", "TPM", "tpm", "expected_count", "count"]:
        if cand in df.columns:
            val_col = cand; break
    if val_col is None:
        num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        if not num_cols:
            raise ValueError("Không tìm thấy cột numeric (TPM/count).")
        val_col = max(num_cols, key=lambda c: pd.to_numeric(df[c], errors="coerce").notna().sum())

    vals = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)

    # key (gene_name hoặc gene_id bỏ version)
    if prefer == "gene_name" and "gene_name" in df.columns and df["gene_name"].notna().any():
        key = df["gene_name"].astype(str).str.strip()
    else:
        if "gene_id" not in df.columns:
            raise ValueError("Không có gene_name và gene_id trong TSV.")
        key = df["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)

    # LOẠI DÒNG KỸ THUẬT
    bad = key.astype(str).str.startswith(("__", "N_"))
    vals = vals[~bad]; key = key[~bad]

    s = pd.Series(vals.values, index=key.values, dtype=float).groupby(level=0).mean()
    out = pd.DataFrame([s]); out.index = ["sample_1"]
    return out

def align_columns_to_model(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Căn cột X đúng thứ tự/tập feature model mong đợi; thiếu → điền 0; thừa → bỏ."""
    cols = []
    for g in feature_names:
        cols.append(X[g] if g in X.columns else pd.Series(0.0, index=X.index))
    X_aligned = pd.concat(cols, axis=1)
    X_aligned.columns = feature_names
    return X_aligned

def get_classes_safe(estimator) -> List[str]:
    """Lấy classes_ từ estimator hay bước cuối của Pipeline."""
    import sklearn
    from sklearn.pipeline import Pipeline
    cls = getattr(estimator, "classes_", None)
    if cls is None and isinstance(estimator, Pipeline):
        cls = getattr(estimator.steps[-1][1], "classes_", None)
    return list(cls) if cls is not None else []

# ====== MODEL FEATURE NAMES ======
def extract_feature_names_from_model(model) -> list:
    """
    Cố gắng lấy danh sách cột (feature order) mà model đã fit trên đó.
    Ưu tiên:
      - model.feature_names_in_
      - bước đầu tiên trong Pipeline có feature_names_in_
      - bất kỳ step nào có feature_names_in_
    Trả về [] nếu không tìm thấy.
    """
    names = []
    # 1) Bản thân model
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    # 2) Nếu là Pipeline, duyệt lần lượt các step
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            for _, step in model.steps:
                if hasattr(step, "feature_names_in_"):
                    try:
                        names = list(step.feature_names_in_)
                        if names:
                            return names
                    except Exception:
                        pass
    except Exception:
        pass

    # 3) Không thấy
    return []

def align_or_warn_by_model(X_raw: pd.DataFrame, model):
    """
    Dùng feature_names lấy trực tiếp từ model để align X theo đúng thứ tự.
    Nếu không trích xuất được, trả về X_raw và cờ cảnh báo.
    """
    base_feats = extract_feature_names_from_model(model)
    if base_feats:
        mis = [c for c in base_feats if c not in X_raw.columns]
        if mis:
            # gene thiếu → điền 0
            import pandas as pd
            cols = []
            for g in base_feats:
                if g in X_raw.columns:
                    cols.append(X_raw[g])
                else:
                    cols.append(pd.Series(0.0, index=X_raw.index))
            X_aligned = pd.concat(cols, axis=1)
            X_aligned.columns = base_feats
        else:
            X_aligned = X_raw.reindex(columns=base_feats)
        return X_aligned, None
    else:
        return X_raw, "Model/Pipeline không cung cấp feature_names_in_. Không thể đảm bảo thứ tự cột khớp lúc train."

def read_feature_list(file_obj_or_path) -> list:
    import io, os
    if hasattr(file_obj_or_path, "read"):  # uploader
        text = file_obj_or_path.read().decode("utf-8", errors="ignore")
        feats = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return feats
    p = str(file_obj_or_path)
    if p.lower().endswith((".tsv",".tab")):
        s = pd.read_csv(p, sep="\t", header=None).iloc[:,0]
    elif p.lower().endswith(".csv"):
        s = pd.read_csv(p, header=None).iloc[:,0]
    else:
        s = pd.read_csv(p, sep=None, engine="python", header=None).iloc[:,0]
    s = s.astype(str).str.strip()
    return [x for x in s.tolist() if x]

def has_var_kbest(pipeline: Pipeline) -> bool:
    if not isinstance(pipeline, Pipeline):
        return False
    names = [n for n,_ in pipeline.steps]
    # tìm theo class (an toàn hơn so với tên)
    has_var = any(isinstance(s, VarianceThreshold) for _,s in pipeline.steps)
    has_kb  = any(isinstance(s, SelectKBest) for _,s in pipeline.steps)
    return has_var and has_kb

# def align_columns_to_model(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
#     cols = []
#     for g in feature_names:
#         cols.append(X[g] if g in X.columns else pd.Series(0.0, index=X.index))
#     X2 = pd.concat(cols, axis=1)
#     X2.columns = feature_names
#     return X2