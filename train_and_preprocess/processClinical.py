#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def choose_baseline_diagnosis(diagnoses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chọn diagnosis baseline:
    - Ưu tiên diagnosis_is_primary_disease == "true" AND classification_of_tumor == "primary"
    - Nếu nhiều cái, lấy days_to_diagnosis nhỏ nhất; nếu vẫn hòa, lấy record đầu tiên.
    - Nếu không có diagnosis thỏa ưu tiên, fallback: diagnosis có days_to_diagnosis nhỏ nhất trong tất cả.
    """
    if not diagnoses:
        return None

    ranked = []
    for i, d in enumerate(diagnoses):
        is_primary_disease = str(d.get("diagnosis_is_primary_disease", "")).lower() == "true"
        is_primary_class = str(d.get("classification_of_tumor", "")).lower() == "primary"
        priority = 0 if (is_primary_disease and is_primary_class) else 1

        days = _to_float(d.get("days_to_diagnosis"))
        if np.isnan(days):
            days = np.inf

        ranked.append((priority, days, i, d))

    ranked.sort(key=lambda t: (t[0], t[1], t[2]))
    return ranked[0][3]


def extract_menopause_status(case: Dict[str, Any]) -> Optional[str]:
    """
    menopause_status (nếu có) thường nằm trong follow_ups[].other_clinical_attributes[].
    Ưu tiên other_clinical_attributes.timepoint_category == "Initial Diagnosis".
    """
    best = None
    best_rank = (1, 10**9, 10**9)  # (is_initial, followup_idx, attr_idx)

    for fu_i, fu in enumerate(case.get("follow_ups", []) or []):
        for oca_i, oca in enumerate(fu.get("other_clinical_attributes", []) or []):
            ms = oca.get("menopause_status")
            if ms in (None, "", "not reported", "Not Reported"):
                continue

            is_initial = 0 if str(oca.get("timepoint_category", "")).lower() == "initial diagnosis" else 1
            rank = (is_initial, fu_i, oca_i)
            if rank < best_rank:
                best_rank = rank
                best = ms

    return best


def category_from_treatment_type(treatment_type: Any) -> Optional[str]:
    """
    Map treatment_type -> {surgery, radiation, chemo, hormone}.
    """
    if not treatment_type:
        return None
    s = str(treatment_type).lower()
    if "surgery" in s:
        return "surgery"
    if "radiation" in s:
        return "radiation"
    if "chemotherapy" in s:
        return "chemo"
    if "hormone" in s:
        return "hormone"
    return None


def extract_treatment_features(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tạo flags tx_*_any dựa vào diagnoses[].treatments[].treatment_type.
    Optional: min(days_to_treatment_start) theo từng loại.
    Quy ước: bỏ qua record có treatment_or_therapy == "no"; còn lại (yes/unknown/missing) tính là có.
    """
    any_flag = {"surgery": 0, "radiation": 0, "chemo": 0, "hormone": 0}
    min_start = {"surgery": np.nan, "radiation": np.nan, "chemo": np.nan, "hormone": np.nan}

    for diag in case.get("diagnoses", []) or []:
        for tx in diag.get("treatments", []) or []:
            cat = category_from_treatment_type(tx.get("treatment_type"))
            if not cat:
                continue

            tor = str(tx.get("treatment_or_therapy", "")).lower()
            if tor == "no":
                continue

            any_flag[cat] = 1

            dts = _to_float(tx.get("days_to_treatment_start"))
            if not np.isnan(dts):
                if np.isnan(min_start[cat]) or dts < min_start[cat]:
                    min_start[cat] = dts

    return {
        "tx_surgery_any": any_flag["surgery"],
        "tx_radiation_any": any_flag["radiation"],
        "tx_chemo_any": any_flag["chemo"],
        "tx_hormone_any": any_flag["hormone"],
        "tx_surgery_min_days_to_start": min_start["surgery"],
        "tx_radiation_min_days_to_start": min_start["radiation"],
        "tx_chemo_min_days_to_start": min_start["chemo"],
        "tx_hormone_min_days_to_start": min_start["hormone"],
    }


def compute_os(case: Dict[str, Any], baseline_diag: Optional[Dict[str, Any]]) -> Tuple[float, int]:
    """
    OS_event:
      - 1 nếu demographic.vital_status == "Dead" (hoặc "deceased")
      - 0 nếu còn lại
    OS_time_days:
      - nếu event==1: days_to_death (fallback: last follow up nếu thiếu)
      - nếu event==0: days_to_last_follow_up (ưu tiên baseline diagnosis; nếu thiếu thì lấy max từ follow_ups.days_to_follow_up)
    """
    demo = case.get("demographic") or {}
    vital = str(demo.get("vital_status", "")).lower()
    is_dead = vital in ("dead", "deceased")

    fu_candidates = []
    if baseline_diag is not None:
        fu_candidates.append(_to_float(baseline_diag.get("days_to_last_follow_up")))
    fu_candidates.append(_to_float(case.get("days_to_last_follow_up")))
    fu_candidates.append(_to_float(demo.get("days_to_last_follow_up")))
    for fu in case.get("follow_ups", []) or []:
        fu_candidates.append(_to_float(fu.get("days_to_follow_up")))

    cand = [x for x in fu_candidates if not np.isnan(x)]
    last_fu = float(np.nanmax(cand)) if cand else np.nan

    if is_dead:
        dtd = _to_float(demo.get("days_to_death"))
        time = dtd if not np.isnan(dtd) else last_fu
        return time, 1

    return last_fu, 0

def extract_marker_status(case, gene_symbol: str, method_priority=("IHC",)) -> str:
    """
    Lấy marker status từ follow_ups[].molecular_tests[] theo gene_symbol.
    Ưu tiên:
      1) follow_up.timepoint_category == "Initial Diagnosis"
      2) molecular_analysis_method theo method_priority
      3) days_to_follow_up nhỏ nhất (nếu có), rồi record đầu tiên
    Return: test_result (string) hoặc None.
    """
    best = None
    best_rank = (1, 10**9, 10**9, 10**9, 10**9)  # (is_initial, method_rank, days, fu_i, mt_i)

    for fu_i, fu in enumerate(case.get("follow_ups", []) or []):
        tpc = str(fu.get("timepoint_category", "")).lower()
        is_initial = 0 if tpc == "initial diagnosis" else 1

        days = fu.get("days_to_follow_up")
        try:
            days = float(days) if days is not None else np.inf
        except Exception:
            days = np.inf

        for mt_i, mt in enumerate(fu.get("molecular_tests", []) or []):
            if str(mt.get("gene_symbol", "")).upper() != gene_symbol.upper():
                continue

            method = str(mt.get("molecular_analysis_method", "")).upper()
            if method_priority:
                method_rank = method_priority.index(method) if method in method_priority else 999
            else:
                method_rank = 0

            test_result = mt.get("test_result")
            if test_result in (None, "", "not reported", "Not Reported"):
                continue

            rank = (is_initial, method_rank, days, fu_i, mt_i)
            if rank < best_rank:
                best_rank = rank
                best = test_result

    return best



def build_feature_row(case: Dict[str, Any]) -> Dict[str, Any]:
    diagnoses = case.get("diagnoses", []) or []
    base = choose_baseline_diagnosis(diagnoses)

    os_time, os_event = compute_os(case, base)

    demo = case.get("demographic") or {}
    

    row: Dict[str, Any] = {
        "case_id": case.get("case_id"),
        "submitter_id": case.get("submitter_id"),
        "OS_time_days": os_time,
        "OS_event": os_event,
        # Demographic
        "age_at_index": _to_float(demo.get("age_at_index")),
        "gender": demo.get("gender"),
        "race": demo.get("race"),
        "ethnicity": demo.get("ethnicity"),
        "country_of_residence_at_enrollment": demo.get("country_of_residence_at_enrollment"),
    }
    row["disease_type"] = case.get("disease_type")
    row["ER_status"] = extract_marker_status(case, "ESR1", method_priority=("IHC",))
    row["PR_status"] = extract_marker_status(case, "PGR", method_priority=("IHC",))
    row["HER2_status"] = extract_marker_status(case, "ERBB2", method_priority=("IHC", "FISH"))
    # Diagnosis + Pathology (baseline)
    if base is None:
        row.update(
            {
                "ajcc_pathologic_stage": None,
                "ajcc_pathologic_t": None,
                "ajcc_pathologic_n": None,
                "ajcc_pathologic_m": None,
                "age_at_diagnosis_years": np.nan,
                "laterality": None,
                "primary_diagnosis": None,
                "morphology": None,
                "year_of_diagnosis": np.nan,
                "method_of_diagnosis": None,
                "prior_malignancy": None,
                "prior_treatment": None,
                "lymph_nodes_positive": np.nan,
                "lymph_nodes_tested": np.nan,
                "positive_ratio": np.nan,
            }
        )
    else:
        row.update(
            {
                "ajcc_pathologic_stage": base.get("ajcc_pathologic_stage"),
                "ajcc_pathologic_t": base.get("ajcc_pathologic_t"),
                "ajcc_pathologic_n": base.get("ajcc_pathologic_n"),
                "ajcc_pathologic_m": base.get("ajcc_pathologic_m"),
                "laterality": base.get("laterality"),
                "primary_diagnosis": base.get("primary_diagnosis"),
                "morphology": base.get("morphology"),
                "year_of_diagnosis": base.get("year_of_diagnosis"),
                "method_of_diagnosis": base.get("method_of_diagnosis"),
                "prior_malignancy": base.get("prior_malignancy"),
                "prior_treatment": base.get("prior_treatment"),
            }
        )

        # age_at_diagnosis: thường là "days" -> đổi sang năm
        aad = _to_float(base.get("age_at_diagnosis"))
        row["age_at_diagnosis_years"] = (aad / 365.25) if not np.isnan(aad) else np.nan

        pdets = base.get("pathology_details", []) or []
        pdet = pdets[0] if pdets else {}
        ln_pos = _to_float(pdet.get("lymph_nodes_positive"))
        ln_tst = _to_float(pdet.get("lymph_nodes_tested"))

        row["lymph_nodes_positive"] = ln_pos
        row["lymph_nodes_tested"] = ln_tst
        row["positive_ratio"] = (ln_pos / ln_tst) if (ln_tst and not np.isnan(ln_pos) and ln_tst > 0) else np.nan

    # Other
    row["menopause_status"] = extract_menopause_status(case)

    # Treatments (across all diagnoses)
    row.update(extract_treatment_features(case))

    return row

import numpy as np



def build_os_feature_table(input_json: Path) -> pd.DataFrame:
    with input_json.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    rows = [build_feature_row(case) for case in cases]
    df = pd.DataFrame(rows)

    # đảm bảo 1 dòng / 1 case_id
    df = df.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TCGA OS feature table from clinical.cart JSON")
    parser.add_argument("--input", type=str, required=True, help="Path to clinical.cart.2025-10-21.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()

    df = build_os_feature_table(Path(args.input))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output} | shape={df.shape}")
