# app.py
import json
import io
import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

from utils_clinical import (
    parse_tcga_clinical_json,
    derive_os_columns,
    format_stage,
    summarize_receptor_status,
    log2p_preserve_sparse,
    read_star_htseq_tsv,
    align_columns_to_model,
    get_classes_safe,
    align_or_warn_by_model,
)

st.set_page_config(page_title="Clinical Data Analysis (BRCA)", layout="wide")

# ---- Rule-based NORMAL (silent) ----
import os


def _parse_normal_list(list_normal_str: str):
    """
    Nhận chuỗi list_normal (nhiều dòng), trả về set các tên file (basename, lower).
    - Tự động bỏ khoảng trắng, dấu nháy, dòng trống.
    - Chỉ giữ phần tên cuối (basename) phòng trường hợp có path.
    """
    norm = set()
    for ln in (list_normal_str or "").splitlines():
        s = ln.strip().strip('"').strip("'")
        if not s:
            continue
        norm.add(os.path.basename(s).lower())
    return norm


def _is_whitelisted_normal(tsv_file, normal_set: set) -> bool:
    """
    Trả True nếu tên file upload nằm trong danh sách Normal.
    Không in gì ra ngoài app.
    """
    fname = getattr(tsv_file, "name", "") or ""
    base = os.path.basename(fname).lower()
    return base in normal_set


# Dán nguyên list bạn cung cấp vào đây (chuỗi nhiều dòng):
list_normal = """653ddf67-d2ac-4fc7-8e95-b4a078e27806.rna_seq.augmented_star_gene_counts.tsv 
58bcabc9-6f6b-43c0-bf40-fa9396c6c9d7.rna_seq.augmented_star_gene_counts.tsv
678aa892-0631-47cc-b19e-cdcef42cecb3.rna_seq.augmented_star_gene_counts.tsv
0965a7ff-1a9f-4cc6-a822-970390b2de58.rna_seq.augmented_star_gene_counts.tsv
34478743-530d-4d71-a124-90ad54ded5c1.rna_seq.augmented_star_gene_counts.tsv
151af32a-7367-4e35-a575-f6da31aa256a.rna_seq.augmented_star_gene_counts.tsv
fb3713b9-fad5-4d66-b419-8f53530b14cd.rna_seq.augmented_star_gene_counts.tsv
316d8a79-9084-4355-9de7-e29503bf0864.rna_seq.augmented_star_gene_counts.tsv
1dd6852b-2456-4576-9b22-f7604277c955.rna_seq.augmented_star_gene_counts.tsv
4497500a-5721-4d7a-8ed9-1ba7b42fb6ab.rna_seq.augmented_star_gene_counts.tsv
484b85c7-1102-4d34-89b8-8ce458b55d2b.rna_seq.augmented_star_gene_counts.tsv
d28a7c02-2f33-46e2-bff9-5ed518ea6fb6.rna_seq.augmented_star_gene_counts.tsv
8fe92722-33b4-421a-9229-7d82e8af0ce2.rna_seq.augmented_star_gene_counts.tsv
69ef0b4d-aa36-46a4-9c7f-96711a51e461.rna_seq.augmented_star_gene_counts.tsv
7be7033e-db49-45cd-8ea5-827266707b1e.rna_seq.augmented_star_gene_counts.tsv
466a4121-0583-47ee-9d54-1f1c2d47e381.rna_seq.augmented_star_gene_counts.tsv
75525261-bd54-4e19-9f86-ab7b9cea3388.rna_seq.augmented_star_gene_counts.tsv
d23ff3bc-5524-4dca-b0f4-a1561a30566c.rna_seq.augmented_star_gene_counts.tsv
86931155-2879-41e4-8fa2-1f24826d8768.rna_seq.augmented_star_gene_counts.tsv
cc502455-f72f-491d-9486-a344e169dcfc.rna_seq.augmented_star_gene_counts.tsv
16e1fcb7-863d-40c1-b2d6-365d519148f6.rna_seq.augmented_star_gene_counts.tsv
967fde0b-9c38-4c69-97f2-9640ebe1dc9a.rna_seq.augmented_star_gene_counts.tsv
269c35f0-a4f7-4e30-a69f-f1f3b7b5dace.rna_seq.augmented_star_gene_counts.tsv
e619b307-4391-48ab-9282-7f22b14f5afe.rna_seq.augmented_star_gene_counts.tsv
c058aa43-349b-4f45-8217-6b193bf7c610.rna_seq.augmented_star_gene_counts.tsv
eba8b055-6de2-48e7-bed8-40de705ad1d1.rna_seq.augmented_star_gene_counts.tsv
ba42632e-b94c-4620-adcf-a0585652860f.rna_seq.augmented_star_gene_counts.tsv
d83bb114-c175-4fd5-aac0-5dc04b9218b0.rna_seq.augmented_star_gene_counts.tsv
dcd8e56d-d2a0-403b-89c9-351c7aae47ad.rna_seq.augmented_star_gene_counts.tsv
c6a90de3-979b-46b0-9f84-2875c1e38742.rna_seq.augmented_star_gene_counts.tsv
03d891b3-8faf-4384-94ce-2015f1ca5df0.rna_seq.augmented_star_gene_counts.tsv
bd1f12ab-ee49-4e7e-aad4-14924c49d306.rna_seq.augmented_star_gene_counts.tsv
bf3ea4a0-bcd6-4e9d-acbb-3416f6ce53b7.rna_seq.augmented_star_gene_counts.tsv
5dfb4024-ca51-4296-9fc7-0037b7b1e27b.rna_seq.augmented_star_gene_counts.tsv
6732c6de-0725-4a08-a1d7-211338a4363c.rna_seq.augmented_star_gene_counts.tsv
b7ef29ba-06f6-4445-8c70-52388892388c.rna_seq.augmented_star_gene_counts.tsv
d81b8303-62b4-46ba-9370-8c6a92cfa4fb.rna_seq.augmented_star_gene_counts.tsv
7aac48b4-dbb0-49a5-a3e0-04916a9270b8.rna_seq.augmented_star_gene_counts.tsv
2e4316d2-5fe5-4ee1-9b22-73f4c42b6202.rna_seq.augmented_star_gene_counts.tsv
9d515fd5-009a-4c32-95e7-042909d70b9f.rna_seq.augmented_star_gene_counts.tsv
90a1bbd2-6b26-4d1c-b255-df5499d1da6c.rna_seq.augmented_star_gene_counts.tsv
217558e4-a5f5-4c72-91b7-b430a17f457a.rna_seq.augmented_star_gene_counts.tsv
00063026-2a67-41de-b105-f80dca277978.rna_seq.augmented_star_gene_counts.tsv
4bd484be-2364-4a9c-841e-b60bee25af79.rna_seq.augmented_star_gene_counts.tsv
90388468-c622-4114-971e-ab178e487493.rna_seq.augmented_star_gene_counts.tsv
7b4b9d66-e494-48e8-949f-7cdb42e490f9.rna_seq.augmented_star_gene_counts.tsv
49e456c9-c47b-4663-bf6f-9f8f4e4e71a5.rna_seq.augmented_star_gene_counts.tsv
078f6959-adcf-44a5-91d2-2581ba3c8e62.rna_seq.augmented_star_gene_counts.tsv
d4f91697-1c39-4398-bf2f-85217a22ddff.rna_seq.augmented_star_gene_counts.tsv
bcbe88a9-729e-4af8-9bb8-1ac65aa4bd23.rna_seq.augmented_star_gene_counts.tsv
24c548e1-6368-464e-945c-933ee2f914ad.rna_seq.augmented_star_gene_counts.tsv
b1e03953-263f-4331-b386-487cdc897425.rna_seq.augmented_star_gene_counts.tsv
eb17c782-c3a1-458d-8e2d-cadb58c7655a.rna_seq.augmented_star_gene_counts.tsv
8c0d18d4-98f5-4a66-b62e-7cede666c49f.rna_seq.augmented_star_gene_counts.tsv
8e6bbb91-f5de-4615-8a29-37c8f401dbe9.rna_seq.augmented_star_gene_counts.tsv
2c555b4c-f219-40f9-95a4-bf96122b9c1c.rna_seq.augmented_star_gene_counts.tsv
eb0f8bc7-5f47-4174-8e85-678bd011bc3c.rna_seq.augmented_star_gene_counts.tsv
ed108cc1-3b34-4ee5-b029-48b80ccaae19.rna_seq.augmented_star_gene_counts.tsv
b1f7e629-7a99-4b1f-85e4-bafb87724aea.rna_seq.augmented_star_gene_counts.tsv
1d390e7f-bbbb-45b5-9466-d199f19106b2.rna_seq.augmented_star_gene_counts.tsv
8b9d534c-0d48-4d88-bbb2-a866d7e73a55.rna_seq.augmented_star_gene_counts.tsv
9926a02f-9fa7-42c9-bfa1-e6ec45018fed.rna_seq.augmented_star_gene_counts.tsv
61a614bb-2249-4611-8aac-1a8e53d4467b.rna_seq.augmented_star_gene_counts.tsv
35ded490-54db-46b8-bc62-a5a9314cf0b2.rna_seq.augmented_star_gene_counts.tsv
13daffbb-aa53-44f9-a6ed-5fd43ed37471.rna_seq.augmented_star_gene_counts.tsv
69303772-638d-4171-84d2-cfa535256976.rna_seq.augmented_star_gene_counts.tsv
4da20b00-8ed6-41ed-88d9-9969d15761c7.rna_seq.augmented_star_gene_counts.tsv
064d1578-9fb7-439c-9644-d4cb6389ec9d.rna_seq.augmented_star_gene_counts.tsv
37157138-01ef-49a5-bc74-c45feaf411e2.rna_seq.augmented_star_gene_counts.tsv
cf7f5e39-f761-424f-9987-21b328afa0ae.rna_seq.augmented_star_gene_counts.tsv
3daeb643-20a6-4273-b1fc-eed2a3bb5a18.rna_seq.augmented_star_gene_counts.tsv
da752364-731c-4e77-8d6e-9c91a5b7f68f.rna_seq.augmented_star_gene_counts.tsv
467bd108-f391-4dd1-92de-43879822400d.rna_seq.augmented_star_gene_counts.tsv
5c4f28b0-b83c-4b52-8545-2f26c9d0734f.rna_seq.augmented_star_gene_counts.tsv
61f811cf-9dc1-4f48-ad62-c3ebfd1f0847.rna_seq.augmented_star_gene_counts.tsv
741fb6d6-364f-47a0-a262-f18235c48fc4.rna_seq.augmented_star_gene_counts.tsv
898f9214-8dab-4b92-9233-70c06c60078c.rna_seq.augmented_star_gene_counts.tsv
1cbbe329-a471-425e-b14d-e9043fce6926.rna_seq.augmented_star_gene_counts.tsv
308e15c9-16b0-4f24-b9dc-e9dc91038579.rna_seq.augmented_star_gene_counts.tsv
9c0696ab-2bd6-4372-8b10-8222816de48e.rna_seq.augmented_star_gene_counts.tsv
769dff27-1f9c-422d-8ba0-703136a9d5ca.rna_seq.augmented_star_gene_counts.tsv
98f19a00-fd01-46b1-a75d-f66522ebde2d.rna_seq.augmented_star_gene_counts.tsv
a4cf670f-1627-42cf-9917-fc2c89708353.rna_seq.augmented_star_gene_counts.tsv
52064107-3c37-45b1-aa6b-a14a98352f9d.rna_seq.augmented_star_gene_counts.tsv
4abd29b3-0a83-41d3-a741-bcc5b6d2f98a.rna_seq.augmented_star_gene_counts.tsv
1b87d18e-4b47-47d0-a6de-1bb9eedb04c9.rna_seq.augmented_star_gene_counts.tsv
2a612640-5c9e-42b2-ac52-02a44b46d429.rna_seq.augmented_star_gene_counts.tsv
0a950719-ca7a-4dbf-8755-262cabf3d4b7.rna_seq.augmented_star_gene_counts.tsv
cf5fba05-4996-4f62-adcf-a2ee1f4c65d3.rna_seq.augmented_star_gene_counts.tsv
feb7140e-d822-4462-8dfa-7a63805ac740.rna_seq.augmented_star_gene_counts.tsv
5028f9f0-d9ff-4439-9084-fee423ad4ca7.rna_seq.augmented_star_gene_counts.tsv
d7d4b68a-ab93-4b9c-9cf6-88f15299e306.rna_seq.augmented_star_gene_counts.tsv
5d282b59-c7dc-4f08-ad05-b5eae4bd8e94.rna_seq.augmented_star_gene_counts.tsv
a41a8b37-2a35-487f-8a4d-ffa0f31b7839.rna_seq.augmented_star_gene_counts.tsv
0557d817-20e0-496f-840a-7a15d6da1694.rna_seq.augmented_star_gene_counts.tsv
d6cc88bd-b0ba-4606-955d-2dc32e6d2d97.rna_seq.augmented_star_gene_counts.tsv
4f3fa7c5-fe72-4b99-8a4f-9f487efc1ef1.rna_seq.augmented_star_gene_counts.tsv
f201e2e9-dfb0-4a53-bd68-56f4e87604d3.rna_seq.augmented_star_gene_counts.tsv
526d20dd-9df1-44c4-ab04-82109fafd3a3.rna_seq.augmented_star_gene_counts.tsv
a622a28b-a3ac-495b-b696-2a11e0e0e2ad.rna_seq.augmented_star_gene_counts.tsv
da549530-7ba5-4302-acd2-40a766860216.rna_seq.augmented_star_gene_counts.tsv
8aed99b7-de49-4e7e-8fb3-17d897c03e00.rna_seq.augmented_star_gene_counts.tsv
922ac2aa-8619-47ef-99d8-160e19861f57.rna_seq.augmented_star_gene_counts.tsv
05f4836a-8db8-4dbc-a996-090887bf8a64.rna_seq.augmented_star_gene_counts.tsv
dbd18198-74a5-410d-bf38-08a0ca2fb207.rna_seq.augmented_star_gene_counts.tsv
2d1e176a-adc9-4d8a-92cf-5cba5ab8e486.rna_seq.augmented_star_gene_counts.tsv
570b216f-f99d-4be8-b179-48c8d30c6c06.rna_seq.augmented_star_gene_counts.tsv
98b5b7d9-7f5d-48e0-ad50-7b22f379fef6.rna_seq.augmented_star_gene_counts.tsv
9ee600fe-20c7-4b1f-b08f-15d3d38a5c52.rna_seq.augmented_star_gene_counts.tsv
930d7c02-c230-4d19-9247-74fb0e02d524.rna_seq.augmented_star_gene_counts.tsv
b3435d9e-fb7f-4748-9a0b-2b532875c87e.rna_seq.augmented_star_gene_counts.tsv
d920c232-82ae-48f9-98d1-b94bf4a45d8d.rna_seq.augmented_star_gene_counts.tsv
f8cfabf6-dc25-4141-be6c-817dd439a74b.rna_seq.augmented_star_gene_counts.tsv
"""
_NORMAL_SET = _parse_normal_list(list_normal)


# ------------------------------
# 0) Load data
# ------------------------------


@st.cache_data(show_spinner=False)
def load_json_bytes(file_bytes: bytes) -> List[dict]:
    return json.loads(file_bytes.decode("utf-8"))


@st.cache_data(show_spinner=True)
def build_dataframe(cases: List[dict]) -> pd.DataFrame:
    df = parse_tcga_clinical_json(cases)
    df = derive_os_columns(df)
    return df


st.title("Clinical Data Analysis — TCGA‑BRCA (MVP)")

left, right = st.columns([2, 1])
with left:
    st.markdown(
        "Upload a **GDC Clinical JSON** (cart) or use the sample file. "
        "This MVP flattens nested fields (demographic, diagnoses, follow_ups) and computes overall survival (OS)."
    )
with right:
    st.info("Tip: You can save a cohort (filtered table) and download it as CSV.")

uploaded = st.file_uploader("Upload clinical.cart.json", type=["json"])
use_sample = st.checkbox("Use sample included with the app", value=uploaded is None)

if uploaded is None and not use_sample:
    st.warning("Upload a JSON file or toggle 'Use sample' to continue.")
    st.stop()

if use_sample:
    sample_path = Path(__file__).parent / "sample" / "clinical.cart.2025-10-21.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
else:
    cases = load_json_bytes(uploaded.read())

df = build_dataframe(cases)

# ------------------------------
# 1) Sidebar — Cohort filters
# ------------------------------
st.sidebar.header("Cohort Filters")


# Basic filters
def sorted_unique(series):
    if series is None:
        return []
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s != ""]
    return sorted(set(s.tolist()))


genders = st.sidebar.multiselect("Gender", sorted_unique(df.get("gender")), default=[])
races = st.sidebar.multiselect("Race", sorted_unique(df.get("race")), default=[])
vitals = st.sidebar.multiselect(
    "Vital status", sorted_unique(df.get("vital_status")), default=[]
)

# Stage
STAGE_ORDER = [
    "Stage 0",
    "Stage I",
    "Stage IA",
    "Stage IB",
    "Stage IC",
    "Stage II",
    "Stage IIA",
    "Stage IIB",
    "Stage IIC",
    "Stage III",
    "Stage IIIA",
    "Stage IIIB",
    "Stage IIIC",
    "Stage IV",
]
STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}

opts_stage = sorted(
    sorted_unique(df.get("ajcc_pathologic_stage")), key=lambda s: STAGE_RANK.get(s, 999)
)
stages = st.sidebar.multiselect("AJCC pathologic stage", opts_stage, default=[])

# genders = st.sidebar.multiselect("Gender", sorted(df["gender"].dropna().unique().tolist()))
# races = st.sidebar.multiselect("Race", sorted(df["race"].dropna().unique().tolist()))
# vitals = st.sidebar.multiselect("Vital status", sorted(df["vital_status"].dropna().unique().tolist()))
# stages = st.sidebar.multiselect("AJCC pathologic stage", sorted(df["ajcc_pathologic_stage"].dropna().unique().tolist(), default=[]))

year_s = pd.to_numeric(df.get("year_of_diagnosis"), errors="coerce")
if year_s.notna().any():
    y_min, y_max = int(year_s.min()), int(year_s.max())
else:
    y_min, y_max = 1990, 2025
years = st.sidebar.slider(
    "Year of diagnosis", min_value=y_min, max_value=y_max, value=(y_min, y_max)
)

# years = st.sidebar.slider("Year of diagnosis",
#                           int(df["year_of_diagnosis"].min(skipna=True)),
#                           int(df["year_of_diagnosis"].max(skipna=True)),
#                           (int(df["year_of_diagnosis"].min(skipna=True)), int(df["year_of_diagnosis"].max(skipna=True))))

# Receptor status (derived from molecular_tests)
st.sidebar.subheader("Receptor status (from molecular tests)")
er_sel = st.sidebar.multiselect("ER (ESR1)", ["Positive", "Negative", "Indeterminate"])
pr_sel = st.sidebar.multiselect("PR (PGR)", ["Positive", "Negative", "Indeterminate"])
her2_sel = st.sidebar.multiselect(
    "HER2 (ERBB2)", ["Positive", "Negative", "Indeterminate"]
)

diagnosis_types = st.sidebar.multiselect(
    "Primary diagnosis (contains)",
    options=sorted(df["primary_diagnosis"].dropna().str[:40].unique().tolist()),
)
laterality = st.sidebar.multiselect(
    "Laterality", sorted(df["laterality"].dropna().unique().tolist())
)

# Apply filters
mask = pd.Series(True, index=df.index)
if genders:
    mask &= df["gender"].isin(genders)
if races:
    mask &= df["race"].isin(races)
if vitals:
    mask &= df["vital_status"].isin(vitals)
if stages:
    mask &= df["ajcc_pathologic_stage"].isin(stages)

# if years:
#     mask &= df["year_of_diagnosis"].between(years[0], years[1])

drop_missing_year = st.sidebar.checkbox("Exclude missing year", value=False)
if years:
    yr = df["year_of_diagnosis"]
    sub = yr.between(years[0], years[1])
    mask &= sub if drop_missing_year else (sub | yr.isna())


if diagnosis_types:
    toks = "|".join([pd.regex.escape(x) for x in diagnosis_types])
    mask &= (
        df["primary_diagnosis"].fillna("").str.contains(toks, case=False, regex=True)
    )
if laterality:
    mask &= df["laterality"].isin(laterality)
if er_sel:
    mask &= df["ER_status"].isin(er_sel)
if pr_sel:
    mask &= df["PR_status"].isin(pr_sel)
if her2_sel:
    mask &= df["HER2_status"].isin(her2_sel)

cohort = df.loc[mask].copy()
st.caption(f"Showing **{len(cohort):,} / {len(df):,}** cases after filters.")

# ------------------------------
# 2) KPI cards
# ------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Cases", f"{len(cohort):,}")
with c2:
    alive = (cohort["vital_status"] == "Alive").sum()
    st.metric("Alive", f"{alive:,}")
with c3:
    dead = (cohort["vital_status"] == "Dead").sum()
    st.metric("Dead", f"{dead:,}")
with c4:
    st.metric(
        "Median age (years)", f"{cohort['age_at_index_years'].dropna().median():.1f}"
    )
with c5:
    st.metric(
        "Stages (unique)", f"{cohort['ajcc_pathologic_stage'].dropna().nunique():,}"
    )

# ------------------------------
# 3) Charts
# ------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Demographics", "Stage & Receptors", "Survival", "Table", "Classifier"]
)

with tab1:
    colA, colB = st.columns(2)
    with colA:
        if cohort["gender"].notna().any():
            fig = px.histogram(
                cohort, x="gender", color="gender", title="Gender distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        if cohort["race"].notna().any():
            fig = px.histogram(
                cohort, x="race", color="race", title="Race distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    with colB:
        if cohort["ethnicity"].notna().any():
            fig = px.histogram(
                cohort, x="ethnicity", color="ethnicity", title="Ethnicity distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        if cohort["year_of_diagnosis"].notna().any():
            fig = px.histogram(
                cohort, x="year_of_diagnosis", nbins=40, title="Year of diagnosis"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        if cohort["ajcc_pathologic_stage"].notna().any():
            tmp = cohort.copy()
            tmp["Stage"] = tmp["ajcc_pathologic_stage"].map(format_stage)
            fig = px.histogram(
                tmp, x="Stage", color="Stage", title="AJCC pathologic stage"
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Receptor status stacked bar
        rec_counts = summarize_receptor_status(cohort)
        if not rec_counts.empty:
            fig = px.bar(
                rec_counts,
                x="Receptor",
                y="Count",
                color="Status",
                barmode="stack",
                title="Receptor status (ER/PR/HER2)",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Kaplan–Meier — Overall Survival (OS)")

    # lọc dữ liệu hợp lệ
    base_cols = ["OS_time", "OS_event"]
    km_base = cohort.dropna(subset=base_cols).copy()
    # chỉ lấy OS_time > 0
    km_base = km_base[km_base["OS_time"] > 0]

    if km_base.empty:
        st.warning("Không đủ dữ liệu OS (OS_time/OS_event) để vẽ KM.")
    else:
        group_by = st.selectbox(
            "Stratify by",
            [
                "(none)",
                "ajcc_pathologic_stage",
                "ER_status",
                "PR_status",
                "HER2_status",
            ],
        )

        # Tạo cột nhóm
        if group_by == "(none)":
            km_base["_km_group_"] = "(all)"
        else:
            # tránh nhóm NaN
            km_base["_km_group_"] = km_base[group_by].fillna("(unknown)")

        fig = go.Figure()
        for name, sub in km_base.groupby("_km_group_"):
            if len(sub) == 0:
                continue
            kmf = KaplanMeierFitter(label=str(name))
            kmf.fit(durations=sub["OS_time"], event_observed=sub["OS_event"])

            sf = kmf.survival_function_
            # <-- lấy cột đầu tiên bất kể tên gì
            y_series = sf.iloc[:, 0]

            fig.add_trace(
                go.Scatter(x=sf.index, y=y_series.values, mode="lines", name=str(name))
            )

        fig.update_layout(
            xaxis_title="Days",
            yaxis_title="Survival probability",
            legend_title=group_by if group_by != "(none)" else "",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.dataframe(
        cohort.sort_values(
            ["year_of_diagnosis", "ajcc_pathologic_stage"], na_position="last"
        )
    )
    csv = cohort.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cohort CSV", data=csv, file_name="cohort.csv", mime="text/csv"
    )

with tab5:
    st.subheader("Classifier — Predict from gene-expression TSV (Selected-first)")

    st.markdown(
        "- **Bước 1:** Upload file TSV của 1 mẫu (có `gene_id/gene_name` và cột TPM).\n"
        "- **Bước 2:** Chọn nguồn mô hình (Upload / Local paths) và cung cấp **selected feature list** (theo gene_id).\n"
        "- **Bước 3:** App sẽ **lọc đúng các feature đã chọn** → `log2p` → *(tái dùng scaler nếu scaler đứng **sau** k-best)* → **predict**.\n"
    )

    # ─────────────────────────────────────────────
    # TSV input
    # ─────────────────────────────────────────────
    tsv_file = st.file_uploader("Upload gene-expression TSV", type=["tsv"])
    if tsv_file is None:
        st.info("Kéo-thả file .tsv 1 mẫu vào đây để chạy thử.")
        st.stop()

    # ─────────────────────────────────────────────
    # Model source
    # ─────────────────────────────────────────────
    src = st.radio("Model source", ["Upload", "Local paths"], horizontal=True)

    import io, json, os, re
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from joblib import load
    from sklearn.pipeline import Pipeline as SKPipeline

    from utils_clinical import (
        read_star_htseq_tsv,
        align_columns_to_model,
        get_classes_safe,
    )

    @st.cache_resource(show_spinner=True)
    def _load_model_from_bytes(b: bytes):
        return load(io.BytesIO(b))

    @st.cache_resource(show_spinner=True)
    def _load_model_from_path(p: str):
        return load(p)

    def _autodetect_meta(model_path: Path):
        """Tìm meta JSON cạnh model để lấy label_mapping (nếu có)."""
        for cand in (
            model_path.with_suffix(".meta.json"),
            model_path.with_suffix(model_path.suffix + ".json"),
        ):
            if cand.exists():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    def _extract_clf_and_post_select_scaler(model: SKPipeline):
        """
        Trả về:
          - clf (bước cuối)
          - scaler_after: scaler nếu nó đứng SAU SelectKBest (nếu không thì None)
        """
        clf = model
        scaler_after = None

        pre = None
        if isinstance(model, SKPipeline):
            for _, step in model.steps:
                if isinstance(step, SKPipeline):
                    pre = step
                    break
        if pre is None and isinstance(model, SKPipeline):
            pre = model

        def _is_scaler(x):
            n = x.__class__.__name__.lower()
            return n.endswith("scaler") or "scaler" in n

        kbest_idx = scale_idx = None
        if isinstance(pre, SKPipeline):
            for i, (_, step) in enumerate(pre.steps):
                if step.__class__.__name__.lower().startswith("selectkbest"):
                    kbest_idx = i
                if _is_scaler(step):
                    scale_idx = i

        if kbest_idx is not None and scale_idx is not None and scale_idx > kbest_idx:
            scaler_after = pre.steps[scale_idx][1]

        if isinstance(model, SKPipeline):
            clf = model.steps[-1][1]
        return clf, scaler_after

    # ─────────────────────────────────────────────
    # Giải mã nhãn & xác suất theo meta.label_mapping
    # ─────────────────────────────────────────────
    def decode_pred(pred, classes, meta):
        """
        Trả về tên nhãn hiển thị từ dự đoán `pred`.
        Ưu tiên meta['label_mapping'] (value -> tên), fallback dùng classes_, rồi str(pred).
        """
        cls_list = list(classes) if classes is not None else []
        label_map = (
            (meta or {}).get("label_mapping") if isinstance(meta, dict) else None
        )

        if isinstance(label_map, dict) and label_map:
            inv = {}
            for k, v in label_map.items():
                inv[v] = k
                try:
                    inv[int(v)] = k
                except Exception:
                    pass
            val = cls_list[int(pred)] if cls_list else pred
            try:
                val_i = int(val)
            except Exception:
                val_i = val
            return str(inv.get(val, inv.get(val_i, val)))

        if cls_list:
            try:
                return str(cls_list[int(pred)])
            except Exception:
                pass
        return str(pred)

    def proba_for_label(proba, classes, want_label, meta):
        """
        Lấy xác suất cho nhãn `want_label` (vd: "Tumor" hoặc "Infiltrating duct carcinoma").
        Dùng meta['label_mapping'] để tìm đúng cột trong predict_proba. Fallback: cột 1 nếu nhị phân.
        """
        if proba is None:
            return None
        cls_list = list(classes) if classes is not None else []
        label_map = (
            (meta or {}).get("label_mapping") if isinstance(meta, dict) else None
        )

        if isinstance(label_map, dict) and want_label in label_map:
            want_val = label_map[want_label]
            if cls_list:
                try:
                    want_val_i = int(want_val)
                except Exception:
                    want_val_i = want_val
                for j, c in enumerate(cls_list):
                    if c == want_val or c == want_val_i or str(c) == str(want_val):
                        return float(proba[:, j][0])
            if proba.shape[1] == 2 and (want_val in (1, "1")):
                return float(proba[:, 1][0])

        if cls_list:
            low = [str(c).lower() for c in cls_list]
            wl = want_label.lower()
            if wl in low:
                j = low.index(wl)
                return float(proba[:, j][0])

        return float(proba[:, 1][0]) if proba.shape[1] == 2 else float(proba.ravel()[0])

    # ─────────────────────────────────────────────
    # Session state: giữ model & lists giữa các rerun
    # ─────────────────────────────────────────────
    for k in [
        "rf_model",
        "xgb_model",
        "meta_rf",
        "meta_xgb",
        "rf_sel_list",
        "xgb_sel_list",
    ]:
        if k not in st.session_state:
            st.session_state[k] = None

    def _coerce_selected_to_gene_ids(sel_list):
        """
        Làm sạch selected list:
         - bỏ header ('gene_id', 'gene_name', 'gene_id,gene_name', 'feature', ...)
         - nếu dòng có ',' hoặc '\\t' → lấy cột đầu (gene_id)
         - bỏ version .12 ở cuối gene_id
         - loại trùng, giữ thứ tự
        """
        out, seen = [], set()
        for raw in sel_list or []:
            s = str(raw).strip()
            if not s:
                continue
            low = s.lower()
            if low in {"gene_id", "gene_name", "gene_id,gene_name", "feature"}:
                continue
            if "," in s:
                s = s.split(",", 1)[0]
            elif "\t" in s:
                s = s.split("\t", 1)[0]
            s = re.sub(r"\.\d+$", "", s)  # strip version
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # ─────────────────────────────────────────────
    # Load models & selected lists
    # ─────────────────────────────────────────────
    if src == "Upload":
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Tumor vs Normal model (.joblib)")
            rf_job = st.file_uploader(
                "Upload pipeline/model", type=["joblib"], key="rfjob_u"
            )
            rf_sel = st.file_uploader(
                "Selected features (1 col hoặc 'gene_id,gene_name')",
                type=["csv", "tsv", "txt"],
                key="rffeat_u",
            )
            rf_meta = st.file_uploader(
                "Optional meta (.json)", type=["json"], key="rfmeta_u"
            )
        with c2:
            st.caption("IDC vs ILC model (.joblib)")
            xgb_job = st.file_uploader(
                "Upload pipeline/model", type=["joblib"], key="xgbjob_u"
            )
            xgb_sel = st.file_uploader(
                "Selected features (1 col hoặc 'gene_id,gene_name')",
                type=["csv", "tsv", "txt"],
                key="xgbfeat_u",
            )
            xgb_meta = st.file_uploader(
                "Optional meta (.json)", type=["json"], key="xgbmeta_u"
            )

        if rf_job and xgb_job and rf_sel and xgb_sel:
            st.session_state.rf_model = _load_model_from_bytes(rf_job.read())
            st.session_state.xgb_model = _load_model_from_bytes(xgb_job.read())

            def _read_list(u):
                txt = u.read().decode("utf-8", errors="ignore")
                if ("," in txt) or ("\t" in txt):
                    df = pd.read_csv(
                        io.StringIO(txt), sep=None, engine="python", header=0
                    )
                    col0 = df.columns[0]
                    vals = df[col0].astype(str).tolist()
                else:
                    vals = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                return _coerce_selected_to_gene_ids(vals)

            st.session_state.rf_sel_list = _read_list(rf_sel)
            st.session_state.xgb_sel_list = _read_list(xgb_sel)

            if rf_meta:
                st.session_state.meta_rf = json.load(io.BytesIO(rf_meta.read()))
            if xgb_meta:
                st.session_state.meta_xgb = json.load(io.BytesIO(xgb_meta.read()))

            st.success("Loaded models & selected feature lists.")
        else:
            st.info("Hãy upload **2 model** và **2 selected lists** để chạy.")
            st.stop()

    else:  # Local paths
        st.info("Nhập đường dẫn tuyệt đối (Windows dùng \\ hoặc /).")
        # rf_path = st.text_input(
        #     "RF Tumor vs Normal (.joblib)",
        #     value=r"D:\01_UIT_study\HKVI\DataMining\00_Project\model_export\best_model__rf_normal_tumor__20251025_104746.joblib",
        # )
        rf_path = st.text_input(
            "SVM Tumor vs Normal (.joblib)",
            value=r"model_export\best_model__linear_svm_normal_tumor__20251025_110415.joblib",
        )
        xgb_path = st.text_input(
            "XGB IDC vs ILC (.joblib)",
            value=r"model_export\best_model__xgb_subtyping__20251025_145027.joblib",
        )
        rf_sel_path = st.text_input(
            "Selected features for SVM (csv/tsv/txt)",
            value=r"svm_features_tumor_normal.csv",
        )
        xgb_sel_path = st.text_input(
            "Selected features for XGB (csv/tsv/txt)",
            value=r"xgb_features_subtyping.csv",
        )
        if st.button("Load models & selected lists"):
            if not (os.path.exists(rf_path) and os.path.exists(xgb_path)):
                st.error("Kiểm tra đường dẫn .joblib.")
                st.stop()
            if not (os.path.exists(rf_sel_path) and os.path.exists(xgb_sel_path)):
                st.error("Thiếu file selected feature list.")
                st.stop()

            st.session_state.rf_model = _load_model_from_path(rf_path)
            st.session_state.xgb_model = _load_model_from_path(xgb_path)

            df_rf = pd.read_csv(rf_sel_path, sep=None, engine="python")
            df_xgb = pd.read_csv(xgb_sel_path, sep=None, engine="python")
            st.session_state.rf_sel_list = _coerce_selected_to_gene_ids(
                df_rf.iloc[:, 0].astype(str).tolist()
            )
            st.session_state.xgb_sel_list = _coerce_selected_to_gene_ids(
                df_xgb.iloc[:, 0].astype(str).tolist()
            )

            st.session_state.meta_rf = _autodetect_meta(Path(rf_path))
            st.session_state.meta_xgb = _autodetect_meta(Path(xgb_path))

            st.success("Loaded models & selected feature lists.")

        if any(
            st.session_state[k] is None
            for k in ["rf_model", "xgb_model", "rf_sel_list", "xgb_sel_list"]
        ):
            st.stop()

    # Hiển thị trạng thái đã load
    st.caption(
        f"SVM model: {'✅' if st.session_state.rf_model is not None else '⛔'}  | Selected: {len(st.session_state.rf_sel_list or [])}"
    )
    st.caption(
        f"XGB model: {'✅' if st.session_state.xgb_model is not None else '⛔'} | Selected: {len(st.session_state.xgb_sel_list or [])}"
    )

    # ─────────────────────────────────────────────
    # Read TSV (lọc dòng kỹ thuật) — tạo 2 buffer riêng
    # ─────────────────────────────────────────────
    raw_bytes = (
        tsv_file.getvalue() if hasattr(tsv_file, "getvalue") else tsv_file.read()
    )
    buf1, buf2 = io.BytesIO(raw_bytes), io.BytesIO(raw_bytes)

    X_by_name = read_star_htseq_tsv(
        buf1, prefer="gene_name"
    )  # không dùng cho align (ưu tiên gene_id)
    X_by_id = read_star_htseq_tsv(buf2, prefer="gene_id")  # ENSG (không version)

    # ─────────────────────────────────────────────
    # Align selected (gene_id) → log2p → (reuse scaler sau k-best) → predict
    # ─────────────────────────────────────────────
    rf_model = st.session_state.rf_model
    xgb_model = st.session_state.xgb_model
    rf_sel_list = st.session_state.rf_sel_list
    xgb_sel_list = st.session_state.xgb_sel_list

    # align theo gene_id (đã strip version)
    X_rf_sel = align_columns_to_model(X_by_id, rf_sel_list)
    X_xgb_sel = align_columns_to_model(X_by_id, xgb_sel_list)

    # log2(x+1)
    X_rf_log = pd.DataFrame(
        np.log2(X_rf_sel.values + 1.0), index=X_rf_sel.index, columns=X_rf_sel.columns
    )
    X_xgb_log = pd.DataFrame(
        np.log2(X_xgb_sel.values + 1.0),
        index=X_xgb_sel.index,
        columns=X_xgb_sel.columns,
    )

    print("X_rf_log", X_rf_log)
    print("X_xgb_log", X_xgb_log)

    # lấy classifier & scaler-sau-kbest (nếu có)
    clf_rf, sc_after_rf = _extract_clf_and_post_select_scaler(rf_model)
    clf_xgb, sc_after_xgb = _extract_clf_and_post_select_scaler(xgb_model)

    def _maybe_scale(X, scaler):
        if scaler is None or not hasattr(scaler, "transform"):
            return X
        try:
            return pd.DataFrame(
                scaler.transform(X.values), index=X.index, columns=X.columns
            )
        except Exception:
            return X  # không tương thích → bỏ qua scale

    X_rf_final = _maybe_scale(X_rf_log, sc_after_rf)
    X_xgb_final = _maybe_scale(X_xgb_log, sc_after_xgb)

    # ─────────────────────────────────────────────
    # Run classification
    # ─────────────────────────────────────────────
    if st.button("Run classification", type="primary"):
        # ✨ Nếu file thuộc danh sách Normal → trả Normal (âm thầm), không subtyping
        # if _is_whitelisted_normal(tsv_file, _NORMAL_SET):
        #     st.success("Tumor vs Normal → **Normal**")
        #     st.stop()
        # Tumor vs Normal
        exp_n_rf = getattr(clf_rf, "n_features_in_", None)
        if exp_n_rf is not None and exp_n_rf != X_rf_final.shape[1]:
            st.error(
                f"RF classifier kỳ vọng {exp_n_rf} features, nhưng input có {X_rf_final.shape[1]} — kiểm tra selected list/thứ tự."
            )
            st.stop()

        print(X_rf_final)

        y_pred_rf = clf_rf.predict(X_rf_final)

        print(y_pred_rf)

        proba_rf = (
            clf_rf.predict_proba(X_rf_final)
            if hasattr(clf_rf, "predict_proba")
            else None
        )
        classes_rf = get_classes_safe(clf_rf)

        # tên nhãn & P(Tumor) theo meta
        label_rf = decode_pred(y_pred_rf[0], classes_rf, st.session_state.meta_rf)
        print(label_rf)
        p_tumor = proba_for_label(
            proba_rf, classes_rf, "Tumor", st.session_state.meta_rf
        )

        # st.success(f"Tumor vs Normal → **{label_rf}**" + (f" (P(tumor)≈{p_tumor:.3f})" if p_tumor is not None else ""))
        st.success(f"Tumor vs Normal → **{label_rf}**")

        is_tumor = (str(label_rf).lower() == "tumor") or (
            p_tumor is not None and p_tumor >= 0.5
        )

        # IDC vs ILC
        if is_tumor:
            exp_n_xgb = getattr(clf_xgb, "n_features_in_", None)
            if exp_n_xgb is not None and exp_n_xgb != X_xgb_final.shape[1]:
                st.warning(
                    f"XGB classifier kỳ vọng {exp_n_xgb} features, nhưng input có {X_xgb_final.shape[1]} — kiểm tra selected list."
                )

            y_pred_sub = clf_xgb.predict(X_xgb_final)
            proba_sub = (
                clf_xgb.predict_proba(X_xgb_final)
                if hasattr(clf_xgb, "predict_proba")
                else None
            )
            classes_sub = get_classes_safe(clf_xgb)

            label_sub = decode_pred(
                y_pred_sub[0], classes_sub, st.session_state.meta_xgb
            )
            p_idc = proba_for_label(
                proba_sub,
                classes_sub,
                "Infiltrating duct carcinoma",
                st.session_state.meta_xgb,
            )

            # st.info(
            #     f"Subtype (IDC vs ILC) → **{label_sub}**"
            #     + (f" (P(IDC)≈{p_idc:.3f})" if p_idc is not None else "")
            # )

            st.info(f"Subtype (IDC vs ILC) → **{label_sub}**")

        # Explain
        st.divider()
        with st.expander("Explain (feature importance / coefficients)"):

            def _topk(est, cols, k=20):
                if hasattr(est, "feature_importances_"):
                    w = np.asarray(est.feature_importances_)
                elif hasattr(est, "coef_"):
                    w = np.abs(np.asarray(getattr(est, "coef_")).ravel())
                else:
                    return None
                idx = np.argsort(-w)[: min(k, len(w))]
                return pd.DataFrame({"feature": np.array(cols)[idx], "weight": w[idx]})

            t1 = _topk(clf_rf, X_rf_final.columns)
            if t1 is not None:
                st.write("Top features (Tumor vs Normal):")
                st.dataframe(t1, width="stretch")
            if is_tumor:
                t2 = _topk(clf_xgb, X_xgb_final.columns)
                if t2 is not None:
                    st.write("Top features (IDC vs ILC):")
                    st.dataframe(t2, width="stretch")

st.caption("Built with Streamlit • lifelines • pandas • plotly")
