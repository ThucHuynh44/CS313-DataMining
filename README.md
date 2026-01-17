# Clinical Data Analysis — TCGA‑BRCA (MVP)

This mini web app lets you explore a **TCGA clinical cart JSON** and reproduce a subset of the *GDC Analysis Center → Clinical Data Analysis* experience:

- Cohort filters (gender, race, stage, year of diagnosis, laterality + receptor status from `molecular_tests`).
- Demographic charts (gender, race, ethnicity), stage distribution, receptor status.
- Kaplan–Meier overall survival (OS) curves, optionally stratified by stage or receptors.
- Download the filtered cohort to CSV.

> **Input**: a JSON file exported from GDC *Repository → Clinical* (aka `clinical.cart.*.json`).  
> **Output**: plots and tables for your selected cohort.

## Run

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## Files

- `app.py` — Streamlit UI.
- `utils_clinical.py` — JSON flattener + derived features.
- `sample/clinical.cart.2025-10-21.json` — a small sample to try immediately.
- `requirements.txt` — Python dependencies.

## Notes

- **OS definition**: `OS_event=1` if `vital_status="Dead"` and `days_to_death` is present.  
  `OS_time = days_to_death` if event else `max(days_to_follow_up, days_to_last_follow_up)` as a proxy.
- **Receptor status** is extracted from `follow_ups[].molecular_tests[]` using `gene_symbol ∈ {ESR1,PGR,ERBB2}` and `test_result` ∈ {`Positive`,`Negative`, otherwise `Indeterminate`}.
- The parser keeps only the **primary diagnosis** per case (or the one with `days_to_diagnosis==0`, otherwise the first).

This is an MVP; feel free to expand with: PFS, treatment timelines, per‑center comparisons, and shapable exports.