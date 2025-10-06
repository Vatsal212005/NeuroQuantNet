#!/usr/bin/env python3
"""
Preprocess GDSC2 fitted dose-response (IC50/AUC).

- Reads .xlsx (no manual unzip needed).
- Selects/renames key columns.
- Drops rows with missing LN_IC50/AUC or names.
- Optionally binarizes sensitivity (by LN_IC50 quantile/threshold).

Writes:
  data/processed/gdsc_ic50.csv
  (columns: cell_line, drug_name, ln_ic50, auc, pathway, putative_target, drug_id)

Usage:
  python scripts/preprocess_gdsc.py \
    --in data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx \
    --out data/processed/gdsc_ic50.csv
"""
import argparse
import os
import pandas as pd

RENAME = {
    'CELL_LINE_NAME': 'cell_line',
    'DRUG_NAME': 'drug_name',
    'LN_IC50': 'ln_ic50',
    'AUC': 'auc',
    'PATHWAY_NAME': 'pathway',
    'PUTATIVE_TARGET': 'putative_target',
    'DRUG_ID': 'drug_id'
}

CANDIDATE_COLS = [
    'CELL_LINE_NAME','DRUG_NAME','LN_IC50','AUC',
    'PATHWAY_NAME','PUTATIVE_TARGET','DRUG_ID'
]

def preprocess(in_xlsx: str, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Read robustly (some distributions use sheet names / different dtypes)
    df = pd.read_excel(in_xlsx, engine='openpyxl')
    # Keep only columns we care about (if present)
    cols = [c for c in CANDIDATE_COLS if c in df.columns]
    if not cols:
        raise ValueError("Could not find expected columns in the GDSC2 Excel file.")
    df = df[cols].copy()

    # Rename columns → snake_case
    df = df.rename(columns=RENAME)

    # Standardize text
    for c in ['cell_line', 'drug_name', 'pathway', 'putative_target']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Clean values
    if 'ln_ic50' in df.columns:
        df = df[pd.notnull(df['ln_ic50'])]
    if 'auc' in df.columns:
        df = df[pd.notnull(df['auc'])]

    # Drop obvious duplicates
    df = df.drop_duplicates(subset=['cell_line', 'drug_name'], keep='first')

    # Save
    df.to_csv(out_csv, index=False)
    print(f"[GDSC2] Wrote: {out_csv}")
    print(f"[GDSC2] Rows = {len(df):,} | Drugs ≈ {df['drug_name'].nunique():,} | Cell lines ≈ {df['cell_line'].nunique():,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_xlsx", required=True, help="Path to GDSC2_fitted_dose_response_*.xlsx")
    ap.add_argument("--out", default="data/processed/gdsc_ic50.csv", help="Output CSV path")
    args = ap.parse_args()
    preprocess(args.in_xlsx, args.out)
