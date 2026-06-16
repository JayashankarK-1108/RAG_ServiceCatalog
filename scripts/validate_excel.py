"""
scripts/validate_excel.py — Pre-flight Excel validation.
Run before committing a new catalog to catch structural issues.

Usage:
    python scripts/validate_excel.py
    python scripts/validate_excel.py --file path/to/catalog.xlsx
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from config.settings import EXCEL_FILE_PATH, EXCEL_SHEET_NAME, EXCEL_HEADER_ROW, COLUMN_MAP

def validate(file_path):
    sheet = int(EXCEL_SHEET_NAME) if EXCEL_SHEET_NAME.isdigit() else EXCEL_SHEET_NAME
    print(f"\n🔍 Validating: {file_path} (sheet='{sheet}', header row={EXCEL_HEADER_ROW})\n")
    if not os.path.exists(file_path):
        print(f"❌ File not found"); return False
    try:
        df = pd.read_excel(file_path, sheet_name=sheet, header=EXCEL_HEADER_ROW, dtype=str)
    except Exception as e:
        print(f"❌ Cannot open: {e}"); return False

    print(f"   Rows    : {len(df)}")
    print(f"   Columns : {list(df.columns)}\n")

    errors, warnings = [], []
    for col in COLUMN_MAP.values():
        if col not in df.columns:
            errors.append(f"Missing column: '{col}'")
    if errors:
        [print(f"   ❌ {e}") for e in errors]; return False

    wu_col = COLUMN_MAP["wu_id"]
    empty = df[wu_col].isna().sum()
    if empty: warnings.append(f"{empty} rows have empty WU Id (will be skipped)")

    dupes = df[wu_col].dropna()
    dupes = dupes[dupes.duplicated()]
    if not dupes.empty: warnings.append(f"Duplicate WU Ids: {dupes.tolist()}")

    desc_col = COLUMN_MAP["project_services"]
    empty_desc = df[desc_col].isna().sum()
    if empty_desc: warnings.append(f"{empty_desc} rows have empty 'Project related Services'")

    [print(f"   ⚠️  {w}") for w in warnings] if warnings else print("   ✅ No issues")
    print(f"\n   📊 {len(df)-empty}/{len(df)} rows will be ingested\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default=EXCEL_FILE_PATH)
    args = parser.parse_args()
    sys.exit(0 if validate(args.file) else 1)
