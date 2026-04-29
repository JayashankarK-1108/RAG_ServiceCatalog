"""
scripts/validate_excel.py — Pre-flight Excel validation.
Run before committing a new catalog to catch structural issues.

Usage:
    python scripts/validate_excel.py
    python scripts/validate_excel.py --file path/to/catalog.xlsx
"""

import sys, os, argparse
import pandas as pd

EXCEL_FILE_PATH: str = os.getenv("EXCEL_FILE_PATH", "data/service_catalog.xlsx")
COLUMN_MAP: dict = {
    "wu_id":               "WU Id",
    "business_scope":      "Business Scope",
    "hosting_environment": "Hosting Environment",
    "tech_tower":          "Tech Tower",
    "technology":          "Technology",
    "activities_category": "Activities Category",
    "project_services":    "Project related Services",
    "sla_notes":           "Column1",
}

def validate(file_path):
    print(f"\n🔍 Validating: {file_path}\n")
    if not os.path.exists(file_path):
        print(f"❌ File not found"); return False
    try:
        df = pd.read_excel(file_path, sheet_name=0, dtype=str)
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
