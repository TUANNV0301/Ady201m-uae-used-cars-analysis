"""
Database Builder: Converts Flat Raw CSV into Relational SQLite Database
UAE Used-Car Market Analysis (ADY201m - Phase 1)
"""

import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_PATH,
    DB_PATH,
    SQL_DIR,
    AED_TO_VND,
    FX_DATE,
    to_rel_path
)
from src.features import clean_raw_dataframe
from src.logger import get_logger

logger = get_logger("build_db", "build_db.log")


def build_relational_database():
    logger.info(f"Starting database build from raw file: {to_rel_path(RAW_DATA_PATH)}")
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data file not found: {to_rel_path(RAW_DATA_PATH)}")
        raise FileNotFoundError(f"Raw data file not found: {to_rel_path(RAW_DATA_PATH)}")

    # Remove existing database file if present to guarantee clean build
    if DB_PATH.exists():
        logger.info(f"Removing existing database file: {to_rel_path(DB_PATH)}")
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")

    # 1. Initialize Schema
    schema_path = SQL_DIR / "01_schema.sql"
    logger.info(f"Executing schema script: {to_rel_path(schema_path)}")
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    # 2. Load Raw CSV and preprocess for staging
    logger.info("Reading and staging raw listings...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    assert len(raw_df) == 10000, f"Expected 10,000 rows, found {len(raw_df)}"
    
    cleaned = clean_raw_dataframe(raw_df)
    cleaned["car_id"] = range(1, len(cleaned) + 1)

    cur = conn.cursor()

    # 3. Populate Dimension: makes
    logger.info("Populating makes dimension...")
    unique_makes = sorted(cleaned["make"].unique())
    cur.executemany(
        "INSERT INTO makes (make_name) VALUES (?)",
        [(m,) for m in unique_makes]
    )
    make_map = {row[1]: row[0] for row in cur.execute("SELECT make_id, make_name FROM makes")}

    # 4. Populate Dimension: models
    print("[*] Populating models...")
    unique_models = cleaned[["make", "model"]].drop_duplicates().sort_values(["make", "model"])
    model_records = [
        (make_map[r["make"]], r["model"]) for _, r in unique_models.iterrows()
    ]
    cur.executemany(
        "INSERT INTO models (make_id, model_name) VALUES (?, ?)",
        model_records
    )
    model_map = {
        (row[1], row[2]): row[0] 
        for row in cur.execute("SELECT model_id, make_id, model_name FROM models")
    }

    # 5. Populate Dimension: emirates
    print("[*] Populating emirates...")
    unique_emirates = sorted(cleaned["location"].unique())
    cur.executemany(
        "INSERT INTO emirates (emirate_name) VALUES (?)",
        [(e,) for e in unique_emirates]
    )
    emirate_map = {row[1]: row[0] for row in cur.execute("SELECT emirate_id, emirate_name FROM emirates")}

    # 6. Populate Dimension: body_types
    print("[*] Populating body_types...")
    unique_body_types = sorted(cleaned["body_type"].unique())
    cur.executemany(
        "INSERT INTO body_types (body_type_name) VALUES (?)",
        [(b,) for b in unique_body_types]
    )
    body_map = {row[1]: row[0] for row in cur.execute("SELECT body_type_id, body_type_name FROM body_types")}

    # 7. Populate Dimension: conditions
    print("[*] Populating conditions...")
    unique_conditions = sorted(cleaned["condition"].unique())
    cur.executemany(
        "INSERT INTO conditions (condition_name) VALUES (?)",
        [(c,) for c in unique_conditions]
    )
    condition_map = {row[1]: row[0] for row in cur.execute("SELECT condition_id, condition_name FROM conditions")}

    # 8. Populate Dimension: features
    print("[*] Populating features...")
    all_features = sorted(list({f for flist in cleaned["features_list"] for f in flist}))
    cur.executemany(
        "INSERT INTO features (feature_name) VALUES (?)",
        [(f,) for f in all_features]
    )
    feature_map = {row[1]: row[0] for row in cur.execute("SELECT feature_id, feature_name FROM features")}

    # 9. Populate FX Rates
    print("[*] Populating fx_rates...")
    cur.execute(
        "INSERT INTO fx_rates (rate_date, aed_to_vnd) VALUES (?, ?)",
        (FX_DATE, AED_TO_VND)
    )

    # 10. Populate Fact Table: cars
    print("[*] Populating cars fact table...")
    car_records = []
    for _, row in cleaned.iterrows():
        mid = model_map[(make_map[row["make"]], row["model"])]
        eid = emirate_map[row["location"]]
        bid = body_map[row["body_type"]]
        cid = condition_map[row["condition"]]
        cyl = int(row["cylinders"]) if pd.notna(row["cylinders"]) else None
        
        car_records.append((
            int(row["car_id"]),
            mid,
            eid,
            bid,
            cid,
            int(row["model_year"]),
            int(row["price_aed"]),
            int(row["mileage_km"]),
            cyl,
            str(row["transmission"]),
            str(row["fuel_type"]),
            str(row["color"])
        ))

    cur.executemany("""
        INSERT INTO cars (
            car_id, model_id, emirate_id, body_type_id, condition_id,
            model_year, price_aed, mileage_km, cylinders,
            transmission, fuel_type, color
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, car_records)

    # 11. Populate Many-to-Many Linking Table: car_features
    print("[*] Populating car_features bridge table...")
    feature_links = []
    for _, row in cleaned.iterrows():
        cid = int(row["car_id"])
        for feat in row["features_list"]:
            fid = feature_map[feat]
            feature_links.append((cid, fid))

    cur.executemany(
        "INSERT INTO car_features (car_id, feature_id) VALUES (?, ?)",
        feature_links
    )

    conn.commit()

    # 12. Run Validation Suite
    print("[*] Running foreign key and integrity verification...")
    fk_violations = list(cur.execute("PRAGMA foreign_key_check;"))
    if fk_violations:
        print(f"[!] ERROR: {len(fk_violations)} Foreign Key violations found: {fk_violations}")
        raise ValueError("Foreign key integrity check failed.")
    else:
        print("[+] SUCCESS: PRAGMA foreign_key_check returned 0 violations.")

    # Print summary counts
    print("\n--- Relational Database Summary ---")
    for tbl in ["makes", "models", "emirates", "body_types", "conditions", "features", "cars", "car_features", "fx_rates"]:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"Table '{tbl}': {cnt:,} rows")

    conn.close()
    print(f"\n[+] Database built successfully at: {to_rel_path(DB_PATH)}")


if __name__ == "__main__":
    build_relational_database()
