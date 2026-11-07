-- ====================================================================
-- UAE Used-Car Market Analysis: Staging and Loading Strategy
-- File: sql/02_load.sql
-- ====================================================================

-- 1. Staging Table: Mirror of raw CSV with added surrogate car_id and parsed columns
CREATE TABLE IF NOT EXISTS stg_listings (
    car_id INTEGER PRIMARY KEY,
    raw_make TEXT,
    raw_model TEXT,
    raw_year INTEGER,
    raw_price INTEGER,
    raw_mileage INTEGER,
    raw_body_type TEXT,
    raw_cylinders TEXT,
    raw_transmission TEXT,
    raw_fuel_type TEXT,
    raw_color TEXT,
    raw_location TEXT,
    raw_description TEXT,
    clean_location TEXT,
    clean_cylinders INTEGER,
    parsed_condition TEXT,
    parsed_features TEXT
);

-- 2. Populate Makes Dimension
-- INSERT OR IGNORE INTO makes (make_name)
-- SELECT DISTINCT LOWER(TRIM(raw_make)) FROM stg_listings ORDER BY 1;

-- 3. Populate Models Dimension
-- INSERT OR IGNORE INTO models (make_id, model_name)
-- SELECT DISTINCT mk.make_id, LOWER(TRIM(stg.raw_model))
-- FROM stg_listings stg
-- JOIN makes mk ON mk.make_name = LOWER(TRIM(stg.raw_make))
-- ORDER BY mk.make_id, 2;

-- 4. Populate Emirates Dimension
-- INSERT OR IGNORE INTO emirates (emirate_name)
-- SELECT DISTINCT clean_location FROM stg_listings ORDER BY 1;

-- 5. Populate Body Types Dimension
-- INSERT OR IGNORE INTO body_types (body_type_name)
-- SELECT DISTINCT TRIM(raw_body_type) FROM stg_listings ORDER BY 1;

-- 6. Populate Conditions Dimension
-- INSERT OR IGNORE INTO conditions (condition_name)
-- SELECT DISTINCT parsed_condition FROM stg_listings ORDER BY 1;

-- 7. Populate Features Dimension
-- Executed via Python parser script extracting unique equipment items.

-- 8. Populate Cars Fact Table
-- INSERT INTO cars (car_id, model_id, emirate_id, body_type_id, condition_id,
--                   model_year, price_aed, mileage_km, cylinders, transmission, fuel_type, color)
-- SELECT 
--     stg.car_id,
--     m.model_id,
--     e.emirate_id,
--     b.body_type_id,
--     c.condition_id,
--     stg.raw_year,
--     stg.raw_price,
--     stg.raw_mileage,
--     stg.clean_cylinders,
--     stg.raw_transmission,
--     stg.raw_fuel_type,
--     stg.raw_color
-- FROM stg_listings stg
-- JOIN makes mk ON mk.make_name = LOWER(TRIM(stg.raw_make))
-- JOIN models m ON m.make_id = mk.make_id AND m.model_name = LOWER(TRIM(stg.raw_model))
-- JOIN emirates e ON e.emirate_name = stg.clean_location
-- JOIN body_types b ON b.body_type_name = TRIM(stg.raw_body_type)
-- JOIN conditions c ON c.condition_name = stg.parsed_condition;
