-- ====================================================================
-- UAE Used-Car Market Analysis: Database Validation Suite
-- File: sql/03_validation.sql
-- ====================================================================

PRAGMA foreign_keys = ON;

-- 1. Foreign Key Integrity Check (Must return 0 rows)
PRAGMA foreign_key_check;

-- 2. Row Count Verification Across Tables
SELECT 'makes' AS table_name, COUNT(*) AS row_count FROM makes
UNION ALL
SELECT 'models', COUNT(*) FROM models
UNION ALL
SELECT 'emirates', COUNT(*) FROM emirates
UNION ALL
SELECT 'body_types', COUNT(*) FROM body_types
UNION ALL
SELECT 'conditions', COUNT(*) FROM conditions
UNION ALL
SELECT 'features', COUNT(*) FROM features
UNION ALL
SELECT 'cars', COUNT(*) FROM cars
UNION ALL
SELECT 'car_features', COUNT(*) FROM car_features
UNION ALL
SELECT 'fx_rates', COUNT(*) FROM fx_rates;

-- 3. Check for NULL Foreign Keys in Fact Table (Must be 0)
SELECT COUNT(*) AS orphaned_cars
FROM cars
WHERE model_id IS NULL 
   OR emirate_id IS NULL 
   OR body_type_id IS NULL 
   OR condition_id IS NULL;

-- 4. Check Cylinders Missingness Count in Database (Expect 155 NULLs before imputation)
SELECT 
    COUNT(*) AS total_cars,
    COUNT(cylinders) AS non_null_cylinders,
    COUNT(*) - COUNT(cylinders) AS null_cylinders
FROM cars;

-- 5. Range and Boundary Integrity
SELECT 
    MIN(model_year) AS min_year,
    MAX(model_year) AS max_year,
    MIN(price_aed) AS min_price,
    MAX(price_aed) AS max_price,
    MIN(mileage_km) AS min_mileage,
    MAX(mileage_km) AS max_mileage
FROM cars;

-- 6. Feature Links per Car Distribution (Expect 2 to 4 features per car)
SELECT 
    features_per_car,
    COUNT(*) AS car_count
FROM (
    SELECT car_id, COUNT(feature_id) AS features_per_car
    FROM car_features
    GROUP BY car_id
)
GROUP BY features_per_car
ORDER BY features_per_car;
