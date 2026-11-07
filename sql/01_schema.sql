-- ====================================================================
-- UAE Used-Car Market Analysis: Relational Database Schema
-- File: sql/01_schema.sql
-- SQLite 3.25+ compliant with Foreign Keys and Window Function support
-- ====================================================================

PRAGMA foreign_keys = ON;

-- 1. Vehicle Makes Lookup
CREATE TABLE IF NOT EXISTS makes (
    make_id INTEGER PRIMARY KEY AUTOINCREMENT,
    make_name TEXT NOT NULL UNIQUE
);

-- 2. Vehicle Models (scoped to make to prevent cross-brand collisions)
CREATE TABLE IF NOT EXISTS models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    make_id INTEGER NOT NULL REFERENCES makes(make_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    UNIQUE (make_id, model_name)
);

-- 3. Listing Locations / Emirates
CREATE TABLE IF NOT EXISTS emirates (
    emirate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    emirate_name TEXT NOT NULL UNIQUE
);

-- 4. Vehicle Body Styles
CREATE TABLE IF NOT EXISTS body_types (
    body_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    body_type_name TEXT NOT NULL UNIQUE
);

-- 5. Stated Vehicle Condition
CREATE TABLE IF NOT EXISTS conditions (
    condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_name TEXT NOT NULL UNIQUE
);

-- 6. Optional Vehicle Equipment / Features
CREATE TABLE IF NOT EXISTS features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL UNIQUE
);

-- 7. Main Listings Fact Table
CREATE TABLE IF NOT EXISTS cars (
    car_id INTEGER PRIMARY KEY,  -- Corresponds to CSV 1-indexed row number
    model_id INTEGER NOT NULL REFERENCES models(model_id),
    emirate_id INTEGER NOT NULL REFERENCES emirates(emirate_id),
    body_type_id INTEGER NOT NULL REFERENCES body_types(body_type_id),
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id),
    model_year INTEGER NOT NULL CHECK (model_year BETWEEN 2000 AND 2025),
    price_aed INTEGER NOT NULL CHECK (price_aed > 0),
    mileage_km INTEGER NOT NULL CHECK (mileage_km >= 0),
    cylinders INTEGER,           -- NULL if unknown/missing in raw data
    transmission TEXT NOT NULL,
    fuel_type TEXT NOT NULL,
    color TEXT NOT NULL
);

-- 8. Many-to-Many Linking Table: Cars <-> Features
CREATE TABLE IF NOT EXISTS car_features (
    car_id INTEGER NOT NULL REFERENCES cars(car_id) ON DELETE CASCADE,
    feature_id INTEGER NOT NULL REFERENCES features(feature_id) ON DELETE CASCADE,
    PRIMARY KEY (car_id, feature_id)
);

-- 9. Foreign Exchange Rates Table
CREATE TABLE IF NOT EXISTS fx_rates (
    rate_date TEXT PRIMARY KEY,
    aed_to_vnd REAL NOT NULL
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_cars_model ON cars(model_id);
CREATE INDEX IF NOT EXISTS idx_cars_emirate ON cars(emirate_id);
CREATE INDEX IF NOT EXISTS idx_cars_year ON cars(model_year);
CREATE INDEX IF NOT EXISTS idx_cars_body_type ON cars(body_type_id);
CREATE INDEX IF NOT EXISTS idx_cars_price ON cars(price_aed);
CREATE INDEX IF NOT EXISTS idx_car_features_feat ON car_features(feature_id, car_id);

-- 10. Analytical View (Denormalized Flat Presentation)
CREATE VIEW IF NOT EXISTS v_listings AS
SELECT 
    c.car_id,
    mk.make_name,
    m.model_name,
    c.model_year,
    c.price_aed,
    ROUND(c.price_aed * fx.aed_to_vnd, 0) AS price_vnd,
    c.mileage_km,
    c.cylinders,
    c.transmission,
    c.fuel_type,
    c.color,
    b.body_type_name,
    e.emirate_name,
    cd.condition_name,
    COUNT(cf.feature_id) AS n_features,
    GROUP_CONCAT(f.feature_name, ', ') AS feature_list
FROM cars c
JOIN models m ON c.model_id = m.model_id
JOIN makes mk ON m.make_id = mk.make_id
JOIN emirates e ON c.emirate_id = e.emirate_id
JOIN body_types b ON c.body_type_id = b.body_type_id
JOIN conditions cd ON c.condition_id = cd.condition_id
LEFT JOIN car_features cf ON c.car_id = cf.car_id
LEFT JOIN features f ON cf.feature_id = f.feature_id
CROSS JOIN (SELECT aed_to_vnd FROM fx_rates ORDER BY rate_date DESC LIMIT 1) fx
GROUP BY c.car_id;
