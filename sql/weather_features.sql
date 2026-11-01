DROP VIEW IF EXISTS weather_features;
CREATE OR REPLACE VIEW weather_features AS 
SELECT
    airport_code,
    observation_time,
    (raw_data ->> 'temp'):: NUMERIC AS temperature_c,
    (raw_data ->> 'dewp'):: NUMERIC AS dewpoint_c,
    (raw_data ->> 'wspd'):: NUMERIC AS wind_speed_kt,
    raw_data ->> 'visib' AS visibility_raw,
    REPLACE(raw_data->>'visib', '+', '')::NUMERIC AS visibility_mi,
    (raw_data->>'visib') LIKE '%+' AS visibility_at_least,
    raw_data ->> 'fltCat' AS flight_category,
    raw_data ->> 'cover' AS cloud_cover
FROM  weather_observations;
