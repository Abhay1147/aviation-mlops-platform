CREATE OR REPLACE VIEW flight_features AS 
SELECT
    f.flight_id,
    f.flight_date,
    f.day_of_week,
    f.carrier,
    f.flight_number,
    f.origin,
    f.destination,
    f.scheduled_departure_hhmm,
    f.scheduled_arrival_hhmm,
    f.distance_miles,

    CASE
        WHEN f.scheduled_elapsed_minutes > 0
        THEN f.scheduled_elapsed_minutes
    END AS scheduled_elapsed_minutes,

    (f.scheduled_elapsed_minutes <= 0)
        AS invalid_scheduled_duration,

    o.departure_delayed_15,
    o.cancelled,
    o.diverted,

    CASE
        WHEN f.scheduled_departure_hhmm = '2400' THEN 0
        ELSE LEFT(f.scheduled_departure_hhmm, 2)::SMALLINT
    END AS scheduled_departure_hour

FROM flights AS f
JOIN flight_outcomes AS o
    ON f.flight_id = o.flight_id;
