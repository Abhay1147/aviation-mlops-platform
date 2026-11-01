CREATE TABLE flights(
    flight_id BIGSERIAL PRIMARY KEY,


    flight_date DATE NOT NULL,

    day_of_week SMALLINT NOT NULL
        CHECK (day_of_week BETWEEN 1 AND 7),
    

    carrier VARCHAR(3) NOT NULL,
    flight_number INTEGER NOT NULL,
    tail_number TEXT,

    origin_airport_id INTEGER NOT NULL,
    origin VARCHAR(3) NOT NULL,
    destination_airport_id INTEGER NOT NULL,
    destination VARCHAR(3) NOT NULL,

    scheduled_departure_hhmm VARCHAR(4) NOT NULL,
    scheduled_arrival_hhmm VARCHAR(4) NOT NULL,
    scheduled_elapsed_minutes NUMERIC,
    distance_miles NUMERIC,


    UNIQUE (
        flight_date,
        carrier,
        flight_number,
        origin,
        destination,
        scheduled_departure_hhmm
    )
);



CREATE TABLE flight_outcomes (
    flight_id BIGINT PRIMARY KEY 
        REFERENCES flights (flight_id),

    actual_departure_hhmm VARCHAR(4),
    actual_arrival_hhmm VARCHAR(4),

    departure_delay_minutes NUMERIC,
    departure_delay_nonnegative_minutes NUMERIC,
    departure_delayed_15 BOOLEAN,
    departure_delay_group SMALLINT,

    arrival_delay_minutes NUMERIC,
    arrival_delay_nonnegative_minutes NUMERIC,
    arrival_delayed_15 BOOLEAN,
    arrival_delay_group SMALLINT,
    
    cancelled BOOLEAN NOT NULL,
    cancellation_code VARCHAR(1),
    diverted BOOLEAN NOT NULL
);
