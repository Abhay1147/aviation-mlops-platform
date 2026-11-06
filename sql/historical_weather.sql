CREATE TABLE historical_weather (
    id BIGSERIAL PRIMARY KEY,

    airport_code VARCHAR(4) NOT NULL,
    observation_time TIMESTAMPTZ NOT NULL,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    temperature_c NUMERIC,
    wind_speed_kt NUMERIC,
    visibility_mi NUMERIC,

    metar TEXT NOT NULL,
    raw_data JSONB NOT NULL,

    CONSTRAINT uq_historical_weather_report
        UNIQUE (
            airport_code,
            observation_time,
            metar
        )
);