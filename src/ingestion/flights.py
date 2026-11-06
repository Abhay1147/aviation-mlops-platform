from sqlalchemy import true
import pandas as pd 
import os
import psycopg2 
from dotenv import load_dotenv
from io import StringIO

def main():
    
    file_path = "data/raw/Aviation_flight.csv"
    
    check_connection()

    chunks = pd.read_csv(
        file_path,
        chunksize=10000,
        dtype="string",
        encoding= "utf-8-sig")
    
    total_rows = 0

    for batch_num, chunk in enumerate(chunks, start=1):
        prepared_data = prep_flights(chunk)
        flights_inserted, outcomes_inserted = save_batch(prepared_data)
        print(
            f"saved {flights_inserted:,} new flights and {outcomes_inserted:,} new outcomes"

        )

        total_rows += len(prepared_data)

        print(
            f" BATCH NUMBER {batch_num} |"
            f"PREPARED_DATA {len(prepared_data):,} | rows "
            f"total rows {total_rows}"
        )

    
def prep_flights(flight_df):

    
    flight_df["FL_DATE"] = pd.to_datetime(
        flight_df["FL_DATE"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="raise"

    )



    numeric_columns = [
        "DAY_OF_WEEK",
        "OP_CARRIER_FL_NUM",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "DEP_DELAY",
        "DEP_DELAY_NEW",
        "DEP_DELAY_GROUP",
        "ARR_DELAY",
        "ARR_DELAY_NEW",
        "ARR_DELAY_GROUP"

    ]

    for column in numeric_columns:
        flight_df[column] = pd.to_numeric(
            flight_df[column],
            errors="raise"  
        )

    boolean_columns = [
        "DEP_DEL15",
        "ARR_DEL15",
        "CANCELLED",
        "DIVERTED"
    ]

    for i in boolean_columns:
        values = pd.to_numeric(flight_df[i], errors='raise')

        if not values.dropna().isin([0,1]).all():
            raise ValueError(f"unexpected flag value in {i}")
        
        flight_df[i] = values.astype("boolean")




    required_cols = [
        "FL_DATE",
        "DAY_OF_WEEK",
        "OP_UNIQUE_CARRIER",
        "OP_CARRIER_FL_NUM",
        "ORIGIN_AIRPORT_ID",
        "ORIGIN",
        "DEST_AIRPORT_ID",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CANCELLED",
        "DIVERTED"   
    ]

    missing_req = flight_df[required_cols].isna().sum()
    missing_req = missing_req[missing_req > 0]

    if not missing_req.empty:
        raise ValueError(f"Missing required values {missing_req}")
    
    return flight_df


def check_connection():
    load_dotenv()

    connection = psycopg2.connect(
        host = "localhost",
        port = 5433,
        dbname = "aviation_mlops",
        user = "aviation",
        password = os.environ["POSTGRES_PASSWORD"]
    )

    try:

        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database();")
            database = cursor.fetchone()[0]
            print(f"connected to database: {database}")
    finally:
        connection.close()

def batch_to_csv(prepared_data):
    buffer = StringIO()

    prepared_data.to_csv(
        buffer,
        index = False,
        header = False,
        na_rep="",
        date_format="%Y-%m-%d"

    )

    buffer.seek(0)

    return buffer
def create_staging_table(cursor):
    cursor.execute("""
        CREATE TEMP TABLE flight_stage (
            day_of_week SMALLINT,
            flight_date DATE,
            carrier TEXT,
            tail_number TEXT,
            flight_number INTEGER,
            origin_airport_id INTEGER,
            origin TEXT,
            destination_airport_id INTEGER,
            destination TEXT,
            scheduled_departure_hhmm TEXT,
            actual_departure_hhmm TEXT,
            departure_delay_minutes NUMERIC,
            departure_delay_nonnegative_minutes NUMERIC,
            departure_delayed_15 BOOLEAN,
            departure_delay_group SMALLINT,
            scheduled_arrival_hhmm TEXT,
            actual_arrival_hhmm TEXT,
            arrival_delay_minutes NUMERIC,
            arrival_delay_nonnegative_minutes NUMERIC,
            arrival_delayed_15 BOOLEAN,
            arrival_delay_group SMALLINT,
            cancelled BOOLEAN,
            cancellation_code TEXT,
            diverted BOOLEAN,
            scheduled_elapsed_minutes NUMERIC,
            distance_miles NUMERIC
        ) ON COMMIT DROP;
    """)

def save_batch(prepared_data):
    load_dotenv()

    connection = psycopg2.connect(
        host = "localhost",
        port = 5433,
        dbname = "aviation_mlops",
        user = "aviation",
        password = os.environ["POSTGRES_PASSWORD"]
    )

    try:
        with connection:
            with connection.cursor() as cursor:
                create_staging_table(cursor)
                buffer = batch_to_csv(prepared_data)

                cursor.copy_expert(
                    "COPY flight_stage FROM STDIN WITH (FORMAT CSV)",
                    buffer
                )

                flights_inserted = insert_flights(cursor)
                outcomes_inserted = insert_outcomes(cursor)
        
        return flights_inserted, outcomes_inserted

    finally:
        connection.close()

def insert_flights(cursor):
    cursor.execute("""
        INSERT INTO flights (
            flight_date,
            day_of_week,
            carrier,
            flight_number,
            tail_number,
            origin_airport_id,
            origin,
            destination_airport_id,
            destination,
            scheduled_departure_hhmm,
            scheduled_arrival_hhmm,
            scheduled_elapsed_minutes,
            distance_miles
        )
        SELECT
            flight_date,
            day_of_week,
            carrier,
            flight_number,
            tail_number,
            origin_airport_id,
            origin,
            destination_airport_id,
            destination,
            scheduled_departure_hhmm,
            scheduled_arrival_hhmm,
            scheduled_elapsed_minutes,
            distance_miles
        FROM flight_stage
        ON CONFLICT (
            flight_date,
            carrier,
            flight_number,
            origin,
            destination,
            scheduled_departure_hhmm
        ) DO NOTHING;
    """)

    return cursor.rowcount

def insert_outcomes(cursor):
    cursor.execute("""
        INSERT INTO flight_outcomes (
            flight_id,
            actual_departure_hhmm,
            actual_arrival_hhmm,
            departure_delay_minutes,
            departure_delay_nonnegative_minutes,
            departure_delayed_15,
            departure_delay_group,
            arrival_delay_minutes,
            arrival_delay_nonnegative_minutes,
            arrival_delayed_15,
            arrival_delay_group,
            cancelled,
            cancellation_code,
            diverted
        )
        SELECT
            f.flight_id,
            s.actual_departure_hhmm,
            s.actual_arrival_hhmm,
            s.departure_delay_minutes,
            s.departure_delay_nonnegative_minutes,
            s.departure_delayed_15,
            s.departure_delay_group,
            s.arrival_delay_minutes,
            s.arrival_delay_nonnegative_minutes,
            s.arrival_delayed_15,
            s.arrival_delay_group,
            s.cancelled,
            s.cancellation_code,
            s.diverted
        FROM flight_stage AS s
        JOIN flights AS f
            ON f.flight_date = s.flight_date
            AND f.carrier = s.carrier
            AND f.flight_number = s.flight_number
            AND f.origin = s.origin
            AND f.destination = s.destination
            AND f.scheduled_departure_hhmm = s.scheduled_departure_hhmm
        ON CONFLICT (flight_id) DO NOTHING;
    """)

    return cursor.rowcount




if __name__ == "__main__":
    main()