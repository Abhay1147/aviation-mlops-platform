'''
Weather Ingestion Script
'''
import os
import json
from datetime import datetime, timezone
import psycopg2
import requests
import time

URL = "https://aviationweather.gov/api/data/metar" 

AIRPORTS = [
    "KATL", "KDEN", "KDFW", "KJFK", "KLAX",
    "KMSP", "KORD", "KPHX", "KSEA", "KSFO"
]

def fetch_weather(airports):
    params = {
        "ids": ",".join(airports),
        "format": "json"
    }

    for attempt in range(3):
        try:
            response = requests.get(
                URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            if attempt == 2:
                raise

            wait_time = 2 ** attempt
            print(f"API request failed: {error}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def save_weather(weather_data):
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname = os.getenv("DB_NAME","aviation_mlops"),
        user = os.getenv("DB_USER","aviation"),
        password=os.environ["POSTGRES_PASSWORD"]
        
    )
    inserted = 0

    try:
        with connection:
            with connection.cursor() as cursor:
                for observation in weather_data:
                    airport = observation.get("icaoId")
                    observation_time = observation.get("obsTime")

                    if not airport or observation_time is None:
                        continue
                    observation_datetime = datetime.fromtimestamp(
                        observation_time, tz = timezone.utc
                    )
                    cursor.execute(
                        """
                        INSERT INTO weather_observations
                            (airport_code, observation_time, raw_data)
                        VALUES (%s,%s,%s::jsonb)
                        ON CONFLICT (airport_code, observation_time)
                        DO NOTHING
                        """,
                        (
                            airport,
                            observation_datetime,
                            json.dumps(observation)
                            
                        )

                    )
                    inserted  += cursor.rowcount
    finally:
        connection.close()
    return inserted

def main():
    weather_data = fetch_weather(AIRPORTS)
    inserted = save_weather(weather_data)  

    print(f"number of observations recived {len(weather_data)}")
    print(f"new observations inserted inti POSTGRESQL{inserted}")

if __name__ == "__main__":
    main()
