import pandas as pd 

def main():

    file_path = "data/raw/Aviation_flight.csv"

    flight_df = pd.read_csv(
        file_path,
        nrows=1000,
        dtype="string",
        encoding= "utf-8-sig")

    
    flight_df["FL_DATE"] = pd.to_datetime(
        flight_df["FL_DATE"],
        format="%m/%d/%Y  %I:%M:%S %p",
        errors="raise"

    )

    print("\n Column types:")
    print(flight_df[["FL_DATE","ORIGIN","DEST","CRS_DEP_TIME"]].head())

    numeric_columns = [
        "DAY_OF_WEEK",
        "OP_CARRIER_FL_NUM",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID",
        "CRS_ELAPSED_TIME",
        "DISTANCE"
    ]

    for column in numeric_columns:
        flight_df[column] = pd.to_numeric(
            flight_df[column],
            errors="raise"  
        )

    print("numeric column types: ")
    print(flight_df[numeric_columns].dtypes)
    print(flight_df[numeric_columns].head)

if __name__ == "__main__":
    main()