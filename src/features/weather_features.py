import pandas as pd

def add_weather_features(weather):
    results = weather.copy()

    results["observation_time"] = pd.to_datetime(
        results["observation_time"],
        utc=True,
        errors="raise"
    )

    source = pd.DataFrame(
        results["raw_data"].tolist(),
        index = results.index
    )

    def numeric_source(column, extra_missing=()):
        values = source[column].astype("string").str.strip()

        values = values.mask(
            values.isin(["M", "", *extra_missing])

        )

        return pd.to_numeric(
            values,
            errors="raise"
        ).astype("float64")
    
    for column in ["temperature_c","wind_speed_kt","visibility_mi"]:
        results[column] = pd.to_numeric(
            results[column], 
            errors="raise"
        ).astype("float64")


    results["dewpoint_c"] =(
        numeric_source("dwpf") - 32
    ) * 5/9

    results["relative_humidity_pct"] = numeric_source("relh")
    results["wind_gust_kt"] = numeric_source("gust")

    results["temperature_dewpoint_spread_c"] = (
        results["temperature_c"] - results["dewpoint_c"]
    )


    ceiling_candidates = []

    for layer in range(1, 5):
        coverage = (
            source[f"skyc{layer}"]
            .astype("string")
            .str.strip()
        )

        height = numeric_source(f"skyl{layer}")

        ceiling_candidates.append(
            height.where(coverage.isin(["BKN", "OVC", "VV"]))
        )

    results["reported_ceiling_ft"] = pd.concat(
        ceiling_candidates,
        axis=1
    ).min(axis=1)

    precipitation = (
        source["p01i"]
        .astype("string")
        .str.strip()
        .replace({"M": pd.NA, "": pd.NA})
    )

    results["precipitation_trace"] = precipitation.eq("T")

    results["precipitation_in"] = numeric_source(
        "p01i",
        extra_missing=("T",)
    )

    return results