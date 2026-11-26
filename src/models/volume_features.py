import pandas as pd

from src.models.check_schedule_volume import VOLUME_COLUMNS, monthly_counts
from src.models.train_lightgbm import DATA_DIR, load_split
from src.models.tune_threshold import prepare_features
from src.prepare_model_data import SPLITS


BASE_RUN = "20261118T013557Z_lightgbm_448d18"


def load_features(connection, split, artifact):
    print(f"Preparing {split} features...", flush=True)
    start, end = SPLITS[split]
    months = pd.date_range(start, end, freq="MS", inclusive="left")

    counts = pd.concat([
        monthly_counts(connection, month, month + pd.offsets.MonthBegin(1))
        for month in months
    ])
    metadata = pd.read_parquet(
        DATA_DIR / f"{split}.parquet",
        columns=["flight_id", "flight_date"],
    )
    aligned = counts.reindex(metadata["flight_id"].to_numpy())

    if aligned[VOLUME_COLUMNS].isna().any().any():
        raise ValueError(f"Missing schedule counts for {split} flights.")

    frame, labels = load_split(split)
    features = prepare_features(frame, artifact, "lightgbm")
    features[VOLUME_COLUMNS] = aligned[VOLUME_COLUMNS].to_numpy()

    return features, labels, metadata["flight_date"]