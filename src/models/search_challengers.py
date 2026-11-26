import json
from datetime import datetime, timezone
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sqlalchemy import URL, create_engine
from xgboost import XGBClassifier

from src.models.evaluation import evaluate
from src.models.final_evaluation import BASE_RUN, ROOT, load_features

import os


def load_data():
    artifact = joblib.load(
        ROOT / "data" / "models" / BASE_RUN / "lightgbm.joblib"
    )

    engine = create_engine(URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER", "aviation"),
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        database=os.getenv("DB_NAME", "aviation_mlops"),

    ))

    try: 

        with engine